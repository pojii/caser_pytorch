import argparse
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import activation_getter, shuffle, minibatch, set_seed
from interactions import Interactions
from evaluation import evaluate_ranking
import pickle  # for saving/loading the model

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

class Caser(nn.Module):
    def __init__(self, num_users, num_items, model_args, muse_dim=512, precomputed_embeddings=None):
        super(Caser, self).__init__()
        self.args = model_args
        self.precomputed_embeddings = precomputed_embeddings

        L = self.args.L
        self.user_dims = model_args.d  # User embedding dimensions
        self.item_dims = muse_dim  # Sentence embedding dimensions (512)
        self.n_h = self.args.nh  # Number of horizontal convolutional filters
        self.n_v = self.args.nv  # Number of vertical convolutional filters
        self.drop_ratio = self.args.drop  # Dropout ratio
        self.ac_conv = activation_getter[self.args.ac_conv]  # Convolutional activation function
        self.ac_fc = activation_getter[self.args.ac_fc]  # Fully connected layer activation function

        # User embeddings
        self.user_embeddings = nn.Embedding(num_users, self.user_dims)

        # Vertical convolution: kernel size spans the embedding dimension (512) and uses all L
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 512))

        # Horizontal convolution: kernel size varies across the sequence dimension, but the embedding dimension is fixed at 512
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, 512)) for i in range(1, L + 1)])

        # Fully connected layer
        self.fc1_dim_v = self.n_v * 1  # Vertical output is flattened
        self.fc1_dim_h = self.n_h * L  # Horizontal output will be concatenated across all filters
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.user_dims)

        # Output layer
        self.W2 = nn.Embedding(num_items, self.user_dims + self.user_dims)
        self.b2 = nn.Embedding(num_items, 1)

        # Dropout layer
        self.dropout = nn.Dropout(self.drop_ratio)

        # Weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        # Clamp seq_var to ensure valid indices
        valid_mask = (seq_var >= 0) & (seq_var < self.precomputed_embeddings.size(0))
        seq_var = torch.where(valid_mask, seq_var, torch.tensor(0, device=seq_var.device))

        # Convert user_var and item_var to long before embedding lookup
        user_var = user_var.long()
        item_var = item_var.long()

        # Look up the precomputed embeddings using seq_var
        item_embs = self.precomputed_embeddings[seq_var]  # [batch_size, L, 512]

        # Replace out-of-bound embeddings with zeros
        item_embs = item_embs * valid_mask.unsqueeze(-1)

        # Add a channel dimension for the convolutional layers
        item_embs = item_embs.unsqueeze(1)  # [batch_size, 1, L, 512]

        user_emb = self.user_embeddings(user_var).squeeze(1)  # User embedding -> [batch_size, user_dims]

        # Vertical Convolution: Apply conv_v over the entire embedding dimension
        out_v = self.conv_v(item_embs).view(-1, self.fc1_dim_v) if self.n_v else None  # [batch_size, n_v]

        # Horizontal Convolution: Apply conv_h with varying kernel sizes along the sequence dimension
        out_hs = []
        for conv in self.conv_h:
            conv_out = self.ac_conv(conv(item_embs)).squeeze(3)  # [batch_size, n_h, L]
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch_size, n_h]
            out_hs.append(pool_out)

        out_h = torch.cat(out_hs, 1) if self.n_h else None  # Concatenate all horizontal conv outputs [batch_size, n_h * L]

        # Fully connected layer
        out = torch.cat([out_v, out_h], 1) if out_v is not None and out_h is not None else out_v or out_h
        out = self.dropout(out)
        z = self.ac_fc(self.fc1(out))

        # Combine user embeddings with fully connected layer output
        x = torch.cat([z, user_emb], 1)  # [batch_size, user_dims + user_dims]

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res

class Recommender(object):
    """
    Contains attributes and methods needed to train a sequential
    recommendation model. Models are trained by many tuples of
    (users, sequences, targets, negatives) and negatives are from negative
    sampling: for any known tuple of (user, sequence, targets), one or more
    items are randomly sampled to act as negatives.
    """

    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=True,
                 model_args=None,
                 precomputed_embeddings=None):

        # Model-related attributes
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args

        # Learning-related attributes
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda")  # Force use of CPU

        # Rank evaluation-related attributes
        self.test_sequence = None
        self._candidate = dict()

        # Precomputed embeddings
        self.precomputed_embeddings = torch.from_numpy(precomputed_embeddings).float().to(self._device)

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        # Initialize the Caser model
        self._net = Caser(self._num_users,
                        self._num_items,
                        self.model_args,
                        muse_dim=self.precomputed_embeddings.shape[1],
                        precomputed_embeddings=self.precomputed_embeddings).to(self._device)

        # Set up the optimizer
        self._optimizer = optim.Adam(self._net.parameters(),
                                    weight_decay=self._l2,
                                    lr=self._learning_rate)
        
    def initialize_model(self, interactions):
        """Initialize the model based on the given interactions data."""
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        # Initialize the Caser model
        self._net = Caser(self._num_users,
                        self._num_items,
                        self.model_args,
                        muse_dim=self.precomputed_embeddings.shape[1],
                        precomputed_embeddings=self.precomputed_embeddings).to(self._device)

        # Set up the optimizer (optional if you're not planning to train)
        self._optimizer = optim.Adam(self._net.parameters(),
                                    weight_decay=self._l2,
                                    lr=self._learning_rate)



    def fit(self, train, test, verbose=False):
        """
        The general training loop to fit the model
        """

        # Convert to sequences, targets, and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        L, T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]
        if not self._initialized:
            self._initialize(train)

        start_epoch = 0

        for epoch_num in range(start_epoch, self._n_iter):

            t1 = time()

            # Set model to training mode
            self._net.train()

            users_np, sequences_np, targets_np = shuffle(users_np, sequences_np, targets_np)

            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

            # Convert numpy arrays to PyTorch tensors and move them to the corresponding devices
            users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                                    torch.from_numpy(sequences_np).long(),
                                                    torch.from_numpy(targets_np).long(),
                                                    torch.from_numpy(negatives_np).long())

            users, sequences, targets, negatives = (users.to(self._device),
                                                    sequences.to(self._device),
                                                    targets.to(self._device),
                                                    negatives.to(self._device))

            epoch_loss = 0.0

            for minibatch_num, (batch_users, batch_sequences, batch_targets, batch_negatives) in enumerate(minibatch(users, sequences, targets, negatives, batch_size=self._batch_size)):
                items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
                items_prediction = self._net(batch_sequences, batch_users, items_to_predict)

                targets_prediction, negatives_prediction = torch.split(items_prediction, [batch_targets.size(1), batch_negatives.size(1)], dim=1)

                self._optimizer.zero_grad()
                
                # Compute the binary cross-entropy loss
                positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction)))
                negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction)))
                loss = positive_loss + negative_loss

                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            t2 = time()
            if verbose and ((epoch_num + 1) % 10 == 0 or epoch_num == 0):
                precisions, recalls, mean_aps, mean_mrr, mean_hit = evaluate_ranking(self, test, train, k=[1, 5, 10])
                print(f"Epoch {epoch_num + 1} [{t2 - t1:.1f} s]\tloss={epoch_loss:.4f}, map={float(mean_aps):.4f}, "
                    f"prec@1={np.mean(precisions[0]):.4f}, prec@5={np.mean(precisions[1]):.4f}, "
                    f"prec@10={np.mean(precisions[2]):.4f}, recall@1={np.mean(recalls[0]):.4f}, "
                    f"recall@5={np.mean(recalls[1]):.4f}, recall@10={np.mean(recalls[2]):.4f}, [{time() - t2:.1f} s]")
                print(f"MRR@10: {mean_mrr:.4f}, Hit Rate@10: {mean_hit:.4f}")
            else:
                print(f"Epoch {epoch_num + 1} [{t2 - t1:.1f} s]\tloss={epoch_loss:.4f} [{time() - t2:.1f} s]")

    def _generate_negative_samples(self, users, interactions, n):
        """
        Sample negatives from a candidate set of each user.
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[np.random.randint(len(x))]

        return negative_samples

    def predict(self, user_id, item_ids=None):
        """
        Make predictions for evaluation: given a user id, it will
        first retrieve the test sequence associated with that user
        and compute the recommendation scores for items.
        """

        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')

        # Set model to evaluation mode
        self._net.eval()
        with torch.no_grad():
            sequences_np = self.test_sequence.sequences[user_id, :]
            sequences_np = np.atleast_2d(sequences_np)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)

            sequences = torch.from_numpy(sequences_np).long().to(self._device)
            item_ids = torch.from_numpy(item_ids).long().to(self._device)
            user_id = torch.from_numpy(np.array([[user_id]])).long().to(self._device)

            out = self._net(sequences, user_id, item_ids, for_pred=True)

        return out.cpu().numpy().flatten()
    
    def save_model(self, file_path):
        """Save the model to a file."""
        model_state = {
            'model_args': self.model_args,
            'model_state_dict': self._net.state_dict(),
            'num_users': self._num_users,
            'num_items': self._num_items
        }
        with open(file_path, 'wb') as f:
            torch.save(model_state, f)
        print(f"Model saved to {file_path}")


    def load_model(self, file_path):
        """Load the model from a file."""
        with open(file_path, 'rb') as f:
            model_state = torch.load(f)
            
        self.model_args = model_state['model_args']
        self._num_users = model_state['num_users']
        self._num_items = model_state['num_items']
        
        self._net = Caser(self._num_users,
                        self._num_items,
                        self.model_args,
                        muse_dim=self.precomputed_embeddings.shape[1],
                        precomputed_embeddings=self.precomputed_embeddings).to(self._device)
        
        self._net.load_state_dict(model_state['model_state_dict'])
        print(f"Model loaded from {file_path}")


    def sample_test(self, user_id, top_k=10):
        """Test the model with a sample user, showing actual vs. predicted courses."""
        actual_courses = self.test_sequence.sequences[user_id, :]
        predicted_scores = self.predict(user_id)

        top_predictions = np.argsort(-predicted_scores)[:top_k]

        print(f"User ID: {user_id}")
        print(f"Actual Courses: {actual_courses}")
        print(f"Top {top_k} Predicted Courses: {top_predictions}")

def test_all_users(model, top_k=10, course_map=None):
    """
    Test the model with all users in the test set, showing actual vs. predicted courses.
    
    Parameters:
    - model: The trained model
    - top_k: Number of top predictions to show
    - course_map: Dictionary mapping course numbers to course names
    """
    num_users = len(model.test_sequence.sequences)
    
    for user_id in range(num_users):
        actual_courses = model.test_sequence.sequences[user_id, :]
        predicted_scores = model.predict(user_id)

        top_predictions = np.argsort(-predicted_scores)[:top_k]

        # Convert course numbers to names using the course_map
        actual_course_names = [course_map.get(course, f"Course {course}") for course in actual_courses if course in course_map]
        predicted_course_names = [course_map.get(course, f"Course {course}") for course in top_predictions if course in course_map]

        print(f"--- User ID: {user_id} ---")
        print(f"Actual Courses: {', '.join(actual_course_names) if actual_course_names else 'No actual courses found'}")
        print(f"Top {top_k} Predicted Courses: {', '.join(predicted_course_names) if predicted_course_names else 'No predicted courses found'}\n")

def load_course_map(course_map_path):
    course_map = {}
    with open(course_map_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                course_num = int(parts[0])
                course_name = parts[1]
                course_map[course_num] = course_name
    return course_map

if __name__ == '__main__':
    # Load dataset and course mappings
    train_root = 'datasets/ml1m/test/train.txt'
    test_root = 'datasets/ml1m/test/test.txt'
    course_map_path = rf'datasets/coursera/text.txt'  # Path to the course number to name mapping

    course_map = load_course_map(course_map_path)

    # Load precomputed embeddings
    precomputed_embeddings = np.load("course_embeddings.npy")

    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--train_root', type=str, default='datasets/ml1m/test/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/ml1m/test/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # Train arguments
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=bool, default=True)  # Set to False to use CPU

    config = parser.parse_args()

    # Model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')

    model_config = model_parser.parse_args()
    model_config.L = config.L

    # Set seed
    set_seed(config.seed, cuda=config.use_cuda)

    # Load dataset
    train = Interactions(config.train_root)
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root, user_map=train.user_map, item_map=train.item_map)

    # Load precomputed embeddings
    precomputed_embeddings = np.load("course_embeddings.npy")

    print(config)
    print(model_config)

    # Fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        model_args=model_config,
                        use_cuda=config.use_cuda,
                        precomputed_embeddings=precomputed_embeddings)

    # Fit model
    model.fit(train, test, verbose=True)

    # Save the trained model
    model.save_model('trained_caser_model.pkl')


    # Sample test with a specific user
    # sample_user_id = 0  # Example user_id to test
    # model.sample_test(user_id=sample_user_id, top_k=10)

    # Load the model and test again to ensure everything works
    model.load_model('trained_caser_model.pkl')
    # Test all users and show actual vs. predicted courses
    test_all_users(model, top_k=10, course_map=course_map)