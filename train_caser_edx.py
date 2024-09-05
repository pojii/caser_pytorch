import argparse
from time import time
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from caser import Caser
from evaluation import evaluate_ranking
from interactions import Interactions
from utils import set_seed, shuffle

import pandas as pd

def save_model(model, file_path):
        """
        Save the trained model to a file.
        
        Parameters
        ----------
        model: Recommender object
            The trained recommender model to save.
        file_path: str
            The file path where the model should be saved.
        """
        torch.save({
            'model_state_dict': model._net.state_dict(),
            'optimizer_state_dict': model._optimizer.state_dict(),
            'num_users': model._num_users,
            'num_items': model._num_items,
            'model_args': model.model_args,
        }, file_path)
        print(f"Model saved to {file_path}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def sample_test(model, user_id, top_k=10, course_map=None):
    """Test the model with a sample user, showing actual vs. predicted courses."""
    actual_courses = model.test_sequence.sequences[user_id, :]
    predicted_scores = model.predict(user_id)

    top_predictions = np.argsort(-predicted_scores)[:top_k]

    # Convert course numbers to names using the course_map
    actual_course_names = [course_map[course] for course in actual_courses if course in course_map]
    predicted_course_names = [course_map[course] for course in top_predictions if course in course_map]

    print(f"User ID: {user_id}")
    print(f"Actual Courses: {actual_course_names}")
    print(f"Top {top_k} Predicted Courses: {predicted_course_names}")

def evaluate_model(model, test_interactions, k_list=[10]):
    """Evaluate the model and print metrics."""
    precisions, recalls, mean_aps, mean_mrr, mean_hit = model.evaluate(test_interactions, k_list)
    
    for k, precision, recall in zip(k_list, precisions, recalls):
        print(f"Precision@{k}: {np.mean(precision):.4f}")
        print(f"Recall@{k}: {np.mean(recall):.4f}")
    
    print(f"Mean Average Precision: {mean_aps:.4f}")
    print(f"Mean Reciprocal Rank: {mean_mrr:.4f}")
    print(f"Hit Rate: {mean_hit:.4f}")

class Recommender(object):
    def evaluate(self, interactions, k_list):
        """
        Evaluate the model on the given interactions.
        
        Parameters
        ----------
        interactions: Interactions object
            The interactions on which to evaluate the model.
        k_list: list of int
            List of k values to compute Precision@K and Recall@K.

        Returns
        -------
        precisions: list of np.array
            Precision@K for each k in k_list.
        recalls: list of np.array
            Recall@K for each k in k_list.
        mean_aps: float
            Mean Average Precision.
        mean_mrr: float
            Mean Reciprocal Rank.
        mean_hit: float
            Hit Rate.
        """
        precisions = []
        recalls = []
        apks = []
        mrrs = []
        hits = []

        for k in k_list:
            precision_k = []
            recall_k = []

            for user_id in range(len(interactions.sequences.user_ids)):
                user_sequence = interactions.sequences.sequences[user_id, :]
                user_target = interactions.sequences.targets[user_id, :]

                predictions = self.predict(user_id)

                top_k_predictions = np.argsort(-predictions)[:k]

                num_hits = sum([1 for i in top_k_predictions if i in user_target])
                precision_k.append(num_hits / k)
                recall_k.append(num_hits / len(user_target))

                if k == 10:
                    apks.append(self._compute_apk(user_target, top_k_predictions))
                    mrrs.append(self._compute_mrr(user_target, top_k_predictions))
                    hits.append(self._compute_hit_rate(user_target, top_k_predictions))

            precisions.append(np.mean(precision_k))
            recalls.append(np.mean(recall_k))

        mean_aps = np.mean(apks)
        mean_mrr = np.mean(mrrs)
        mean_hit = np.mean(hits)

        return precisions, recalls, mean_aps, mean_mrr, mean_hit


    def _compute_apk(self, actual, predicted, k=10):
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(actual), k)

    def _compute_mrr(self, actual, predicted, k=10):
        for i, p in enumerate(predicted[:k]):
            if p in actual:
                return 1.0 / (i + 1)
        return 0.0

    def _compute_hit_rate(self, actual, predicted, k=10):
        return 1.0 if any(p in actual for p in predicted[:k]) else 0.0
    
    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=False,
                 model_args=None,
                 precomputed_embeddings=None):

        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self.test_sequence = None
        self._candidate = dict()
        self.precomputed_embeddings = torch.from_numpy(precomputed_embeddings).float().to(self._device)

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users
        self.test_sequence = interactions.test_sequences
        self._net = Caser(self._num_users,
                          self._num_items,
                          self.model_args,
                          muse_dim=self.precomputed_embeddings.shape[1],
                          precomputed_embeddings=self.precomputed_embeddings).to(self._device)
        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def fit(self, train, val, test, verbose=False):
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)
        L, T = train.sequences.L, train.sequences.T
        n_train = sequences_np.shape[0]
        if not self._initialized:
            self._initialize(train)
        start_epoch = 0
        best_val_loss = float('inf')
        patience = 10
        wait = 0

        for epoch_num in range(start_epoch, self._n_iter):
            t1 = time()
            self._net.train()
            users_np, sequences_np, targets_np = shuffle(users_np, sequences_np, targets_np)
            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)
            users = torch.from_numpy(users_np).long().to(self._device)
            sequences = torch.from_numpy(sequences_np).long().to(self._device)
            targets = torch.from_numpy(targets_np).long().to(self._device)
            negatives = torch.from_numpy(negatives_np).long().to(self._device)

            items_to_predict = torch.cat((targets, negatives), 1)
            items_prediction = self._net(sequences, users, items_to_predict)

            targets_prediction, negatives_prediction = torch.split(items_prediction, [targets.size(1), negatives.size(1)], dim=1)
            self._optimizer.zero_grad()

            positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction)))
            negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction)))
            loss = positive_loss + negative_loss
            loss.backward()
            self._optimizer.step()

            epoch_loss = loss.item()

            val_loss = self._evaluate_loss(val)

            t2 = time()
            print(f"Epoch {epoch_num + 1} [{t2 - t1:.1f} s]\tloss={epoch_loss:.4f} val_loss={val_loss:.4f} [{time() - t2:.1f} s]")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

        # Evaluate metrics after training
        print("Evaluating model on test data...")
        evaluate_model(self, test)

    def _evaluate_loss(self, val):
        self._net.eval()
        sequences_np = val.sequences.sequences
        targets_np = val.sequences.targets
        users_np = val.sequences.user_ids.reshape(-1, 1)
        negatives_np = self._generate_negative_samples(users_np, val, n=self._neg_samples)

        users = torch.from_numpy(users_np).long().to(self._device)
        sequences = torch.from_numpy(sequences_np).long().to(self._device)
        targets = torch.from_numpy(targets_np).long().to(self._device)
        negatives = torch.from_numpy(negatives_np).long().to(self._device)

        items_to_predict = torch.cat((targets, negatives), 1)
        with torch.no_grad():
            items_prediction = self._net(sequences, users, items_to_predict)
            targets_prediction, negatives_prediction = torch.split(items_prediction, [targets.size(1), negatives.size(1)], dim=1)
            positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction)))
            negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction)))
            val_loss = positive_loss + negative_loss

        return val_loss.item()

    def _generate_negative_samples(self, users, interactions, n):
        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))
        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[np.random.randint(len(x))]
        return negative_samples

    def predict(self, user_id, item_ids=None):
        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')
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
    
    def load_model(self, model_path, interactions):
        """
        Load the model's state from the given checkpoint.

        Parameters
        ----------
        model_path : str
            Path to the model checkpoint file.
        interactions : Interactions object
            The interactions object used to initialize the model.
        """
        self._initialize(interactions)

        # Load the checkpoint
        checkpoint = torch.load(model_path)
        
        # Extract the state_dict that contains the model's parameters
        state_dict = checkpoint['model_state_dict']
        
        # Load the model's state dict
        self._net.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")

        # If you want to load the optimizer state as well
        if 'optimizer_state_dict' in checkpoint:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, default='datasets/edx/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/edx/test.txt')
    parser.add_argument('--val_root', type=str, default='datasets/edx/val.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)

    config = parser.parse_args()

    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')
    model_config = model_parser.parse_args()
    model_config.L = config.L

    set_seed(config.seed, cuda=config.use_cuda)

    train = Interactions(config.train_root)
    val = Interactions(config.val_root)
    test = Interactions(config.test_root, user_map=train.user_map, item_map=train.item_map)

    # Generate sequences using to_sequence method
    train.to_sequence(sequence_length=config.L, target_length=config.T)
    val.to_sequence(sequence_length=config.L, target_length=config.T)
    test.to_sequence(sequence_length=config.L, target_length=config.T)

    # Load embeddings from course_embeddings.csv
    embeddings_df = pd.read_csv("datasets/edx/course_embeddings.csv")
    precomputed_embeddings = embeddings_df.iloc[:, 2:].values  # Assuming the first two columns are course_id and course_name

    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        model_args=model_config,
                        use_cuda=config.use_cuda,
                        precomputed_embeddings=precomputed_embeddings)

    model.fit(train, val, test, verbose=True)
    save_model(model, 'edx_caser.pkl')

