import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import time

# Assuming these are custom modules you have in your project
from interactions import Interactions
from evaluation import evaluate_ranking
from utils import set_seed, shuffle, str2bool
from caser import Caser

class Recommender(object):
    def __init__(self, n_iter, batch_size, learning_rate, l2, neg_samples, model_args, use_cuda, precomputed_embeddings):
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.precomputed_embeddings = torch.from_numpy(precomputed_embeddings).float().to(self._device)
        self._initialized = False

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users
        self.test_sequence = interactions.test_sequences

        # Assume the model_args have the necessary parameters
        self._net = Caser(self._num_users, self._num_items, self.model_args).to(self._device)

        self._optimizer = optim.Adam(self._net.parameters(), weight_decay=self._l2, lr=self._learning_rate)
        self._initialized = True

    def fit(self, train, test, verbose=False):
        if not self._initialized:
            self._initialize(train)

        # Convert to sequences, targets, and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        L, T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]

        print(f'Total training instances: {n_train}')

        for epoch_num in range(self._n_iter):
            t1 = time()

            # Set model to training mode
            self._net.train()

            users_np, sequences_np, targets_np = shuffle(users_np, sequences_np, targets_np)

            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

            # Convert numpy arrays to PyTorch tensors and move them to the corresponding devices
            users = torch.from_numpy(users_np).long().to(self._device)
            sequences = torch.from_numpy(sequences_np).long().to(self._device)
            targets = torch.from_numpy(targets_np).long().to(self._device)
            negatives = torch.from_numpy(negatives_np).long().to(self._device)

            items_to_predict = torch.cat((targets, negatives), 1)
            items_prediction = self._net(sequences, users, items_to_predict)

            targets_prediction = items_prediction[:, :targets.size(1)]
            negatives_prediction = items_prediction[:, targets.size(1):]

            # Compute the binary cross-entropy loss
            positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction) + 1e-8))
            negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction) + 1e-8))
            loss = positive_loss + negative_loss

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            t2 = time()

            # if verbose and (epoch_num + 1) % 10 == 0:
            #     precision, recall, mean_aps, mrr, ndcg = evaluate_ranking(self, test, train, k=[1, 5, 10])
            #     print(f"Epoch {epoch_num + 1}/{self._n_iter} [{t2 - t1:.2f}s]\tloss={loss.item():.4f}" + 
            #           f"\tPrec@1={precision[0]:.4f}\tPrec@5={precision[1]:.4f}\tPrec@10={precision[2]:.4f}" +
            #           f"\tRecall@1={recall[0]:.4f}\tRecall@5={recall[1]:.4f}\tRecall@10={recall[2]:.4f}" +
            #           f"\tMAP={mean_aps:.4f}\tMRR={mrr:.4f}\tNDCG={ndcg:.4f}")
            # else:
            print(f"Epoch {epoch_num + 1}/{self._n_iter} [{t2 - t1:.2f}s]\tloss={loss.item():.4f}")


    def _generate_negative_samples(self, users, interactions, n):
        """
        Sample negative items for each user.
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not hasattr(self, '_candidate'):
            self._candidate = {}
            for u, row in enumerate(interactions.tocsr()):
                self._candidate[u] = list(set(np.arange(interactions.num_items)) - set(row.indices))

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

        self._net.eval()

        with torch.no_grad():
            sequences_np = self.test_sequence.sequences[user_id, :]
            sequences_np = np.atleast_2d(sequences_np)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1)

            sequences = torch.from_numpy(sequences_np).long().to(self._device)
            item_ids = torch.from_numpy(item_ids).long().to(self._device)
            user_id = torch.from_numpy(np.array([[user_id]])).long().to(self._device)

            # Reshape inputs to match the expected shapes
            sequences = sequences.unsqueeze(0)  # Add batch dimension
            user_id = user_id.squeeze(1)  # Remove unnecessary dimension

            out = self._net(sequences, user_id, item_ids, for_pred=True)

        return out.cpu().numpy().flatten()

    def load_pretrained_model(self, path):
        if self._net is None:
            raise ValueError("Model is not initialized. Please call the _initialize method first.")
        
        pretrained_dict = torch.load(path, map_location=self._device)
        model_dict = self._net.state_dict()

        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # Load the new state dict into the model
        self._net.load_state_dict(model_dict)

    def save_model(self, path):
        torch.save(self._net.state_dict(), path)
        print(f"Model saved to {path}")

def train_model(model, train_data, test_data, config, pretrained_model_path=None, is_pretrain=True):
    print(f"{'Pretraining' if is_pretrain else 'Fine-tuning'} the model...")
    print(f"Number of users: {train_data.num_users}")
    print(f"Number of items: {train_data.num_items}")
    print(f"Number of interactions: {len(train_data.sequences.sequences)}")
    
    # เรียกใช้ _initialize ก่อนการโหลดโมเดล
    if not model._initialized:
        model._initialize(train_data)

    if not is_pretrain and pretrained_model_path:
        model.load_pretrained_model(pretrained_model_path)
    
    model.fit(train_data, test_data, verbose=True)
    
    if is_pretrain:
        model.save_model('edx_pretrained_model.pth')
    else:
        model.save_model('coursera_finetuned_model.pth')

    # Perform final evaluation
    print("Performing final evaluation...")
    precision, recall, mean_aps, mrr, ndcg = evaluate_ranking(model, test_data, train_data, k=[1, 5, 10])
    print(f"Final results:")
    print(f"Precision: @1={precision[0].mean():.4f}, @5={precision[1].mean():.4f}, @10={precision[2].mean():.4f}")
    print(f"Recall: @1={recall[0].mean():.4f}, @5={recall[1].mean():.4f}, @10={recall[2].mean():.4f}")
    print(f"MAP={mean_aps:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--edx_train_root', type=str, default='datasets/edx/train.txt')
    parser.add_argument('--edx_test_root', type=str, default='datasets/edx/test.txt')
    parser.add_argument('--coursera_train_root', type=str, default='datasets/coursera/train.txt')
    parser.add_argument('--coursera_test_root', type=str, default='datasets/coursera/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # Train arguments
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)

    config = parser.parse_args()

    # Model arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=512)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')

    model_config = model_parser.parse_args([])
    model_config.L = config.L

    # Set seed for reproducibility
    set_seed(config.seed, config.use_cuda)

    # Load edX dataset
    edx_train = Interactions(config.edx_train_root)
    edx_train.to_sequence(config.L, config.T)
    edx_test = Interactions(config.edx_test_root, user_map=edx_train.user_map, item_map=edx_train.item_map)

    # Load Coursera dataset
    coursera_train = Interactions(config.coursera_train_root)
    coursera_train.to_sequence(config.L, config.T)
    coursera_test = Interactions(config.coursera_test_root, user_map=coursera_train.user_map, item_map=coursera_train.item_map)

    # Load precomputed embeddings for edX
    edx_precomputed_embeddings = np.load("datasets/edx/precomputed_embeddings.npy")

    print("Configuration:")
    print(config)
    print("Model configuration:")
    print(model_config)

    # Create and pretrain the model on edX data
    edx_model = Recommender(n_iter=config.n_iter,
                            batch_size=config.batch_size,
                            learning_rate=config.learning_rate,
                            l2=config.l2,
                            neg_samples=config.neg_samples,
                            model_args=model_config,
                            use_cuda=config.use_cuda,
                            precomputed_embeddings=edx_precomputed_embeddings)

    train_model(edx_model, edx_train, edx_test, config, is_pretrain=True)

    # Load precomputed embeddings for Coursera
    coursera_precomputed_embeddings = np.load("datasets/coursera/precomputed_embeddings.npy")

    # Create a new model for Coursera, initialize with pretrained weights
    coursera_model = Recommender(n_iter=config.n_iter,
                                 batch_size=config.batch_size,
                                 learning_rate=config.learning_rate,
                                 l2=config.l2,
                                 neg_samples=config.neg_samples,
                                 model_args=model_config,
                                 use_cuda=config.use_cuda,
                                 precomputed_embeddings=coursera_precomputed_embeddings)

    # Fine-tune on Coursera data
    train_model(coursera_model, coursera_train, coursera_test, config, pretrained_model_path='edx_pretrained_model.pth', is_pretrain=False)

    print("Pretraining and fine-tuning completed successfully.")