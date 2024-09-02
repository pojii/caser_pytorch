import argparse
from time import time

import torch.optim as optim
from torch.autograd import Variable

from caser import Caser
from evaluation import evaluate_ranking
from interactions import Interactions
from utils import *

import numpy as np

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
                 use_cuda=False,
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
        self._device = torch.device("cuda" if use_cuda else "cpu")

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

        self._net = Caser(self._num_users,
                          self._num_items,
                          self.model_args,
                          muse_dim=self.precomputed_embeddings.shape[1],
                          precomputed_embeddings=self.precomputed_embeddings).to(self._device)

        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def fit(self, train, test, verbose=False):
        """
        The general training loop to fit the model without using minibatches
        """
        # Convert to sequences, targets, and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        L, T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]

        print(f'Total training instances: {n_train}')

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
            users = torch.from_numpy(users_np).long().to(self._device)
            sequences = torch.from_numpy(sequences_np).long().to(self._device)
            targets = torch.from_numpy(targets_np).long().to(self._device)
            negatives = torch.from_numpy(negatives_np).long().to(self._device)

            items_to_predict = torch.cat((targets, negatives), 1)
            items_prediction = self._net(sequences, users, items_to_predict)

            targets_prediction, negatives_prediction = torch.split(items_prediction, [targets.size(1), negatives.size(1)], dim=1)

            self._optimizer.zero_grad()
            
            # Compute the binary cross-entropy loss
            positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction)))
            negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction)))
            loss = positive_loss + negative_loss

            loss.backward()
            self._optimizer.step()

            epoch_loss = loss.item()

            t2 = time()
            if verbose and (epoch_num + 1) % 10 == 0:
                precision, recall, mean_aps, mean_mrr, mean_hit = evaluate_ranking(self, test, train, k=[1, 5, 10])
                print(f"Epoch {epoch_num + 1} [{t2 - t1:.1f} s]\t"
                    f"loss={epoch_loss:.4f}, MAP={mean_aps:.4f}, MRR@10={mean_mrr:.4f}, HR@10={mean_hit:.4f}\n"
                    f"Precision: @1={np.mean(precision[0]):.4f}, @5={np.mean(precision[1]):.4f}, @10={np.mean(precision[2]):.4f}\n"
                    f"Recall: @1={np.mean(recall[0]):.4f}, @5={np.mean(recall[1]):.4f}, @10={np.mean(recall[2]):.4f}\n"
                    f"[{time() - t2:.1f} s]")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--train_root', type=str, default='datasets/coursera/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/coursera/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # Train arguments
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)

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
    sequences = train.sequences.sequences
    targets = train.sequences.targets

    # Sample and print 10 random sequences and targets
    num_samples = min(10, len(sequences))  # In case there are fewer than 10 sequences
    sample_indices = np.random.choice(len(sequences), num_samples, replace=False)
    
    print(f"Displaying {num_samples} randomly sampled sequences and targets:")
    for i, idx in enumerate(sample_indices):
        print(f"Sample {i+1}:")
        print(f"Sequence: {sequences[idx]}")
        print(f"Target: {targets[idx]}")
        print()

    test = Interactions(config.test_root, user_map=train.user_map, item_map=train.item_map)

    # Load precomputed embeddings
    precomputed_embeddings = np.load("precomputed_embeddings.npy")

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

    model.fit(train, test, verbose=True)
