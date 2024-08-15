import argparse
from time import time

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from caser import Caser
from evaluation import evaluate_ranking
from interactions import Interactions
from utils import *

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import copy
import tensorflow_hub as hub
import tensorflow_text

# โหลด Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Recommender:
    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=False,
                 model_args=None):

        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args

        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        # self._device = torch.device("cpu")  # Always use CPU
        self.test_sequence = None
        self._candidate = dict()
        self.best_weights = None
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.embed = embed

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_users,
                          self._num_items,
                          self.model_args).to(self._device)

        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def fit(self, train, test, verbose=False):
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        L, T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]
        print(f'Total training instances: {n_train}')

        if not self._initialized:
            self._initialize(train)

        train_losses = []
        val_losses = []
        early_stopping = EarlyStopping(patience=20, min_delta=0.001)

        for epoch_num in range(self._n_iter):
            t1 = time()
            
            # Training
            self._net.train()
            epoch_loss = self._train_epoch(train)
            train_losses.append(epoch_loss)

            t2 = time()

            # Validation
            self._net.eval()
            with torch.no_grad():
                val_loss = self._compute_validation_loss(test)
            val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = copy.deepcopy(self._net.state_dict())
                self.best_epoch = epoch_num + 1

            if verbose:
                precision, recall, mean_aps = evaluate_ranking(self, test, train, k=[1, 5, 10])
                output_str = f"Epoch {epoch_num + 1} [{t2 - t1:.1f} s]\ttrain_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, map={mean_aps:.4f}, " \
                             f"prec@1={np.mean(precision[0]):.4f}, prec@5={np.mean(precision[1]):.4f}, prec@10={np.mean(precision[2]):.4f}, " \
                             f"recall@1={np.mean(recall[0]):.4f}, recall@5={np.mean(recall[1]):.4f}, recall@10={np.mean(recall[2]):.4f}, " \
                             f"[{time() - t2:.1f} s]"
                print(output_str)
            else:
                print(f"Epoch {epoch_num + 1} [{t2 - t1:.1f} s]\ttrain_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

            if early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch_num + 1}")
                break

        print(f"Best model was from epoch {self.best_epoch} with validation loss {self.best_val_loss:.4f}")
        self._net.load_state_dict(self.best_weights)
        self._plot_losses(train_losses, val_losses)

    def _plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig('loss_plot.png')
        plt.close()  # Close the figure to free up memory

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

        if user_id < 0 or user_id >= self.test_sequence.sequences.shape[0]:
            print(f"Warning: user_id {user_id} is out of bounds!")
            return np.zeros(self._num_items)

        self._net.eval()
        with torch.no_grad():
            sequences_np = self.test_sequence.sequences[user_id, :]
            sequences_np = np.atleast_2d(sequences_np)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)

            sequences = torch.from_numpy(sequences_np).long()
            item_ids = torch.from_numpy(item_ids).long()
            user_id = torch.tensor([[user_id]]).long()

            user = user_id.to(self._device)
            sequences = sequences.to(self._device)
            items = item_ids.to(self._device)

            try:
                out = self._net(sequences, user, items, for_pred=True)
                return out.cpu().numpy().flatten()
            except RuntimeError as e:
                print(f"Error in predict: {e}")
                return np.zeros(self._num_items)

    def _compute_validation_loss(self, test):
        if not hasattr(test, 'sequences') or test.sequences is None:
            print("Warning: Test data does not have sequence information. Skipping validation loss computation.")
            return None

        try:
            sequences_np = test.sequences.sequences
            users_np = test.sequences.user_ids.reshape(-1, 1)
            targets_np = test.sequences.targets

            # Convert numpy arrays to PyTorch tensors
            users = torch.from_numpy(users_np).long()
            sequences = torch.from_numpy(sequences_np).long()
            targets = torch.from_numpy(targets_np).long()

            # Check if any index is out of range
            max_item_id = self._num_items - 1
            if sequences.max() > max_item_id or targets.max() > max_item_id:
                print(f"Warning: Some item IDs in the test set are larger than the number of items in the training set ({self._num_items}).")
                print(f"Max item ID in sequences: {sequences.max().item()}")
                print(f"Max item ID in targets: {targets.max().item()}")
                
                # Clip the values to be within the valid range
                sequences = torch.clamp(sequences, max=max_item_id)
                targets = torch.clamp(targets, max=max_item_id)

            items_to_predict = targets
            items_prediction = self._net(sequences, users, items_to_predict)

            loss = -torch.mean(torch.log(torch.sigmoid(items_prediction) + 1e-8))

            return loss.item()
        except RuntimeError as e:
            print(f"Error in _compute_validation_loss: {e}")
            return None
        
    def _train_epoch(self, train):
        def data_generator(sequences, batch_size):
            users = sequences.user_ids
            sequence_data = sequences.sequences
            targets = sequences.targets
            num_samples = len(users)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                yield users[batch_indices], sequence_data[batch_indices], targets[batch_indices]

        from torch.nn.utils.rnn import pad_sequence
        epoch_loss = 0.0
        num_batches = 0
        accumulation_steps = 4  # Adjust as needed
        accumulated_loss = 0.0

        for batch_users, batch_sequences, batch_targets in data_generator(train.sequences, self._batch_size):
            # Prepare batch data
            batch_sequences_text = [' '.join(map(str, seq)) for seq in batch_sequences]
            batch_sequences_emb = self.embed(batch_sequences_text).numpy()
            
            # Convert to list of tensors
            batch_sequences_emb = [torch.from_numpy(emb).float() for emb in batch_sequences_emb]
            
            # Pad sequences
            batch_sequences_emb_padded = pad_sequence(batch_sequences_emb, batch_first=True, padding_value=0)
            if batch_sequences_emb_padded.dim() != 3:
                print("Unexpected shape of batch_sequences_emb_padded:", batch_sequences_emb_padded.shape)
                # If it's 2D, add a dimension
                if batch_sequences_emb_padded.dim() == 2:
                    batch_sequences_emb_padded = batch_sequences_emb_padded.unsqueeze(1)
                print("Adjusted shape:", batch_sequences_emb_padded.shape)
            
            batch_negatives = self._generate_negative_samples(batch_users, train, n=self._neg_samples)

            # Convert to PyTorch tensors and move to device
            batch_sequences_emb_padded = batch_sequences_emb_padded.to(self._device)
            batch_users = torch.from_numpy(batch_users).long().to(self._device)
            batch_targets = torch.from_numpy(batch_targets).long().to(self._device)
            batch_negatives = torch.from_numpy(batch_negatives).long().to(self._device)

            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)

            # Use half precision
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                items_prediction = self._net(batch_sequences_emb_padded, batch_users, items_to_predict)
                targets_prediction, negatives_prediction = torch.split(items_prediction, 
                                                                    [batch_targets.shape[1], 
                                                                        batch_negatives.shape[1]], dim=1)

                positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction) + 1e-8))
                negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction) + 1e-8))
                loss = positive_loss + negative_loss

            # Normalize the loss and backward
            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            if (num_batches + 1) % accumulation_steps == 0:
                self.scaler.step(self._optimizer)
                self.scaler.update()
                self._optimizer.zero_grad()
                epoch_loss += accumulated_loss
                accumulated_loss = 0.0

            # Free up memory
            del batch_sequences_emb_padded, batch_users, batch_targets, batch_negatives, items_to_predict, items_prediction
            torch.cuda.empty_cache()

            num_batches += 1

        return epoch_loss / num_batches

    def get_best_weights(self):
        return self.best_weights

def prepare_sequences(interactions, L, T):
    sequences = interactions.sequences.sequences
    targets = interactions.sequences.targets
    users = interactions.sequences.user_ids

    # แปลง sequences เป็นข้อความ
    sequences_text = [' '.join(map(str, seq[-L:])) for seq in sequences]  # ใช้เฉพาะ L ตัวสุดท้าย
    
    # ใช้ Universal Sentence Encoder
    sequences_emb = embed(sequences_text).numpy()
    
    return sequences_emb, targets, users

if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, default='datasets/coursera/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/coursera/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=64)
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
    train.to_sequence(config.L, config.T)
    train.sequences.sequences, train.sequences.targets, train.sequences.user_ids = prepare_sequences(train, config.L, config.T)

    test = Interactions(config.test_root, user_map=train.user_map, item_map=train.item_map)
    test.to_sequence(config.L, config.T)
    test.sequences.sequences, test.sequences.targets, test.sequences.user_ids = prepare_sequences(test, config.L, config.T)

    print(config)
    print(model_config)
    print(f"Number of users in training set: {train.num_users}")
    print(f"Number of items in training set: {train.num_items}")
    print(f"Max user ID in training set: {train.user_ids.max()}")
    print(f"Max item ID in training set: {train.item_ids.max()}")
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        model_args=model_config,
                        use_cuda=config.use_cuda)
    print(f"Max user ID in test set: {test.user_ids.max()}")
    print(f"Max item ID in test set: {test.item_ids.max()}")
    model.fit(train, test, verbose=True)