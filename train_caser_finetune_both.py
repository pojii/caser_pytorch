import argparse
import torch
import torch.optim as optim
import numpy as np
from time import time

# Assuming these are custom modules you have in your project
from interactions import Interactions
from evaluation_without_user import evaluate_ranking
from utils import set_seed, shuffle, str2bool
from caser_without_user_emb import Caser
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Recommender:
    def __init__(self, n_iter, batch_size, learning_rate, l2, neg_samples, model_args, use_cuda, precomputed_embeddings, patience=50):
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
        self.precomputed_embeddings = torch.from_numpy(precomputed_embeddings).float()
        if self._device.type == 'cuda':
            try:
                self.precomputed_embeddings = self.precomputed_embeddings.to(self._device)
            except RuntimeError as e:
                print(f"Error moving embeddings to GPU: {e}")
                print("Falling back to CPU")
                self._device = torch.device('cpu')
        self._initialized = False
        self.patience = patience
        self.train_losses = []  # To store training losses
        self.val_losses = []    # To store validation losses
        self.scheduler = None  # สร้างที่นี่และกำหนดค่าทีหลังใน _initialize()
        self.learning_rates = []  # To store learning rates

    def _initialize(self, interactions):
        """
        Initialize the network and optimizer based on the given interactions.
        """
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users
        self.test_sequence = interactions.test_sequences

        # Initialize the Caser model
        self._net = Caser(self._num_users, self._num_items, self.model_args).to(self._device)

        # Initialize the optimizer
        self._optimizer = optim.Adam(self._net.parameters(), weight_decay=self._l2, lr=self._learning_rate)
        
        # **Initialize ReduceLROnPlateau scheduler**
        self.scheduler = ReduceLROnPlateau(self._optimizer, mode='min', factor=0.5, patience=1, verbose=True)

        self._initialized = True

    def fit(self, train, val, verbose=False):
        if not self._initialized:
            self._initialize(train)

        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets

        n_train = sequences_np.shape[0]
        print(f'Total training instances: {n_train}')

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch_num in range(self._n_iter):
            start_time = time()

            self._net.train()

            sequences_np, targets_np = shuffle(sequences_np, targets_np)

            negatives_np = self._generate_negative_samples(train, n=self._neg_samples)

            sequences = torch.from_numpy(sequences_np).long().to(self._device)
            targets = torch.from_numpy(targets_np).long().to(self._device)
            negatives = torch.from_numpy(negatives_np).long().to(self._device)

            items_to_predict = torch.cat((targets, negatives), 1)
            items_prediction = self._net(sequences, items_to_predict)

            targets_prediction = items_prediction[:, :targets.size(1)]
            negatives_prediction = items_prediction[:, targets.size(1):]

            # positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction) + 1e-8))
            # negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction) + 1e-8))
            # loss = positive_loss + negative_loss
            
            # **BPR Loss calculation**
            # Compute the difference between targets (positive samples) and negatives (negative samples)
            loss_diff = targets_prediction - negatives_prediction

            # Apply the BPR Loss using log-sigmoid
            loss = -torch.mean(torch.log(torch.sigmoid(loss_diff) + 1e-8))

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            end_time = time()
            print(f"Epoch {epoch_num + 1}\tloss={loss.item():.4f} [{end_time - start_time:.2f}s]")

            # Save the training loss
            self.train_losses.append(loss.item())

            # Validate the model using validation data (val.txt)
            val_loss = self._evaluate_loss(val)
            print(f"Validation loss: {val_loss:.4f}")

            # Save the validation loss
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            
            # เก็บค่า learning rate และปริ้นออกมา
            current_lr = self.scheduler.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            print(f"Current learning rate: {current_lr}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print("Validation loss improved, saving the model...")
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience counter: {patience_counter}/{self.patience}")

            if patience_counter >= self.patience:
                print("Early stopping triggered. Stopping training.")
                break

    def _generate_negative_samples(self, interactions, n):
        num_sequences = interactions.sequences.sequences.shape[0]
        # ตรวจสอบและปรับค่า self._num_items ถ้าจำเป็น
        self._num_items = max(self._num_items, interactions.num_items)
        # สร้าง candidate items
        all_items = set(range(self._num_items))
        interacted_items = set(interactions.item_ids)
        self._candidate = list(all_items - interacted_items)
        if len(self._candidate) == 0:
            # print("Warning: All items have been interacted with. Using a fallback method for negative sampling.")
            # ใช้วิธี fallback, เช่น สุ่มจากทุก items ยกเว้น item ที่กำลังพิจารณา
            negative_samples = np.zeros((num_sequences, n), np.int64)
            for i in range(num_sequences):
                current_items = set(interactions.sequences.sequences[i])
                current_candidates = list(all_items - current_items)
                negative_samples[i] = np.random.choice(current_candidates, size=n, replace=len(current_candidates) < n)
        else:
            negative_samples = np.zeros((num_sequences, n), np.int64)
            for i in range(num_sequences):
                if len(self._candidate) < n:
                    negative_samples[i] = np.random.choice(self._candidate, size=n, replace=True)
                else:
                    negative_samples[i] = np.random.choice(self._candidate, size=n, replace=False)
        
        return negative_samples

    def predict(self, sequence, item_ids=None):
        """
        Predict the ranking for the given user.
        """
        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')

        self._net.eval()

        with torch.no_grad():
            sequence = np.atleast_2d(sequence)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1)

            sequence = torch.from_numpy(sequence).long().to(self._device)
            item_ids = torch.from_numpy(item_ids).long().to(self._device)

            # Reshape inputs
            sequence = sequence.unsqueeze(0)

            out = self._net(sequence, item_ids, for_pred=True)

        return out.cpu().numpy().flatten()

    def load_pretrained_model(self, path):
        """
        Load the pre-trained model weights.
        """
        if self._net is None:
            raise ValueError("Model is not initialized. Please call the _initialize method first.")
        
        pretrained_dict = torch.load(path, map_location=self._device)
        model_dict = self._net.state_dict()

        # Only load matching keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self._net.load_state_dict(model_dict)

    def save_model(self, path):
        """
        Save the model to a file.
        """
        torch.save(self._net.state_dict(), path)
        print(f"Model saved to {path}")
        
    # def _evaluate_loss(self, val): # แก้ไขเพื่อใช้ binary cross-entropy loss
    #     self._net.eval()
    #     sequences_np = val.sequences.sequences
    #     targets_np = val.sequences.targets

    #     negatives_np = self._generate_negative_samples(val, n=self._neg_samples)

    #     sequences = torch.from_numpy(sequences_np).long().to(self._device)
    #     targets = torch.from_numpy(targets_np).long().to(self._device)
    #     negatives = torch.from_numpy(negatives_np).long().to(self._device)

    #     with torch.no_grad():
    #         items_to_predict = torch.cat((targets, negatives), 1)
    #         items_prediction = self._net(sequences, items_to_predict)

    #         targets_prediction = items_prediction[:, :targets.size(1)]
    #         negatives_prediction = items_prediction[:, targets.size(1):]

    #         positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction) + 1e-8))
    #         negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction) + 1e-8))
    #         val_loss = positive_loss + negative_loss

    #     return val_loss.item()
    
    def _evaluate_loss(self, val): # แก้ไขเพื่อใช้ BPR Loss
        self._net.eval()
        sequences_np = val.sequences.sequences
        targets_np = val.sequences.targets

        # สร้าง negative samples
        negatives_np = self._generate_negative_samples(val, n=self._neg_samples)

        # แปลงเป็น torch tensor และย้ายไปยังอุปกรณ์ที่กำหนด (เช่น CPU หรือ GPU)
        sequences = torch.from_numpy(sequences_np).long().to(self._device)
        targets = torch.from_numpy(targets_np).long().to(self._device)
        negatives = torch.from_numpy(negatives_np).long().to(self._device)

        with torch.no_grad():
            # รวม target และ negative samples เพื่อสร้าง items ที่ต้องการทำนาย
            items_to_predict = torch.cat((targets, negatives), 1)
            items_prediction = self._net(sequences, items_to_predict)

            # แยกการทำนาย target และ negative samples
            targets_prediction = items_prediction[:, :targets.size(1)]
            negatives_prediction = items_prediction[:, targets.size(1):]

            # **ใช้ BPR Loss**
            # คำนวณความแตกต่างระหว่างการทำนาย target และ negative
            loss_diff = targets_prediction - negatives_prediction

            # คำนวณ BPR loss โดยใช้ log-sigmoid
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(loss_diff) + 1e-8))

        return bpr_loss.item()


def train_model(model, train_data, test_data, config, pretrained_model_path=None, save_model_path=None, is_pretrain=True):
    """
    Train or fine-tune the model and evaluate it.
    """
    print(f"{'Pretraining' if is_pretrain else 'Fine-tuning'} the model...")

    if not model._initialized:
        model._initialize(train_data)

    if not is_pretrain and pretrained_model_path:
        model.load_pretrained_model(pretrained_model_path)

    # Get max index of items in training data sequences
    max_item_index = np.max(train_data.sequences.sequences)
    print(f"Max item index in training data: {max_item_index}")
    
    model.fit(train_data, test_data, verbose=True)

    if save_model_path is None:
        save_model_path = 'edx_pretrained_model.pth' if is_pretrain else 'kaggle_finetuned_model.pth'
    model.save_model(save_model_path)

    print("Performing final evaluation...")
    precision, recall, mean_aps, mrr, ndcg = evaluate_ranking(model, test_data, train_data, k=[1, 5, 10])
    print(f"Final results:")
    print(f"Precision: @1={precision[0].mean():.4f}, @5={precision[1].mean():.4f}, @10={precision[2].mean():.4f}")
    print(f"Recall: @1={recall[0].mean():.4f}, @5={recall[1].mean():.4f}, @10={recall[2].mean():.4f}")
    print(f"MAP={mean_aps:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}")
    
def plot_losses(edx_model, kaggle_model, thairobotics_model):
    # หาความยาวที่น้อยที่สุดระหว่างทั้งสามโมเดล
    min_epochs = min(len(edx_model.train_losses), len(kaggle_model.train_losses), len(thairobotics_model.train_losses))

    # สร้างช่วง epochs ตามความยาวที่น้อยที่สุด
    epochs = range(1, min_epochs + 1)

    # Plot edX losses (ตัดข้อมูลให้มีความยาวตรงกับ min_epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, edx_model.train_losses[:min_epochs], 'b-', label='edX Train Loss')
    plt.plot(epochs, edx_model.val_losses[:min_epochs], 'r-', label='edX Validation Loss')

    # Plot Kaggle losses
    plt.plot(epochs, kaggle_model.train_losses[:min_epochs], 'g-', label='Kaggle Train Loss')
    plt.plot(epochs, kaggle_model.val_losses[:min_epochs], 'y-', label='Kaggle Validation Loss')

    # Plot ThaiRobotics losses
    plt.plot(epochs, thairobotics_model.train_losses[:min_epochs], 'c-', label='ThaiRobotics Train Loss')
    plt.plot(epochs, thairobotics_model.val_losses[:min_epochs], 'm-', label='ThaiRobotics Validation Loss')

    plt.title('Training and Validation Losses for edX, Kaggle, and ThaiRobotics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # บันทึกรูปเป็นไฟล์ PNG
    plt.savefig('losses_plot.png')

    # ปิดรูปภาพที่ plot เสร็จแล้วเพื่อเคลียร์หน่วยความจำ
    plt.close()

def plot_losses_and_lr(edx_model, kaggle_model, thairobotics_model):
    # หาความยาวที่น้อยที่สุดระหว่างทั้งสามโมเดล
    min_epochs = min(len(edx_model.train_losses), len(kaggle_model.train_losses), len(thairobotics_model.train_losses))

    # สร้างช่วง epochs ตามความยาวที่น้อยที่สุด
    epochs = range(1, min_epochs + 1)

    # Plot edX losses and learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, edx_model.train_losses[:min_epochs], 'b-', label='edX Train Loss')
    plt.plot(epochs, edx_model.val_losses[:min_epochs], 'r-', label='edX Validation Loss')
    plt.plot(epochs, edx_model.learning_rates[:min_epochs], 'k--', label='edX Learning Rate')
    plt.title('Training and Validation Losses and Learning Rate for edX')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('edx_losses_and_lr_plot.png')
    plt.close()

    # Plot Kaggle losses and learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, kaggle_model.train_losses[:min_epochs], 'g-', label='Kaggle Train Loss')
    plt.plot(epochs, kaggle_model.val_losses[:min_epochs], 'y-', label='Kaggle Validation Loss')
    plt.plot(epochs, kaggle_model.learning_rates[:min_epochs], 'k--', label='Kaggle Learning Rate')
    plt.title('Training and Validation Losses and Learning Rate for Kaggle')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('kaggle_losses_and_lr_plot.png')
    plt.close()

    # Plot ThaiRobotics losses and learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, thairobotics_model.train_losses[:min_epochs], 'c-', label='ThaiRobotics Train Loss')
    plt.plot(epochs, thairobotics_model.val_losses[:min_epochs], 'm-', label='ThaiRobotics Validation Loss')
    plt.plot(epochs, thairobotics_model.learning_rates[:min_epochs], 'k--', label='ThaiRobotics Learning Rate')
    plt.title('Training and Validation Losses and Learning Rate for ThaiRobotics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('thairobotics_losses_and_lr_plot.png')
    plt.close()


# เรียกใช้ function เพื่อสร้างรูปแต่ละรูป
# plot_losses_separately(edx_model, kaggle_model, thairobotics_model)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--edx_train_root', type=str, default='datasets/edx/train.txt')
    parser.add_argument('--edx_test_root', type=str, default='datasets/edx/test.txt')
    parser.add_argument('--kaggle_train_root', type=str, default='datasets/kaggle/train.txt')
    parser.add_argument('--kaggle_test_root', type=str, default='datasets/kaggle/test.txt')
    parser.add_argument('--thairobotics_train_root', type=str, default='datasets/coursera_thairobotics/train.txt')
    parser.add_argument('--thairobotics_test_root', type=str, default='datasets/coursera_thairobotics/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=10)
    parser.add_argument('--use_cuda', type=str2bool, default=True)

    config = parser.parse_args()

    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=512)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')

    model_config = model_parser.parse_args([])
    model_config.L = config.L

    set_seed(config.seed, config.use_cuda)

    # Load edX data
    edx_train = Interactions(config.edx_train_root)
    edx_train.to_sequence(config.L, config.T)
    edx_test = Interactions(config.edx_test_root, user_map=edx_train.user_map, item_map=edx_train.item_map)

    # Load validation data for edX
    edx_val = Interactions('datasets/edx/val.txt')
    edx_val.to_sequence(config.L, config.T)
    # Precomputed embeddings for edX
    edx_precomputed_embeddings = np.load("datasets/edx/precomputed_embeddings.npy")

    # Pretrain the model on edX data
    edx_model = Recommender(n_iter=config.n_iter,
                            batch_size=config.batch_size,
                            learning_rate=config.learning_rate,
                            l2=config.l2,
                            neg_samples=config.neg_samples,
                            model_args=model_config,
                            use_cuda=config.use_cuda,
                            precomputed_embeddings=edx_precomputed_embeddings)

    train_model(edx_model, edx_train, edx_val, config, is_pretrain=True, save_model_path='edx_pretrained_model.pth')

    # Precomputed embeddings for Kaggle
    kaggle_precomputed_embeddings = np.load("datasets/kaggle/precomputed_embeddings.npy")

    # Load validation data for Kaggle
    kaggle_val = Interactions('datasets/kaggle/val.txt')
    kaggle_val.to_sequence(config.L, config.T)

    # Fine-tune the model on Kaggle data
    kaggle_train = Interactions(config.kaggle_train_root)
    kaggle_train.to_sequence(config.L, config.T)
    kaggle_test = Interactions(config.kaggle_test_root, user_map=kaggle_train.user_map, item_map=kaggle_train.item_map)

    kaggle_model = Recommender(n_iter=config.n_iter,
                            batch_size=config.batch_size,
                            learning_rate=config.learning_rate,
                            l2=config.l2,
                            neg_samples=config.neg_samples,
                            model_args=model_config,
                            use_cuda=config.use_cuda,
                            precomputed_embeddings=kaggle_precomputed_embeddings)

    train_model(kaggle_model, kaggle_train, kaggle_val, config, pretrained_model_path='edx_pretrained_model.pth', is_pretrain=False, save_model_path='kaggle_finetuned_model.pth')

    # Precomputed embeddings for ThaiRobotics
    thairobotics_precomputed_embeddings = np.load("datasets/coursera_thairobotics/precomputed_embeddings.npy")

    # Load validation data for ThaiRobotics
    thairobotics_val = Interactions('datasets/coursera_thairobotics/val.txt')
    thairobotics_val.to_sequence(config.L, config.T)

    # Fine-tune the model on ThaiRobotics data
    thairobotics_train = Interactions(config.thairobotics_train_root)
    thairobotics_train.to_sequence(config.L, config.T)
    thairobotics_test = Interactions(config.thairobotics_test_root, user_map=thairobotics_train.user_map, item_map=thairobotics_train.item_map)

    thairobotics_model = Recommender(n_iter=config.n_iter,
                                    batch_size=config.batch_size,
                                    learning_rate=config.learning_rate,
                                    l2=config.l2,
                                    neg_samples=config.neg_samples,
                                    model_args=model_config,
                                    use_cuda=config.use_cuda,
                                    precomputed_embeddings=thairobotics_precomputed_embeddings)

    train_model(thairobotics_model, thairobotics_train, thairobotics_val, config, pretrained_model_path='kaggle_finetuned_model.pth', is_pretrain=False, save_model_path='thairobotics_finetuned_model.pth')

    print("Pretraining and fine-tuning completed successfully.")
    # Call plot_losses after training is completed for all datasets
    plot_losses(edx_model, kaggle_model, thairobotics_model)
    # เรียกใช้ฟังก์ชันเพื่อ plot ทั้งค่า Loss และ Learning Rate
    plot_losses_and_lr(edx_model, kaggle_model, thairobotics_model)
