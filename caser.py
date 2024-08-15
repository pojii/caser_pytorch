import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import activation_getter

class Caser(nn.Module):
    def __init__(self, num_users, num_items, model_args, muse_dim=512, precomputed_embeddings=None):
        super(Caser, self).__init__()
        self.args = model_args
        self.precomputed_embeddings = precomputed_embeddings

        # Initialize parameters
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
        # Clamp seq_var to valid indices
        seq_var = torch.clamp(seq_var, max=self.precomputed_embeddings.size(0) - 1)

        # Convert user_var and item_var to long before embedding lookup
        user_var = user_var.long()
        item_var = item_var.long()

        # Before indexing into precomputed_embeddings
        print('seq_var.shape', seq_var.shape)
        print('seq_var min:', seq_var.min().item(), 'max:', seq_var.max().item())

        # Check the shape of precomputed_embeddings
        print('precomputed_embeddings.shape', self.precomputed_embeddings.shape)

        # Look up the precomputed embeddings using seq_var
        item_embs = self.precomputed_embeddings[seq_var]  # [batch_size, L, 512]

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

