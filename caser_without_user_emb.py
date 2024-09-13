import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import activation_getter

class Caser(nn.Module):
    def __init__(self, num_users, num_items, model_args, muse_dim=512, precomputed_embeddings=None):
        super(Caser, self).__init__()
        self.args = model_args
        self.num_items = num_items
        self.item_dims = muse_dim
        self.user_dims = model_args.d
        self.n_h = model_args.nh
        self.n_v = model_args.nv
        self.drop_ratio = model_args.drop
        self.ac_conv = activation_getter[model_args.ac_conv]
        self.ac_fc = activation_getter[model_args.ac_fc]

        # Initialize precomputed embeddings
        self.precomputed_embeddings = self._initialize_embeddings(num_items, precomputed_embeddings)

        # Vertical convolution
        self.conv_v = nn.Conv2d(1, self.n_v, (self.args.L, self.item_dims))

        # Horizontal convolution
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, self.item_dims)) for i in range(1, self.args.L + 1)])

        # Fully connected layer
        self.fc1_dim_v = self.n_v * 1
        self.fc1_dim_h = self.n_h * self.args.L
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.user_dims)

        # Output layer
        self.W2 = nn.Embedding(num_items, self.user_dims)
        self.b2 = nn.Embedding(num_items, 1)

        # Dropout layer
        self.dropout = nn.Dropout(self.drop_ratio)

        # Weight initialization
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def _initialize_embeddings(self, num_items, precomputed_embeddings):
        if precomputed_embeddings is None:
            return nn.Parameter(torch.randn(num_items, self.item_dims), requires_grad=False)
        if precomputed_embeddings.shape[0] != num_items:
            print(f"Adjusting precomputed embeddings from {precomputed_embeddings.shape[0]} to {num_items} items")
            new_embeddings = torch.zeros((num_items, precomputed_embeddings.shape[1]), dtype=torch.float)
            new_embeddings[:min(num_items, precomputed_embeddings.shape[0])] = precomputed_embeddings[:min(num_items, precomputed_embeddings.shape[0])]
            return nn.Parameter(new_embeddings, requires_grad=False)
        return nn.Parameter(torch.tensor(precomputed_embeddings, dtype=torch.float), requires_grad=False)


    def forward(self, item_seq, item_var, for_pred=False):
        # Convert item_var to long before embedding lookup
        item_var = item_var.long()

        # Look up the precomputed embeddings using item_seq
        item_embs = self.precomputed_embeddings[item_seq]  # Expected [batch_size, L, d]
        
        if len(item_embs.shape) == 3:
            item_embs = item_embs.unsqueeze(1)
        
        # Vertical Convolution
        out_v = self.conv_v(item_embs).view(-1, self.fc1_dim_v) if self.n_v else None  # [batch_size, n_v]

        # Horizontal Convolution
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

        # No need to combine with user embeddings anymore
        x = z  # [batch_size, user_dims]

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res



    def update_item_count(self, new_num_items):
        if new_num_items != self.num_items:
            print(f"Updating item count from {self.num_items} to {new_num_items}")
            self.num_items = new_num_items
            
            # Update precomputed embeddings
            old_embeddings = self.precomputed_embeddings.data
            new_embeddings = torch.zeros((new_num_items, old_embeddings.shape[1]), dtype=torch.float)
            new_embeddings[:min(new_num_items, old_embeddings.shape[0])] = old_embeddings[:min(new_num_items, old_embeddings.shape[0])]
            self.precomputed_embeddings = nn.Parameter(new_embeddings, requires_grad=False)
            
            # Update W2 and b2
            old_W2 = self.W2.weight.data
            new_W2 = torch.zeros((new_num_items, old_W2.shape[1]))
            new_W2[:min(new_num_items, old_W2.shape[0])] = old_W2[:min(new_num_items, old_W2.shape[0])]
            self.W2 = nn.Embedding(new_num_items, self.user_dims + self.user_dims)
            self.W2.weight.data = new_W2

            old_b2 = self.b2.weight.data
            new_b2 = torch.zeros((new_num_items, old_b2.shape[1]))
            new_b2[:min(new_num_items, old_b2.shape[0])] = old_b2[:min(new_num_items, old_b2.shape[0])]
            self.b2 = nn.Embedding(new_num_items, 1)
            self.b2.weight.data = new_b2

    def get_item_count(self):
        return self.num_items