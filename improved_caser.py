import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ImprovedCaser(nn.Module):
    def __init__(self, num_users, num_items, model_args):
        super(ImprovedCaser, self).__init__()
        self.args = model_args
        self.num_items = num_items
        self.item_dims = model_args.d
        self.n_h = model_args.nh
        self.n_v = model_args.nv
        self.drop_ratio = model_args.drop
        self.L = model_args.L
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(num_items, self.item_dims)
        
        # Vertical convolution
        self.conv_v = nn.ModuleList([nn.Conv2d(1, self.n_v, (i, self.item_dims)) for i in range(1, self.L + 1)])
        
        # Horizontal convolution
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, self.item_dims)) for i in range(1, self.L + 1)])
        
        # Fully connected layers
        fc_dim_in = self.n_v * self.L + self.n_h * self.L
        self.fc1 = nn.Linear(fc_dim_in, self.item_dims)
        self.fc2 = nn.Linear(self.item_dims, self.item_dims)
        self.fc3 = nn.Linear(self.item_dims, self.item_dims)
        
        # Output layer
        self.output = nn.Linear(self.item_dims, num_items)
        
        # Dropout
        self.dropout = nn.Dropout(self.drop_ratio)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(self.item_dims)
        self.bn2 = nn.BatchNorm1d(self.item_dims)
        self.bn3 = nn.BatchNorm1d(self.item_dims)
        
    def forward(self, item_seq):
        # Look up the embeddings
        item_embs = self.item_embeddings(item_seq).unsqueeze(1)
        
        # Check if the input has the correct shape
        if item_embs.dim() != 4:
            print(f"Unexpected input shape: {item_embs.shape}")
            item_embs = item_embs.unsqueeze(0) if item_embs.dim() == 3 else item_embs
        
        # Vertical convolution
        out_v = [F.relu(conv(item_embs)).squeeze(3) for conv in self.conv_v]
        out_v = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out_v]
        out_v = torch.cat(out_v, 1)
        
        # Horizontal convolution
        out_h = [F.relu(conv(item_embs)).squeeze(3) for conv in self.conv_h]
        out_h = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out_h]
        out_h = torch.cat(out_h, 1)
        
        # Concatenate
        out = torch.cat([out_v, out_h], 1)
        
        # Fully connected layers with residual connections and batch normalization
        z = F.relu(self.bn1(self.fc1(out)))
        z = self.dropout(z)
        z = z + F.relu(self.bn2(self.fc2(z)))  # Residual connection
        z = self.dropout(z)
        z = z + F.relu(self.bn3(self.fc3(z)))  # Another residual connection
        z = self.dropout(z)
        
        # Output
        scores = self.output(z)
        
        return scores

    def predict(self, sequence):
        self.eval()
        with torch.no_grad():
            scores = self.forward(sequence)
            if scores.dim() == 2 and scores.size(0) == 1:
                scores = scores.squeeze(0)
        return scores
    
    def compute_loss(self, pos_scores, neg_scores):
        # pos_scores shape: (batch_size, 1)
        # neg_scores shape: (batch_size, num_neg_samples)
        # Expand pos_scores to match the shape of neg_scores
        pos_scores = pos_scores.expand_as(neg_scores)
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

def train_step(model, optimizer, sequences, pos_items, neg_items):
    model.train()
    optimizer.zero_grad()
    
    scores = model(sequences)
    pos_scores = scores.gather(1, pos_items.unsqueeze(1)).squeeze()
    neg_scores = scores.gather(1, neg_items.unsqueeze(1)).squeeze()
    
    loss = bpr_loss(pos_scores, neg_scores)
    
    # Add L2 regularization
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += 1e-5 * l2_reg
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_model(model, train_data, val_data, epochs, batch_size, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            sequences, pos_items, neg_items = zip(*batch)
            
            sequences = torch.LongTensor(sequences).to(device)
            pos_items = torch.LongTensor(pos_items).to(device)
            neg_items = torch.LongTensor(neg_items).to(device)
            
            loss = train_step(model, optimizer, sequences, pos_items, neg_items)
            total_loss += loss
        
        avg_loss = total_loss / (len(train_data) // batch_size)
        val_loss = evaluate(model, val_data, batch_size, device)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
    
    return model

def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            sequences, pos_items, neg_items = zip(*batch)
            
            sequences = torch.LongTensor(sequences).to(device)
            pos_items = torch.LongTensor(pos_items).to(device)
            neg_items = torch.LongTensor(neg_items).to(device)
            
            scores = model(sequences)
            pos_scores = scores.gather(1, pos_items.unsqueeze(1)).squeeze()
            neg_scores = scores.gather(1, neg_items.unsqueeze(1)).squeeze()
            
            loss = bpr_loss(pos_scores, neg_scores)
            total_loss += loss.item()
    
    return total_loss / (len(data) // batch_size)

    

# Example usage
if __name__ == "__main__":
    # Define your hyperparameters
    num_users = 1000
    num_items = 10000
    model_args = type('Args', (), {
        'd': 50,
        'nh': 16,
        'nv': 4,
        'drop': 0.5,
        'L': 5
    })()

    # Create the model
    model = ImprovedCaser(num_users, num_items, model_args)

    # Define your training parameters
    epochs = 50
    batch_size = 128
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the appropriate device
    model = model.to(device)

    # Prepare your data (this is just a placeholder, replace with your actual data)
    train_data = [(np.random.randint(0, num_items, 5), np.random.randint(0, num_items), np.random.randint(0, num_items)) for _ in range(10000)]
    val_data = [(np.random.randint(0, num_items, 5), np.random.randint(0, num_items), np.random.randint(0, num_items)) for _ in range(1000)]

    # Train the model
    trained_model = train_model(model, train_data, val_data, epochs, batch_size, lr, device)

    print("Training completed!")