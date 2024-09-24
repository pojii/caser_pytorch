from flask import Flask, request, jsonify, send_from_directory
import torch
import argparse
import numpy as np
from caser_without_user_emb import Caser
from train_caser_finetune_both import Recommender, Interactions, set_seed

# Set device to CPU
device = torch.device('cpu')

# Define model_args from train_caser_finetune_both.py
model_parser = argparse.ArgumentParser()
model_parser.add_argument('--d', type=int, default=512)
model_parser.add_argument('--nv', type=int, default=4)
model_parser.add_argument('--nh', type=int, default=16)
model_parser.add_argument('--drop', type=float, default=0.5)
model_parser.add_argument('--ac_conv', type=str, default='relu')
model_parser.add_argument('--ac_fc', type=str, default='relu')
model_parser.add_argument('--L', type=int, default=5)  # Add L parameter

model_args = model_parser.parse_args([])

# Define num_users and num_items
num_users = 1000  # Replace with actual number of users
num_items = 275  # Replace with actual number of items

# Load precomputed embeddings
precomputed_embeddings = np.load("datasets/coursera_thairobotics/precomputed_embeddings.npy")

# Initialize the Recommender model
model = Recommender(
    n_iter=100,
    batch_size=512,
    learning_rate=1e-3,
    l2=1e-6,
    neg_samples=10,
    model_args=model_args,
    use_cuda=False,
    precomputed_embeddings=precomputed_embeddings
)

# Load training data for initialization
thairobotics_train = Interactions('datasets/coursera_thairobotics/train.txt')
thairobotics_train.to_sequence(model_args.L, 10)  # Assuming T=10

# Initialize the model with training data
model._initialize(thairobotics_train)

# Load the pretrained model
model.load_pretrained_model('thairobotics_finetuned_model.pth')
model._net.eval()

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def serve_recommend_html():
    return send_from_directory('.', 'recommend.html')

@app.route('/text')
def serve_txt():
    return send_from_directory(r'datasets\coursera_thairobotics', 'text.txt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Assuming input data is in the correct format for the model
    input_tensor = torch.tensor(data['input']).long().to(device)
    item_var = torch.arange(1, num_items).long().to(device)  # Create item_var as a range from 1 to 245
    
    # Reshape input_tensor to 4D (batch_size, channels, height, width)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
    
    # Reshape item_var to match the expected dimensions
    item_var = item_var.unsqueeze(0).unsqueeze(2).to(device)  # Add batch and channel dimensions
    
    with torch.no_grad():
        output = model._net(input_tensor, item_var, for_pred=True)  # Pass item_var to the model with for_pred=True
    
    # Sort the output in descending order and get the indices
    sorted_scores, sorted_indices = torch.sort(output, descending=True)
    
    # Convert the indices to a list or any serializable format and add 1
    output_list = (sorted_indices + 1).tolist()
    
    # Read the text file and store the lines in a list
    with open('datasets/coursera_thairobotics/text.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(lines)
    # Check if indices are within the range of lines
    max_index = len(lines)
    mapped_output = []

    # Iterate over each index in the output_list
    for idx in output_list:
        # Convert to 0-based index by subtracting 1
        zero_based_idx = idx - 1
        
        # Check if the index is within the valid range
        if 0 <= zero_based_idx < max_index:
            # Get the corresponding line from the text file and strip any leading/trailing whitespace
            line = lines[zero_based_idx].strip()
            
            # Append the line to the mapped_output list
            mapped_output.append(line)
    return jsonify({'prediction': mapped_output})

if __name__ == '__main__':
    app.run(debug=True)