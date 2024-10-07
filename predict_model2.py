from flask import Flask, request, jsonify, send_from_directory
import torch
import argparse
import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from caser_without_user_emb import Caser
from train_caser_finetune_both import Recommender, Interactions, set_seed, load_item_texts

# Set device to CPU
device = torch.device('cpu')

# Define model_args
model_parser = argparse.ArgumentParser()
model_parser.add_argument('--d', type=int, default=512)
model_parser.add_argument('--nv', type=int, default=4)
model_parser.add_argument('--nh', type=int, default=16)
model_parser.add_argument('--drop', type=float, default=0.5)
model_parser.add_argument('--ac_conv', type=str, default='relu')
model_parser.add_argument('--ac_fc', type=str, default='relu')
model_parser.add_argument('--L', type=int, default=5)

model_args = model_parser.parse_args([])

# Load training data for initialization
thairobotics_train = Interactions('datasets/coursera_thairobotics/train.txt')
thairobotics_train.to_sequence(model_args.L, 10)  # Assuming T=10

# Load item texts and create a mapping from text to ID
item_texts = load_item_texts('datasets/coursera_thairobotics/text.txt', thairobotics_train.num_items)
item_text_to_id = {text: idx for idx, text in enumerate(item_texts) if text != '<PAD>'}

# Initialize the Recommender model
model = Recommender(
    n_iter=100,
    batch_size=512,
    learning_rate=1e-3,
    l2=1e-6,
    neg_samples=10,
    model_args=model_args,
    use_cuda=False
)

# Initialize the model with training data
model._initialize(thairobotics_train)

# Load the pretrained model
model.load_pretrained_model('thairobotics_finetuned_model.pth')
model._net.eval()

# Initialize Flask app
app = Flask(__name__)

def find_closest_course(input_course, item_texts, model):
    input_embedding = model._get_muse_embeddings([input_course])[0]
    item_embeddings = model._get_muse_embeddings(item_texts)
    
    closest_course = None
    min_distance = float('inf')
    
    for idx, item_embedding in enumerate(item_embeddings):
        distance = cosine(input_embedding, item_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_course = item_texts[idx]
    
    return closest_course

@app.route('/')
def serve_recommend_html():
    return send_from_directory('.', 'recommend.html')

@app.route('/text')
def serve_txt():
    return send_from_directory(r'datasets\coursera_thairobotics', 'text.txt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_courses = data['input']
    
    # Find closest matches for input courses
    matched_courses = [find_closest_course(course, item_texts, model) for course in input_courses]
    print(f"Input courses: {input_courses}")
    print(f"Matched courses: {matched_courses}")
    
    # Convert matched courses to their IDs
    matched_ids = [item_text_to_id.get(course, 0) for course in matched_courses]  # Use 0 (PAD) if not found
    
    # Get embeddings for matched courses
    matched_embeddings = model._get_muse_embeddings(matched_courses)
    
    # Prepare input for the Caser model
    sequence = torch.tensor(matched_embeddings).float().unsqueeze(0).to(device)  # Add batch dimension and ensure float type
    
    # Get the number of items in the model
    num_items = model._net.W2.num_embeddings

    # Create item_var for all items
    item_var = torch.arange(num_items).long().to(device)
    
    print(f"Number of items in model: {num_items}")
    print(f"Number of item_texts: {len(item_texts)}")
    print(f"Max value in item_var: {item_var.max().item()}")
    
    with torch.no_grad():
        output = model._net(sequence, item_var, for_pred=True)
    
    # Sort the output in descending order and get the indices
    sorted_scores, sorted_indices = torch.sort(output, descending=True)
    
    # Convert the indices to a list
    output_list = sorted_indices.squeeze().tolist()
    
    # Map indices to item texts, excluding input courses and limiting to valid indices
    mapped_output = [item_texts[idx] for idx in output_list if idx < len(item_texts) and item_texts[idx] not in matched_courses][:10]  # Get top 10 recommendations
    
    return jsonify({'prediction': mapped_output, 'matched_input': matched_courses})

if __name__ == '__main__':
    # Generate embeddings for all courses in advance
    print("Generating embeddings for all courses...")
    _ = model._get_muse_embeddings(item_texts)
    print("Embeddings generated.")
    
    app.run(debug=True)