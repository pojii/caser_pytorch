import argparse
import numpy as np
from train import Recommender, Interactions
import torch
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

if __name__ == '__main__':
    state_dict = torch.load('trained_caser_model.pkl')
    for k, v in state_dict.items():
        print(k, v.shape)

    # Load dataset and course mappings
    train_root = 'datasets/ml1m/test/train.txt'
    test_root = 'datasets/ml1m/test/test.txt'
    course_map_path = rf'datasets/coursera/text.txt'  # Path to the course number to name mapping

    course_map = load_course_map(course_map_path)

    # Load precomputed embeddings
    precomputed_embeddings = np.load("course_embeddings.npy")

    # Define model_config
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')
    
    model_config = model_parser.parse_args()
    model_config.L = 5  # Example value, make sure this matches what was used during training

    # Load dataset for model initialization
    test = Interactions(test_root)  # Load only test set as training is not required

    # Initialize the model
    model = Recommender(n_iter=50, batch_size=512, learning_rate=0.001, l2=1e-6, neg_samples=3, model_args=model_config, use_cuda=False, precomputed_embeddings=precomputed_embeddings)
    
    # Use the new method to initialize the model
    model.initialize_model(test)

    # Now load the model weights
    model.load_model('trained_caser_model.pkl')

    # Sample and test every user
    for user_id in range(len(model.test_sequence.sequences)):
        print(f"--- User {user_id} ---")
        sample_test(model, user_id=user_id, top_k=10, course_map=course_map)
