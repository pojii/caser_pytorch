import argparse
import pandas as pd
import numpy as np
from train_caser_edx import Recommender, Interactions
import torch

def load_course_map(course_map_path):
    course_map = {}
    with open(course_map_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                course_num = parts[0]  # Use as a string, not int
                course_name = parts[1]
                course_map[course_num] = course_name
    return course_map

def sample_test(model, user_id, top_k=10, course_map=None):
    """Test the model with a sample user, showing actual vs. predicted courses."""
    actual_courses = model.test_sequence.sequences[user_id, :]
    predicted_scores = model.predict(user_id)

    top_predictions = np.argsort(-predicted_scores)[:top_k]

    # Convert course numbers to names using the course_map
    actual_course_names = [course_map.get(str(course), "Unknown Course") for course in actual_courses]
    predicted_course_names = [course_map.get(str(course), "Unknown Course") for course in top_predictions]

    print(f"User ID: {user_id}")
    print(f"Actual Courses: {actual_course_names}")
    print(f"Top {top_k} Predicted Courses: {predicted_course_names}")

def evaluate_model(model, test_interactions, k_list=[10]):
    """Evaluate the model and print metrics."""
    precisions, recalls, mean_aps, mean_mrr, mean_hit = model.evaluate(test_interactions, k_list)
    
    for k, precision, recall in zip(k_list, precisions, recalls):
        print(f"Precision@{k}: {np.mean(precision):.4f}")
        print(f"Recall@{k}: {np.mean(recall):.4f}")
    
    print(f"Mean Average Precision: {mean_aps:.4f}")
    print(f"Mean Reciprocal Rank: {mean_mrr:.4f}")
    print(f"Hit Rate: {mean_hit:.4f}")

if __name__ == '__main__':
    # Parse command-line arguments for model configuration
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')
    model_config = model_parser.parse_args()
    model_config.L = 5  # Example value, should match what was used during training

    # Load the dataset and course mappings
    train_root = 'datasets/edx/train.txt'
    test_root = 'datasets/edx/test.txt'
    course_map_path = 'datasets/edx/text.txt'  # Path to the course number to name mapping
    course_map = load_course_map(course_map_path)

    # Load precomputed embeddings from CSV
    course_embeddings_df = pd.read_csv("datasets/edx/course_embeddings.csv")
    precomputed_embeddings = course_embeddings_df.iloc[:, 2:].to_numpy()

    # Load dataset for model initialization
    test_interactions = Interactions(test_root)  # Load only test set as training is not required

    # Initialize the model
    model = Recommender(
        n_iter=50, 
        batch_size=512, 
        learning_rate=0.001, 
        l2=1e-6, 
        neg_samples=3, 
        model_args=model_config, 
        use_cuda=False, 
        precomputed_embeddings=precomputed_embeddings
    )

    # Initialize the model using test data
    model.initialize_model(test_interactions)

    # Load trained model weights
    model.load_model('trained_caser_model.pkl')

    # Evaluate the model on the test set
    evaluate_model(model, test_interactions, k_list=[10])

    # Optionally, sample test for individual users
    for user_id in range(len(model.test_sequence.sequences)):
        print(f"--- User {user_id} ---")
        sample_test(model, user_id=user_id, top_k=10, course_map=course_map)
