import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

# Load the MUSE model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# Step 1: Load the dataset
def load_dataset(file_path):
    user_course_rating = []
    with open(file_path, 'r') as file:
        for line in file:
            user_id, course_id, rating = map(int, line.strip().split())
            user_course_rating.append((user_id, course_id, rating))
    return user_course_rating

# Load data
train_data = load_dataset('./datasets/coursera/train.txt')
test_data = load_dataset('./datasets/coursera/test.txt')

# Step 2: Extract unique course IDs and shift them
course_ids = set()
for _, course_id, _ in train_data + test_data:
    course_ids.add(course_id)

# Shift course IDs to start from 1
course_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(course_ids), start=1)}

# Step 3: Load course names
course_names = {}
with open('./datasets/coursera/text.txt', 'r') as file:
    for line in file:
        course_id, course_name = line.strip().split(maxsplit=1)
        course_id = int(course_id)
        if course_id in course_ids:
            course_names[course_id_mapping[course_id]] = course_name

# Step 4: Generate embeddings
# Sort new course_ids to ensure consistent ordering
sorted_new_course_ids = sorted(course_id_mapping.values())
sorted_course_names = [course_names[course_id] for course_id in sorted_new_course_ids]

# Generate embeddings for the sorted course names
course_embeddings = embed(sorted_course_names)

# Step 5: Convert embeddings to numpy array
precomputed_embeddings = course_embeddings.numpy()

# Step 6: Add padding embedding (zeros) at index 0
padding_embedding = np.zeros((1, precomputed_embeddings.shape[1]))
precomputed_embeddings = np.vstack((padding_embedding, precomputed_embeddings))

# Step 7: Save the embeddings to a file
np.save("precomputed_embeddings.npy", precomputed_embeddings)

# Step 8: Save the mapping of original course IDs to new indices
course_id_to_index = {old_id: course_id_mapping[old_id] for old_id in course_ids}
np.save("course_id_to_index.npy", course_id_to_index)

print("Precomputed embeddings and course ID mapping have been saved.")
