import numpy as np
# Load the feature vectors
feature_vectors = np.load("feature_vectors.npy")

# Initialize a matrix to store pairwise distances
pairwise_distances = np.zeros((5, 5))

# Perform pairwise comparison
for i in range(5):
    for j in range(i+1, 5):
        # Calculate Euclidean distance between feature vectors
        distance = np.linalg.norm(feature_vectors[i] - feature_vectors[j])
        # Store the distance in the matrix
        pairwise_distances[i, j] = distance

# Visualize the pairwise distances
print("Pairwise Distances:")
print(pairwise_distances)
