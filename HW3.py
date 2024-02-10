import cv2
import numpy as np
import os

# Load the 5 hand images
images = []
for i in range(5):
    img = cv2.imread("image_" + str(i) + ".jpeg")
    images.append(img)

# Define the lines and axes for each image
lines = []
axes = []
for i in range(5):
    # Define the lines manually (e.g. two points along the length of each finger)
    line = []
    for j in range(5):
        # Define two points along the length of the jth finger
        # Placeholder values (replace with actual coordinates)
        p1 = (10, 20)  # Example coordinates
        p2 = (100, 200)  # Example coordinates
        line.append((p1, p2))
    
    # Define the axes manually (perpendicular to the lines)
    axis = []
    for j in range(5):
        # Define two points along the width of the jth finger
        # Placeholder values (replace with actual coordinates)
        p1 = (30, 40)  # Example coordinates
        p2 = (150, 250)  # Example coordinates
        axis.append((p1, p2))
    
    lines.append(line)
    axes.append(axis)

# Calculate the feature vectors for each image
feature_vectors = []
for i, img in enumerate(images):
    line = lines[i]
    axis = axes[i]
    
    # Calculate the intensity profiles along the defined axes
    profiles = []
    for j in range(5):
        p1, p2 = axis[j]
        profile = []
        for k in range(5):  # Iterate over the length of the axis
            # Interpolate between the two points to sample intensity values
            x = np.linspace(p1[0], p2[0], num=50, dtype=int)
            y = np.linspace(p1[1], p2[1], num=50, dtype=int)
            intensities = img[y, x]  # Extract intensity values along the line
            profile.append(np.mean(intensities))  # Use mean intensity as profile value
        profiles.append(profile)
    
    # Calculate the distances along the defined lines
    distances = []
    for j in range(5):
        p1, p2 = line[j]
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        distances.append(distance)
    
    # Combine the distances and profiles into a single feature vector
    feature_vector = np.concatenate((distances, np.ravel(profiles)))
    feature_vectors.append(feature_vector)

# Save feature vectors
np.save("feature_vectors.npy", feature_vectors)

# Load feature vectors
loaded_feature_vectors = np.load("feature_vectors.npy", allow_pickle=True)

# Compare the feature vectors
threshold = 100  # Example threshold (adjust as needed)
for i in range(4):
    for j in range(i + 1, 5):
        feature_vector1 = loaded_feature_vectors[i]
        feature_vector2 = loaded_feature_vectors[j]
        # Calculate the Euclidean distance between the feature vectors
        distance = np.linalg.norm(feature_vector1 - feature_vector2)
        # Compare the distance with the threshold to determine similarity
        if np.array_equal(feature_vector1, feature_vector2):
            print("Feature vectors", i, "and", j, "are identical")
        elif distance < threshold:
            print("Feature vectors", i, "and", j, "are similar (distance:", distance, ")")
        else:
            print("Feature vectors", i, "and", j, "are not similar (distance:", distance, ")")