import numpy as np
import os
import open3d as o3d
import gudhi as gd
from gtda.homology import VietorisRipsPersistence
import joblib  # Assuming you saved the model using joblib (adjust if using a different format)

# The filename you specified
filename = os.path.join('dataset', '02691156', '1a04e3eab45ca15dd86060f189eb133.txt')

# Load your trained classifier (ensure you've already trained and saved this model)
# For example, if it's a RandomForestClassifier
# model = joblib.load('path_to_trained_model.pkl')  # Replace with your model file path

# Function to compute the persistence diagram in various dimensions
def compute_persistence_diagram(P, homology_dimensions=[0, 1, 2]):
    alpha_complex = gd.AlphaComplex(points=P)
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.persistence()
    pers_pairs = []
    for i in homology_dimensions:
        pairs = simplex_tree.persistence_intervals_in_dimension(i)
        for pair in pairs:
            # Ensure the persistence pairs are in the format [birth, death, dimension]
            pers_pairs.append([np.float64(pair[0]), np.float64(pair[1]), np.float64(i)])
    return pers_pairs

# Extract topological features from the persistence diagram
def extract_topological_features(persistence_diagram):
    # Extract the 18 topological features (3 for each dimension: 0, 1, 2)
    features = []
    for dim in range(3):
        dim_pairs = [pair for pair in persistence_diagram if pair[2] == dim]
        birth_death_diffs = [pair[1] - pair[0] for pair in dim_pairs]
        features.extend(birth_death_diffs)
    # Ensure we have 18 features
    if len(features) < 18:
        features.extend([0] * (18 - len(features)))  # Pad with zeros if less than 18 features
    return np.array(features)

# Read the point cloud and compute the persistence diagram and features
def process_point_cloud(filename):
    # Read the point cloud file (each line is a point)
    points = []
    with open(filename, 'r') as f:
        for line in f:
            point = list(map(float, line.split()[:3]))  # Get x, y, z values
            points.append(point)

    points = np.array(points)
    
    # Compute the persistence diagram for the point cloud
    persistence_diagram = compute_persistence_diagram(points)
    
    # Extract the 18 topological features
    topological_features = extract_topological_features(persistence_diagram)
    
    return topological_features

# Load point cloud data and extract features
topological_features = process_point_cloud(filename)

# Reshape the features to match the classifier's expected input format
# If the classifier expects a 3D array (batch_size, n_features, 1), we reshape it accordingly
topological_features = topological_features.reshape((1, -1, 1))  # 1 sample, n_features, 1
print(len(topological_features[0]))
# Assuming you have a trained model to make predictions
# For example, if you're using a Random Forest or SVM classifier:
# prediction = model.predict(topological_features)  # Make the prediction
# print("Prediction:", prediction)

# If you have a pre-trained model:
# model = joblib.load('your_trained_model.pkl')  # Adjust to your model path
# prediction = model.predict(topological_features)
# print(f"Prediction for point cloud: {prediction}")
