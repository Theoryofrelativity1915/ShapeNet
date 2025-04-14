from gtda.homology import VietorisRipsPersistence
import os
import numpy as np
import open3d as o3d
import gudhi as gd

# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]

# Collapse edges to speed up H2 persistence calculation!
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)


def compute_persistence_diagram_in_dimension_k(P, dimensions):
    alpha_complex = gd.AlphaComplex(points=P)
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.persistence()
    pers_pairs = []
    for i in dimensions:
        pairs = simplex_tree.persistence_intervals_in_dimension(i)
        for pair in pairs:
            # edited it for giotto-tda; i is now at the back instead of front
            pers_pairs.append([np.float64(pair[0]), (pair[1]), np.float64(i)])
    return pers_pairs


training_point_clouds_read_from_pcds = []
training_persistence_diagrams = []

# Replace with your .npz file path
npz_file_path = os.path.join('dataset', '02691156', '1a04e3eab45ca15dd86060f189eb133_8x8.npz')

# Load the .npz file
data = np.load(npz_file_path)

# Assuming the .npz file contains point cloud data as raw points, not file paths
point_cloud_data = data['pc']  # This should be an array of point clouds (each point cloud is a 2D array of shape (N, 3))

# Loop over each point cloud in the .npz file
for P in point_cloud_data:
    # Convert the points to np.float64 (Open3D expects this)
    P = np.asarray(P, dtype=np.float64)  # Ensure the points are in double precision
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)  # Set points for the point cloud
    training_point_clouds_read_from_pcds.append(np.asarray(pc.points))

    # Compute persistence diagram
    persistence_pairs = compute_persistence_diagram_in_dimension_k(P, homology_dimensions)
    training_persistence_diagrams.append(np.asarray(persistence_pairs))

    if len(training_persistence_diagrams) % 100 == 0:
        print(len(training_persistence_diagrams), "persistence diagrams computed")

# Convert to NumPy array (if desired)
training_point_clouds = np.asarray(training_point_clouds_read_from_pcds)
# print(training_point_clouds.shape)
