from gtda.homology import VietorisRipsPersistence
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

with open('training-9600.txt', 'r') as f:
    for source_pcd_file_path in f:
        P = o3d.io.read_point_cloud(
            source_pcd_file_path.replace("\n", '')).points
        training_point_clouds_read_from_pcds.append(np.asarray(P))

        persistence_pairs = compute_persistence_diagram_in_dimension_k(
            P, homology_dimensions)
        training_persistence_diagrams.append(np.asarray(persistence_pairs))

        if len(training_persistence_diagrams) % 100 == 0:
            print(len(training_persistence_diagrams),
                  "persistence diagrams computed")

training_point_clouds = np.asarray(training_point_clouds_read_from_pcds)
print(training_point_clouds.shape)
validation_point_clouds_read_from_pcds = []
validation_persistence_diagrams = []

with open('validation-9600.txt', 'r') as f:
    for source_pcd_file_path in f:
        P = o3d.io.read_point_cloud(
            source_pcd_file_path.replace("\n", '')).points
        validation_point_clouds_read_from_pcds.append(np.asarray(P))

        persistence_pairs = compute_persistence_diagram_in_dimension_k(
            P, homology_dimensions)
        validation_persistence_diagrams.append(np.asarray(persistence_pairs))

        if len(validation_persistence_diagrams) % 100 == 0:
            print(len(validation_persistence_diagrams),
                  "persistence diagrams computed")

validation_point_clouds = np.asarray(validation_point_clouds_read_from_pcds)
print(validation_point_clouds.shape)
