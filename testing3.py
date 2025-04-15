from os.path import isfile, join
from os import listdir
from generate_datasets import make_point_clouds
from gtda.homology import VietorisRipsPersistence
import numpy as np
import open3d as o3d
import gudhi as gd
# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]

# Collapse edges to speed up H2 persistence calculation!
ghosh_persistence = VietorisRipsPersistence(
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
path = './shapenet_data/02691156/points/'
files = [f for f in listdir(path) if isfile(join(path, f))]
for point_cloud_file_name in files:
    training_point_clouds_read_from_pcds.append(
        np.load(join(path, point_cloud_file_name)))

print(training_point_clouds_read_from_pcds[0][0])
# output shape is (2690, 2734, 3) which is 2690 point clouds, each having 2734
# points, and each point has x, y, z


# persistence_pairs = compute_persistence_diagram_in_dimension_k(training_persistence_diagrams, homology_dimensions)
# training_persistence_diagrams.append(np.asarray(persistence_pairs))
# # print("Training persistence diagrams", training_persistence_diagrams)
#
# point_clouds_basic, labels_basic = make_point_clouds(
#     n_samples_per_shape=10, n_points=20, noise=0.5)
# # print(point_clouds_basic.shape)
