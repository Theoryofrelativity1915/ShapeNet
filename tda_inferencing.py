import gudhi as gd
import open3d as o3d
from os import listdir
from os.path import isfile, join
from gtda.diagrams import Amplitude
from sklearn.pipeline import make_union
from gtda.diagrams import NumberOfPoints
from openml.datasets.functions import get_dataset
from gtda.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from gtda.diagrams import PersistenceEntropy
from gtda.plotting import plot_diagram
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_point_cloud
from generate_datasets import make_point_clouds
from ShapeNetDataset import ShapeNetDataset
from constants import DATA_FOLDER
import numpy as np
import pickle

# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]
# point_clouds.shape
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)


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


point_clouds = []
path = './shapenet_data/02691156/points/'
files = [f for f in listdir(path) if isfile(join(path, f))]
for point_cloud_file_name in files:
    point_clouds.append(
        np.load(join(path, point_cloud_file_name)))

persistence_diagrams = persistence.fit_transform(point_clouds)
# Index - (human_arms_out, 0), (vase, 10), (dining_chair, 20), (biplane, 30)
# index = 30
# # plot_diagram(persistence_diagrams[index])
# persistence_entropy = PersistenceEntropy(normalize=True)
# # Calculate topological feature matrix
# X = persistence_entropy.fit_transform(persistence_diagrams)
# Visualise feature matrix
# plot_point_cloud(X)

labels = np.zeros(40)
labels[10:20] = 1
labels[20:30] = 2
labels[30:] = 3

# rf = RandomForestClassifier(oob_score=True, random_state=42)
# rf.fit(X, labels)
#

# Select a variety of metrics to calculate amplitudes
metrics = [
    {"metric": metric}
    for metric in ["bottleneck",
                   "wasserstein",
                   "landscape",
                   "persistence_image"]
]
# Concatenate to generate 3 + 3 + (4 x 3) = 18 topological features
feature_union = make_union(
    PersistenceEntropy(normalize=True),
    NumberOfPoints(n_jobs=-1),
    *[Amplitude(**metric, n_jobs=-1) for metric in metrics]
)
pipe = Pipeline(
    [
        ("features", feature_union),
        ("rf", RandomForestClassifier(oob_score=True, random_state=42)),
    ]
)
pipe.fit(persistence_diagrams, labels)
print(f'OOB score: {pipe["rf"].oob_score_:.3f}')
