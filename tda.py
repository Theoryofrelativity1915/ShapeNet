from utils import get_point_clouds_and_labels, compute_persistence_diagram_in_dimension_k
from constants import DATA_FOLDER
import numpy as np
import pickle
from gtda.diagrams import Amplitude
from sklearn.pipeline import make_union
from gtda.diagrams import NumberOfPoints
from gtda.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence

homology_dimensions = [0, 1, 2]

# Let's improve our model
print("Fetching point clouds.")
point_clouds, labels = get_point_clouds_and_labels()
point_clouds = point_clouds[0:10]


# point_clouds.shape
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)
print("Calculating persistence diagrams.")
persistence_diagrams = [compute_persistence_diagram_in_dimension_k(
    point_cloud, homology_dimensions) for point_cloud in point_clouds]


def normalize_dimensions(persistence_diagrams):
    m = get_max_len(persistence_diagrams)
    print("M = ", m)
    for i in range(len(persistence_diagrams)):
        diagram = persistence_diagrams[i]
        while len(diagram) < m:
            diagram.append(0)


def get_max_len(persistence_diagrams):
    m = 0
    for diagram in persistence_diagrams:
        if len(diagram) > m:
            m = len(diagram)
    return m


normalize_dimensions(persistence_diagrams)
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
print("Fitting pipeline to persistence diagrams.")
pipe.fit(persistence_diagrams, labels)
print(f'OOB score: {pipe["rf"].oob_score_:.3f}')
with open('./tda_weights', 'wb') as f:
    pickle.dump(pipe, f)
