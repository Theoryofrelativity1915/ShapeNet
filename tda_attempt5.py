import os
from utils import get_point_clouds_and_labels
from constants import DATA_FOLDER
import numpy as np
import pickle
from gtda.diagrams import Amplitude, NumberOfPoints, PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from sklearn.pipeline import make_union
from gtda.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

homology_dimensions = [0, 1, 2]

# Fetch data
print("Fetching point clouds.")
point_clouds, labels = get_point_clouds_and_labels()
point_clouds = point_clouds[0:100]  # Small subset for testing

# Compute persistence diagrams with gtda
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)
print("Calculating persistence diagrams.")

if os.path.exists("persistence_diagramsV2.pkl"):
    with open("persistence_diagramsV2.pkl", "rb") as f:
        persistence_diagrams = pickle.load(f)
    print("Loaded cached diagrams.")
else:
    persistence_diagrams = persistence.fit_transform(point_clouds)
    with open("persistence_diagramsV2.pkl", "wb") as f:
        pickle.dump(persistence_diagrams, f)
    print("Computed and cached diagrams.")

# Create features from diagrams
metrics = [{"metric": metric} for metric in [
    "bottleneck", "wasserstein", "landscape", "persistence_image"]]

feature_union = make_union(
    PersistenceEntropy(normalize=True),
    NumberOfPoints(n_jobs=-1),
    *[Amplitude(**metric, n_jobs=-1) for metric in metrics]
)

pipe = Pipeline([
    ("features", feature_union),
    ("rf", RandomForestClassifier(oob_score=True, random_state=42)),
])

print("Fitting pipeline to persistence diagrams.")
# Make sure labels match the sample subset
pipe.fit(persistence_diagrams, labels[:100])

print(f'OOB score: {pipe["rf"].oob_score_:.3f}')

with open('./tda_weights', 'wb') as f:
    pickle.dump(pipe, f)
