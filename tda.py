from gtda.diagrams import Amplitude
from sklearn.pipeline import make_union
from gtda.diagrams import NumberOfPoints
import numpy as np
from openml.datasets.functions import get_dataset
from gtda.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from gtda.diagrams import PersistenceEntropy
from gtda.plotting import plot_diagram
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_point_cloud
from generate_datasets import make_point_clouds

# point_clouds_basic, labels_basic = make_point_clouds(
#     n_samples_per_shape=10, n_points=20, noise=0.5)
# point_clouds_basic.shape, labels_basic.shape
#
# plot_point_cloud(point_clouds_basic[0])
# plot_point_cloud(point_clouds_basic[10])
# plot_point_cloud(point_clouds_basic[-1])
#
# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]

# Collapse edges to speed up H2 persistence calculation!
# persistence = VietorisRipsPersistence(
#     metric="euclidean",
#     homology_dimensions=homology_dimensions,
#     n_jobs=6,
#     collapse_edges=True,
# )

# diagrams_basic = persistence.fit_transform(point_clouds_basic)

# # Circle
# plot_diagram(diagrams_basic[0])
# # Sphere
# plot_diagram(diagrams_basic[10])
# # Torus
# plot_diagram(diagrams_basic[-1])


# persistence_entropy = PersistenceEntropy()

# calculate topological feature matrix
# X_basic = persistence_entropy.fit_transform(diagrams_basic)

# expect shape - (n_point_clouds, n_homology_dims)
# X_basic.shape
# plot_point_cloud(X_basic)

# rf = RandomForestClassifier(oob_score=True)
# rf.fit(X_basic, labels_basic)

# print(f"OOB score: {rf.oob_score_:.3f}")


# steps = [
#     ("persistence", VietorisRipsPersistence(metric="euclidean",
#      homology_dimensions=homology_dimensions, n_jobs=6)),
#     ("entropy", PersistenceEntropy()),
#     ("model", RandomForestClassifier(oob_score=True)),
# ]

# pipeline = Pipeline(steps)
# pipeline.fit(point_clouds_basic, labels_basic)
# Pipeline(steps=[('persistence',
#                  VietorisRipsPersistence(homology_dimensions=[0, 1, 2],
#                                          n_jobs=6)),
#                 ('entropy', PersistenceEntropy()),
#                 ('model', RandomForestClassifier(oob_score=True))])
# pipeline["model"].oob_score_

# Let's improve our model

df = get_dataset('shapes').get_data(dataset_format='dataframe')[0]
# df.head()
# plot_point_cloud(df.query('target == "biplane0"')[["x", "y", "z"]].values)

point_clouds = np.asarray(
    [
        df.query("target == @shape")[["x", "y", "z"]].values
        for shape in df["target"].unique()
    ]
)
# point_clouds.shape
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)
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
print(f"OOB score: {pipe["rf"].oob_score_:.3f}")
