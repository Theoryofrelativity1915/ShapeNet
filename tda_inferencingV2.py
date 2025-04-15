from gtda.homology import VietorisRipsPersistence
import pickle
from utils import get_point_clouds_and_labels

with open("tda_weights", "rb") as f:
    pipe = pickle.load(f)


# Example: load new point clouds
new_point_clouds, true_labels = get_point_clouds_and_labels()
new_point_clouds = new_point_clouds[10:15]  # or any new data

# Important: you still need to compute diagrams for new data

homology_dimensions = [0, 1, 2]

persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)

new_persistence_diagrams = persistence.fit_transform(new_point_clouds)
