import torch
import numpy as np
import os
from torchvision import transforms
from PointSampler import PointSampler

homology_dimensions = [0, 1, 2]

"""

build_combined_dataset grabs the file path to a pointcloud and its associated
tda persistence diagram, then scales the tda vector and combines them.

"""


def normalize_tda_vector(vec):
    min_value = min(vec)
    max_value = max(vec)
    value_range = max_value - min_value

    # All values are the same. Gonna return the 0 vector
    if value_range == 0:
        return [0.0 for _ in vec]

    # normalize vector-values between 0 and 1
    normalized_vec = [(v - min_value) / value_range for v in vec]
    return normalized_vec


def build_combined_dataset(root_dir, split_file, tda_file=None, class_name_id_map=None):
    combined = []
    tda_lookup = []

    # Load TDA vectors if provided
    if tda_file:
        with open(tda_file, 'r') as f:
            for line in f:
                # parts = [float, float float, "airplane"]
                parts = line.strip().split()
                # All but last element assigned to features
                features = parts[0:-1]
                # Last element (the name) is assigned to class_name
                class_name = parts[-1]
                vec = normalize_tda_vector(list(map(float, features)))
                tda_lookup.append((vec, class_name.lower()))

    # Load file paths and match with labels and tda vecs
    with open(split_file, 'r') as f:
        for i, line in enumerate(f):
            name = line.strip()  # name = "airplane_0001"
            # Extract the class name from the filename
            class_name = None
            for cls in class_name_id_map:
                # all class names have an underscore after the class name that
                # we can use to help as a delimiter
                if name.startswith(cls + "_"):
                    class_name = cls
                    break
            if class_name is None:  # Should never get to this point
                raise ValueError(
                    f"Cannot determine class from filename: {name}")
            label = class_name_id_map[class_name]

            folder = 'test' if 'test' in split_file else 'train'
            file_path = os.path.join(
                root_dir, class_name, folder, f"{name}.txt")

            # Joins the file_path to the pointcloud and the tda_vector together
            # so there is no chance of them getting misaligned due to data shuffling
            tda_vec = tda_lookup[i][0] if tda_file else None
            combined.append((file_path, label, tda_vec, name))

    return combined


def read_txt_pointcloud(file_path):
    # Reads a point cloud from a .txt file with x,y,z per line.
    with open(file_path, 'r') as f:
        points = [list(map(float, line.strip().split(',')))
                  for line in f if line.strip()]
    return np.array(points)


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


def default_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])


def get_point_clouds_and_labels(dataset_dir):
    labels = []
    pcds = []
    for dr in os.listdir(dataset_dir):
        class_dir = dataset_dir + "/" + dr
        if not os.path.isdir(class_dir):  # Need to skip over the metadata file
            continue
        for folder in os.listdir(class_dir):
            if folder == "test":
                train_file_dir = class_dir + "/" + folder
                for file in os.listdir(train_file_dir):
                    file_path = train_file_dir + "/" + file
                    pcds.append(file_path)
                    labels.append(dr)
    return [pcds, labels]
