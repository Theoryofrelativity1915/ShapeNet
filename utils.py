import torch
import numpy as np
import os
from torchvision import transforms
from PointSampler import PointSampler

homology_dimensions = [0, 1, 2]


def read_txt_pointcloud(file_path):
    """ Reads a point cloud from a .txt file with x,y,z per line. """
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
