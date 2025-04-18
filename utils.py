import torch
from gtda.homology import VietorisRipsPersistence
import numpy as np
import open3d as o3d
import gudhi as gd
import os
import numpy as np
import json
from torchvision import transforms
from PointSampler import PointSampler
from constants import data_path

homology_dimensions = [0, 1, 2]


def get_folders():
    folders = [dr for dr in sorted(os.listdir(
        data_path)) if os.path.isdir(f'{data_path}/{dr}')]
    return folders


def farthest_point_sample(xyz, npoint):
    """Randomly selects farthest points to downsample the point cloud."""
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def save_point_clouds(folders):
    ii = 0
    for dr in folders:
        for sub_folder in ['test', 'train']:
            print(os.path.join(data_path, dr, sub_folder))
            dataset_dir = os.path.join(data_path, dr, sub_folder)
            dataset_files = os.listdir(dataset_dir)
            for file in dataset_files:
                file_path = os.path.join(dataset_dir, file)
                file_name = file.rstrip(".txt")
                verts, faces = read_txt_pointcloud(file_path)
                pointcloud = PointSampler(1024)((verts, faces))
                pointcloud = Normalize()(pointcloud)
                ii += 1
                torch.save(pointcloud, f"point_clouds/{file_name}.pt")
        print(f'{ii} number of point clouds have been saved!')


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


def collate_fn(batch_list):
    ret = {}
    ret['class_id'] = torch.from_numpy(
        np.array([x['class_id'] for x in batch_list])).long()
    ret['class_name'] = np.array([x['class_name'] for x in batch_list])
    ret['points'] = torch.from_numpy(
        np.stack([x['points'] for x in batch_list], axis=0)).float()
    ret['seg_labels'] = torch.from_numpy(
        np.stack([x['seg_labels'] for x in batch_list], axis=0)).long()
    return ret


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
