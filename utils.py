import torch
from gtda.homology import VietorisRipsPersistence
import numpy as np
import open3d as o3d
import gudhi as gd
import os
from torchvision import transforms
from PointSampler import PointSampler
from constants import data_path

homology_dimensions = [0, 1, 2]


def load_tda_features(filepath, class_name_id_map):
    tda_vectors = []
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            *features, class_name = parts
            vec = list(map(float, features))
            tda_vectors.append(vec)
            labels.append(class_name_id_map[class_name])
    return tda_vectors, labels


def get_folders():
    return [dr for dr in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, dr))]


def farthest_point_sample(xyz, npoint):
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
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
            for file in os.listdir(dataset_dir):
                if not file.endswith('.off'):
                    continue
                file_path = os.path.join(dataset_dir, file)
                file_name = file.rstrip(".off")
                verts, faces = read_off(file_path)
                pointcloud = PointSampler(1024)((verts, faces))
                pointcloud = Normalize()(pointcloud)
                ii += 1
                torch.save(pointcloud, f"point_clouds/{file_name}.pt")
    print(f'{ii} point clouds saved.')


def read_off(file_path):
    with open(file_path, 'r') as file:
        off_header = file.readline().strip()
        if off_header == 'OFF':
            n_verts, n_faces, _ = map(int, file.readline().strip().split())
        else:
            n_verts, n_faces, _ = map(int, off_header[3:].split())
        verts = [list(map(float, file.readline().strip().split()))
                 for _ in range(n_verts)]
        faces = [list(map(int, file.readline().strip().split()))[1:]
                 for _ in range(n_faces)]
        return np.array(verts), faces


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        return torch.from_numpy(pointcloud)


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        min_coords = np.min(pointcloud, axis=0)
        max_coords = np.max(pointcloud, axis=0)
        scale = max_coords - min_coords + 1e-6
        normed = (pointcloud - min_coords) / scale  # [0, 1]
        normed = normed * 2.0 - 1.0  # [-1, 1]
        return normed


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
        class_dir = os.path.join(dataset_dir, dr)
        if not os.path.isdir(class_dir):
            continue
        for folder in os.listdir(class_dir):
            if folder == "test":
                test_dir = os.path.join(class_dir, folder)
                for file in os.listdir(test_dir):
                    pcds.append(os.path.join(test_dir, file))
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
            pers_pairs.append([np.float64(pair[0]), pair[1], np.float64(i)])
    return pers_pairs
