import torch
from gtda.homology import VietorisRipsPersistence
import numpy as np
import open3d as o3d
import gudhi as gd
import os
import numpy as np
import json

homology_dimensions = [0, 1, 2]


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


def get_point_clouds_and_labels():
    labels = []
    pcds = []
    with open('./shapenet_data/train_split.json') as f:
        training_data = json.load(f)
        for training_data_file in training_data:
            labels.append(training_data_file[0])
            training_file = training_data_file[-1].split('/')
            actual_training_file_path = "shapenet_data/" + \
                training_file[0] + "/points/" + \
                training_file[-1].split('.')[0] + ".npy"
            pcds.append(np.load(actual_training_file_path))
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

