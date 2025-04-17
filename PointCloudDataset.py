import os
import numpy as np
from utils import read_txt_pointcloud, default_transforms
import torch


class PointCloudDataset:
    def __init__(self, root_dir, get_testset=False, transform=default_transforms(), combined_data=None):
        self.root_dir = root_dir
        self.transforms = transform
        # list of (file_path, label, tda_vector or None)
        self.samples = combined_data

    def __len__(self):
        return len(self.samples)

    def __preproc__(self, file_path):
        points = read_txt_pointcloud(file_path)
        return self.transforms(points)

    def __getitem__(self, idx):
        file_path, label, tda_vec, filename = self.samples[idx]
        pointcloud = self.__preproc__(file_path)
        item = {
            'pointcloud': pointcloud,
            'category': label,
            'filename': filename
        }
        if tda_vec is not None:
            item['tda'] = torch.tensor(tda_vec, dtype=torch.float32)
        return item

