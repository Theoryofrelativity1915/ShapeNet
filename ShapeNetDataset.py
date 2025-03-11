import os
import torch
import json
import numpy as np
class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split_type, num_samples=2500):
        self.root_dir = root_dir
        self.split_type = split_type
        self.num_samples = num_samples
        with open(os.path.join(root_dir, f'{self.split_type}_split.json'), 'r') as f:
            self.split_data = json.load(f)

    def __getitem__(self, index):
        # read point cloud data
        class_id, class_name, point_cloud_path, seg_label_path = self.split_data[index]

        # point cloud data
        point_cloud_path = os.path.join(self.root_dir, point_cloud_path)
        pc_data = np.load(point_cloud_path)

        # segmentation labels
        # -1 is to change part values from [1-16] to [0-15]
        # which helps when running segmentation
        pc_seg_labels = np.loadtxt(os.path.join(
            self.root_dir, seg_label_path)).astype(np.int8) - 1
#         pc_seg_labels = pc_seg_labels.reshape(pc_seg_labels.size,1)

        # Sample fixed number of points
        num_points = pc_data.shape[0]
        if num_points < self.num_samples:
            # Duplicate random points if the number of points is less than max_num_points
            additional_indices = np.random.choice(
                num_points, self.num_samples - num_points, replace=True)
            pc_data = np.concatenate(
                (pc_data, pc_data[additional_indices]), axis=0)
            pc_seg_labels = np.concatenate(
                (pc_seg_labels, pc_seg_labels[additional_indices]), axis=0)

        else:
            # Randomly sample max_num_points from the available points
            random_indices = np.random.choice(num_points, self.num_samples)
            pc_data = pc_data[random_indices]
            pc_seg_labels = pc_seg_labels[random_indices]

        # return variable
        data_dict = {}
        data_dict['class_id'] = class_id
        data_dict['class_name'] = class_name
        data_dict['points'] = pc_data
        data_dict['seg_labels'] = pc_seg_labels
        return data_dict

    def __len__(self):
        return len(self.split_data)
