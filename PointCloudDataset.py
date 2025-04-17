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
        }

        if tda_vec is not None:
            item['tda'] = torch.tensor(tda_vec, dtype=torch.float32)

        # 🔍 Alignment check for validation file: airplane_0726.txt
        if "airplane_0726" in filename:
            known_tda = np.array([
                -5.348405991861834, -4.782133603317537, -
                0.9409446476670644, 9999.0, 13224.0, 3005.0,
                294.0201193966182, 0.02940668850195373, 0.16393500855869012, 0.0022072449714838033,
                0.0, 0.0, 0.04280019125000001, 0.00020514790847418125, 0.004106460002235668,
                0.04283308660854048, 0.001907972524559497, 0.0041957055304018215, 0.007229375420284279,
                0.0, 0.0, 13377.81401074921, 1127.5232194997568, 1389.2635557833303
            ])
            known_points = np.array([
                [-0.002363, 0.09491, -0.07672],
                [-0.3283, -0.03349, 0.9416],
                [0.3154, 0.0216, -0.9443],
                [0.8149, 0.3792, -0.1114],
                [-0.7335, 0.01062, -0.1139],
                [0.3218, 0.03774, 0.6977],
                [0.5046, -0.05566, -0.1145],
                [7e-06, -0.03766, -0.56],
                [0.01635, -0.03713, 0.3667],
                [0.7469, 0.1054, -0.4161],
                [0.7445, 0.1037, 0.1898],
                [-0.3508, -0.06238, -0.174],
                [-0.2537, -0.09024, 0.1526],
                [-0.0597, -0.149, -0.2465],
                [0.7699, 0.07781, -0.1042],
                [0.2291, -0.03723, -0.1944],
                [0.5474, 0.2268, -0.1098],
                [0.2106, -0.0322, -0.7108],
                [0.04643, -0.04135, 0.1186],
                [-0.5044, 0.08197, -0.05531],
                [0.2127, -0.02269, 0.4958],
                [-0.2247, 0.06697, -0.03964],
                [-0.2387, -0.07539, -0.3598],
                [0.5853, 0.06976, 0.04352],
                [0.5833, 0.07045, -0.2777],
                [0.34, 0.06146, -0.03593],
                [0.07791, -0.03183, -0.3616],
                [0.06197, -0.08428, -0.1093],
                [-0.0847, -0.1442, 0.02779],
                [-0.1608, 0.007365, -0.2068],
                [-0.5599, -0.0578, -0.1647],
                [0.1704, 0.02153, -0.02069],
                [-0.4454, -0.1468, -0.293],
                [0.3888, 0.03673, -0.2038],
                [0.7148, 0.2406, -0.1075],
                [-0.0831, -0.02682, -0.4151],
                [-0.1096, -0.0181, 0.1912],
                [-0.3577, 0.09989, -0.1313],
                [0.08831, 0.04668, -0.1993],
                [0.1856, -0.03055, -0.8724],
                [-0.4505, -0.1643, -0.1078],
                [-0.3681, -0.01746, -0.02486],
                [0.3418, -0.08348, -0.09949],
                [-0.1764, -0.07824, -0.07935],
                [0.1211, -0.03339, 0.2549],
                [0.1563, -0.02673, -0.5684],
                [0.6215, 0.09697, -0.1324],
                [0.4799, 0.0959, -0.08085],
                [0.1922, -0.02368, 0.6577],
                [0.2226, 0.09931, -0.1349]
            ])

            # TDA check
            if tda_vec is not None and np.allclose(tda_vec[:24], known_tda, atol=1e-4):
                print("Good! TDA vector matches known airplane_0726 features")
            else:
                print("Bad! TDA vector mismatch for airplane_0726")

            # Point cloud check
            raw = np.loadtxt(file_path, delimiter=",")
            if np.allclose(raw[:50], known_points, atol=1e-4):
                print("Good! Point cloud matches known airplane_0726.txt points")
            else:
                print("Bad! Point cloud mismatch for airplane_0726.txt")
        return item

    # def __getitem__(self, idx):
    #     file_path, label, tda_vec, filename = self.samples[idx]
    #     pointcloud = self.__preproc__(file_path)
    #     item = {
    #         'pointcloud': pointcloud,
    #         'category': label,
    #         'filename': filename
    #     }
    #     if tda_vec is not None:
    #         item['tda'] = torch.tensor(tda_vec, dtype=torch.float32)
    #     return item

