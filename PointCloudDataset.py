import os
from utils import read_off, default_transforms
import torch


class PointCloudDataset:
    def __init__(self, root_dir, get_testset=False, transform=default_transforms(), tda_features=None):
        self.root_dir = root_dir
        self.tda_features = tda_features

        folders = [dr for dr in sorted(os.listdir(
            root_dir)) if os.path.isdir(f'{root_dir}/{dr}')]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = default_transforms()
        self.files = []
        self.labels = []
        sub_folder = 'test' if get_testset else 'train'

        for class_name, label in self.classes.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                class_dir += f'/{sub_folder}/'
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.off'):
                        self.files.append(class_dir + file_name)
                        # striped_file_name = file_name.rstrip(".off")
                        # self.files.append(
                        #     f'point_clouds/{striped_file_name}.pt')
                        self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file_path):
        verts, faces = read_off(file_path)
        pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]
        label = self.labels[idx]
        pointcloud = self.__preproc__(pcd_path)
        return {
            'pointcloud': pointcloud,
            'category': label,
            'tda': torch.tensor(self.tda_features[idx], dtype=torch.float32) if self.tda_features else None
        }
