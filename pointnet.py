# Usual Imports
import torch.optim as optim
import torch.nn.functional as F
import glob
import torch.nn as nn
import torch
import os
import json
import numpy as np
from tqdm import tqdm

# plotting library
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
# from plotly.subplots import make_subplots

DATA_FOLDER = './shapenet_core'

class_name_id_map = {'Airplane': 0,
                     'Bag': 1,
                     'Cap': 2,
                     'Car': 3,
                     'Chair': 4,
                     'Earphone': 5,
                     'Guitar': 6,
                     'Knife': 7,
                     'Lamp': 8,
                     'Laptop': 9,
                     'Motorbike': 10,
                     'Mug': 11,
                     'Pistol': 12,
                     'Rocket': 13,
                     'Skateboard': 14,
                     'Table': 15}

class_id_name_map = {v: k for k, v in class_name_id_map.items()}

PCD_SCENE = dict(xaxis=dict(visible=False), yaxis=dict(
    visible=False), zaxis=dict(visible=False), aspectmode='data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_split_data = json.load(open(
    'shapenet_core/train_split.json', 'r'))
train_class_count = np.array([x[0] for x in train_split_data])

# plot classwise count in train set
train_dist_plots = [
    go.Bar(x=list(class_name_id_map.keys()), y=np.bincount(train_class_count))]
layout = dict(template="plotly_dark",
              title="Shapenet Core Train Distribution", title_x=0.5)
fig = go.Figure(data=train_dist_plots, layout=layout)
# fig.show()
points_list = glob.glob(
    "shapenet_core/04379243/points/*.npy")
print(len(points_list))


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


train_set = ShapeNetDataset(root_dir=DATA_FOLDER, split_type='train')
val_set = ShapeNetDataset(root_dir=DATA_FOLDER, split_type='val')
test_set = ShapeNetDataset(root_dir=DATA_FOLDER, split_type='test')
print(f"Train set length = {len(train_set)}")
print(f"Validation set length = {len(val_set)}")
print(f"Test set length = {len(test_set)}")


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


sample_loader = torch.utils.data.DataLoader(
    train_set, batch_size=16, num_workers=2, shuffle=True, collate_fn=collate_fn)
dataloader_iter = iter(sample_loader)
batch_dict = next(dataloader_iter)
print(batch_dict.keys())
for key in ['points', 'seg_labels', 'class_id']:
    print(f"batch_dict[{key}].shape = {batch_dict[key].shape}")
batchSize = 64
workers = 2
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batchSize, shuffle=True, num_workers=workers, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=batchSize, shuffle=True, num_workers=workers, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batchSize, shuffle=True, num_workers=workers, collate_fn=collate_fn)


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


class PointNet2Classification(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet2Classification, self).__init__()

        # Point Feature Extractor (MLP layers)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

        # Fully Connected Layers
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: (batch_size, num_points, 3) -> Input point cloud
        """
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 3, num_points)

        # Extract Point Features
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)

        # Global Max Pooling
        x = torch.max(x, 2)[0]  # (batch_size, 1024)

        # Fully Connected Classification Head
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x


def train_model(model, num_epochs, criterion, optimizer, dataloader_train,
                label_str='class_id', lr_scheduler=None, output_name='pointnet_pp.pth'):
    # move model to device
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Starting {epoch + 1} epoch ...")

        # Training
        model.train()
        train_loss = 0.0
        for batch_dict in tqdm(dataloader_train, total=len(dataloader_train)):
            # Forward pass
            x = batch_dict['points'].to(device)
            labels = batch_dict[label_str].to(device)
            pred = model(x)
            loss = criterion(pred, labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

        # compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)

        print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}')
    torch.save(model.state_dict(), output_name)


N_EPOCHS = 20
num_points = 2500
num_classes = 16
criterion = nn.CrossEntropyLoss()

# create model, optimizer, lr_scheduler and pass to training function
num_classes = len(class_id_name_map.items())
classifier = PointNet2Classification(num_classes=num_classes)

# DEFINE OPTIMIZERS
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
if torch.cuda.is_available():
    classifier.cuda()


_ = train_model(classifier, N_EPOCHS, criterion, optimizer, train_loader)

classifier = PointNet2Classification(num_classes=num_classes).to(device)
classifier.load_state_dict(torch.load('pointnet_pp.pth'))
classifier.eval()

total_loss = 0.0
all_labels = []
all_preds = []
total = 0
correct = 0

with torch.no_grad():
    for batch_dict in tqdm(test_loader, total=len(test_loader)):
        x = batch_dict['points'].to(device)
        labels = batch_dict['class_id'].to(device)
        pred = classifier(x)

        # calculate loss
        loss = criterion(pred, labels)
        total_loss += loss.item()
        total += labels.size(0)
        correct += pred.argmax(dim=1).eq(labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.argmax(dim=1).cpu().numpy())

evaluation_loss = total_loss / len(test_loader)
print(f"evaluation loss: {evaluation_loss}")
print(f"Test Accuracy: {100 * correct/total:.2f}%")
