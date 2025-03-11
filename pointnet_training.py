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
from PointNet import PointNet2Classification
from ShapeNetDataset import ShapeNetDataset
from utils import collate_fn
# plotting library
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from constants import class_name_id_map, DATA_FOLDER, num_classes, N_EPOCHS


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


train_set = ShapeNetDataset(root_dir=DATA_FOLDER, split_type='train')
val_set = ShapeNetDataset(root_dir=DATA_FOLDER, split_type='val')
test_set = ShapeNetDataset(root_dir=DATA_FOLDER, split_type='test')
print(f"Train set length = {len(train_set)}")
print(f"Validation set length = {len(val_set)}")
print(f"Test set length = {len(test_set)}")


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


criterion = nn.CrossEntropyLoss()

# create model, optimizer, lr_scheduler and pass to training function
num_classes = len(class_id_name_map.items())
classifier = PointNet2Classification(num_classes=num_classes)

# DEFINE OPTIMIZERS
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
if torch.cuda.is_available():
    classifier.cuda()


_ = train_model(classifier, N_EPOCHS, criterion, optimizer, train_loader)
