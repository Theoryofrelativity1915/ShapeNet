import torch.nn.functional as F
import torch.nn as nn
import torch


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
