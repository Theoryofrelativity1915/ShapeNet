import torch.nn.functional as F
import torch.nn as nn
import torch


class PointNet2Classification(nn.Module):
    def __init__(self, num_classes=40, tda_dim=18):
        super(PointNet2Classification, self).__init__()
        self.tda_dim = tda_dim

        # Point Feature Extractor (MLP layers)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

        # Now input to fc1 is 1024 + tda_dim
        # Allows the model to train regularly without tda inputs
        self.fc1 = nn.Linear(1024 + tda_dim if tda_dim is not None else 0, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x, tda_feats=None):
        """
        x: (batch_size, num_points, 3)
        tda_feats: (batch_size, tda_dim)
        """
        x = x.permute(0, 2, 1)  # (B, 3, N)

        # Feature extraction
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = torch.max(x, 2)[0]  # (B, 1024)

        # Inject TDA features
        if tda_feats is not None:
            x = torch.cat([x, tda_feats], dim=1)  # (B, 1024 + tda_dim)

        # Classifier head
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x
