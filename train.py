import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from PointNet import PointNet2Classification as PointNet
from PointCloudDataset import PointCloudDataset
from constants import class_name_id_map, data_path
from utils import build_combined_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
This is essentially a generic training function for PointNet. 

The key here is loading and intervweaving the tda data. The most difficult part of this was probably aligning the persistence diagrams with their corresponding point clouds. The act of shuffling the data set meant that I had to combine all of the data prior to dataset shuffling. Hence the "build_combined_dataset" function.

"""


def train_model(use_tda=False,
                tda_train_file=None,
                tda_test_file=None,
                train_split_file=None,
                test_split_file=None,
                epochs=100):

    print(f"Using device: {device}")

    # Build samples (pointcloud file, label, tda_vec)
    train_samples = build_combined_dataset(
        data_path, train_split_file,
        tda_train_file if use_tda else None,
        class_name_id_map
    )

    val_samples = build_combined_dataset(
        data_path, test_split_file,
        tda_test_file if use_tda else None,
        class_name_id_map
    )

    tda_dim = len(train_samples[0][2]) if use_tda else None

    # Dataset and loaders
    train_dataset = PointCloudDataset(
        root_dir=data_path, combined_data=train_samples)
    val_dataset = PointCloudDataset(
        root_dir=data_path, combined_data=val_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Model and training setup
    model = PointNet(num_classes=40, tda_dim=tda_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = total = 0

        for batch_idx, data in enumerate(train_loader):
            inputs = data['pointcloud'].to(device).float()
            labels = data['category'].to(device)
            tda_feats = data.get('tda', None)
            if tda_feats is not None:
                tda_feats = tda_feats.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs, tda_feats) if use_tda else model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(
            f"[Epoch {epoch+1}/{epochs}] Train Loss: {total_loss: .4f} | Accuracy: {100*correct/total: .2f} %")

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in valid_loader:
                inputs = data['pointcloud'].to(device).float()
                labels = data['category'].to(device)
                tda_feats = data.get('tda', None)
                if tda_feats is not None:
                    tda_feats = tda_feats.to(device).float()

                outputs = model(
                    inputs, tda_feats) if use_tda else model(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate precision, recall, and F1 score
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}\n")

    # Save the model
    suffix = "_with_tda" if use_tda else "_no_tda"
    torch.save(model.state_dict(), f"pointnet{suffix}.pth")
