import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PointNet import PointNet2Classification as PointNet
from PointCloudDataset import PointCloudDataset
from constants import class_name_id_map, data_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Helper: Build combined dataset -----


def build_combined_dataset(root_dir, split_file, tda_file=None, class_name_id_map=None):
    combined = []
    tda_lookup = []

    # Load TDA vectors if provided
    if tda_file:
        with open(tda_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                *features, class_name = parts
                vec = list(map(float, features))
                tda_lookup.append((vec, class_name.lower()))

    # Load file paths and match with labels and tda vecs
    with open(split_file, 'r') as f:
        for i, line in enumerate(f):
            name = line.strip()  # e.g., airplane_0001
            class_name = next(
                (cls for cls in class_name_id_map if name.startswith(cls + "_")), None)
            if class_name is None:
                raise ValueError(
                    f"Cannot determine class from filename: {name}")
            label = class_name_id_map[class_name]

            folder = 'test' if 'test' in split_file else 'train'
            file_path = os.path.join(
                root_dir, class_name, folder, f"{name}.txt")

            tda_vec = tda_lookup[i][0] if tda_file else None
            combined.append((file_path, label, tda_vec, name))

    return combined

# ----- Training pipeline -----


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
    # if use_tda:
    #     print(f"TDA dim: {tda_dim}")

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

            # Debug first batch
            # if epoch == 0 and batch_idx == 0:
            #     print("\n DEBUG: Verifying alignment in first batch")
            #     for i in range(len(labels)):
            #         label = labels[i].item()
            #         tda_value = tda_feats[i][0].item(
            #         ) if tda_feats is not None else None
            #         filename = data['filename'][i]
            #         print(f"  Sample {i}: File = {filename}, Label = {
            #             label}, TDA[0] = {tda_value:.4f}")

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
            f"[Epoch {epoch+1}/{epochs}] Train Loss: { total_loss: .4f} | Accuracy: {100*correct/total: .2f} %")

        # Validation
        model.eval()
        correct = total = 0
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
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total:.2f}%\n")

    # Save the model
    suffix = "_with_tda" if use_tda else "_no_tda"
    torch.save(model.state_dict(), f"pointnet{suffix}.pth")
