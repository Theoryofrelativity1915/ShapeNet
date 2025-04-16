import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PointNet import PointNet2Classification as PointNet
from PointCloudDataset import PointCloudDataset
from constants import data_path

# --------------- Class map and device setup -----------------
class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
               'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
               'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio',
               'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand',
               'vase', 'wardrobe', 'xbox']
class_name_id_map = {name: idx for idx, name in enumerate(class_names)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------- Load & Reorder TDA features -----------------
def load_tda_features_dict(filepath, class_name_id_map):
    tda_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            *features, class_name = parts
            vec = list(map(float, features))
            key = class_name.lower()
            if key not in class_name_id_map:
                continue
            label = class_name_id_map[key]
            tda_dict.setdefault(key, []).append((vec, label))
    return tda_dict


def reorder_tda_from_filenames(filenames_file, tda_dict):
    reordered_features = []
    reordered_labels = []
    with open(filenames_file, 'r') as f:
        for line in f:
            name = line.strip()
            class_name = next(
                (c for c in tda_dict if name.startswith(c + "_")), None)
            if class_name is None:
                raise ValueError(f"TDA data missing for file: {name}")
            candidates = tda_dict[class_name]
            if not candidates:
                raise ValueError(f"No TDA vector left for {name}")
            vec, label = candidates.pop(0)
            reordered_features.append(vec)
            reordered_labels.append(label)
    return reordered_features, reordered_labels


tda_train_dict = load_tda_features_dict(
    "./dutta_modelnet/train-modelnet40-giottofeatures.txt", class_name_id_map)
tda_train_vecs, train_labels = reorder_tda_from_filenames(
    "./dutta_modelnet/modelnet40_train.txt", tda_train_dict)

tda_test_dict = load_tda_features_dict(
    "./dutta_modelnet/test-modelnet40-giottofeatures.txt", class_name_id_map)
tda_test_vecs, test_labels = reorder_tda_from_filenames(
    "./dutta_modelnet/modelnet40_test.txt", tda_test_dict)

tda_dim = len(tda_train_vecs[0])
print("TDA dim:", tda_dim)

# --------------- Create Datasets & Loaders -----------------
train_dataset = PointCloudDataset(
    root_dir=data_path,
    get_testset=False,
    tda_features=tda_train_vecs
)

val_dataset = PointCloudDataset(
    root_dir=data_path,
    get_testset=True,
    tda_features=tda_test_vecs
)

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(val_dataset, batch_size=64,
                          shuffle=False, num_workers=4, pin_memory=True)

# --------------- Initialize Model -----------------
model = PointNet(num_classes=40, tda_dim=tda_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
criterion = nn.CrossEntropyLoss()

# --------------- Training Function -----------------


def train(model, train_loader, val_loader=None, epochs=100):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_loss = 0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs = data['pointcloud'].to(device).float()
            labels = data['category'].to(device)
            tda_feats = data['tda'].to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs, tda_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            running_loss += loss.item()
            if i % 20 == 19:
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / 5))
                running_loss = 0.0

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss /
              len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        # Validation
        if val_loader:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for data in val_loader:
                    inputs = data['pointcloud'].to(device).float()
                    labels = data['category'].to(device)
                    tda_feats = data['tda'].to(device).float()
                    outputs = model(inputs, tda_feats)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

    # Save the model
    torch.save(model.state_dict(), "pointnet_with_tda.pth")


# --------------- Start Training -----------------
with warnings.catch_warnings():
    train(model, train_loader, valid_loader)

