import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PointNet import PointNet2Classification as PointNet
from PointCloudDataset import PointCloudDataset
from constants import data_path
class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
               'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
               'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio',
               'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand',
               'vase', 'wardrobe', 'xbox']
class_name_id_map = {name: idx for idx, name in enumerate(class_names)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------- Load TDA features -----------------


def load_tda_features(filepath, class_name_id_map):
    tda_vectors = []
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            *features, class_name = parts
            vec = list(map(float, features))
            tda_vectors.append(vec)
            labels.append(class_name_id_map[class_name])
    return tda_vectors, labels


# Update if your filenames are different
tda_train_vecs, train_labels = load_tda_features(
    "./dutta_modelnet/train-modelnet40-giottofeatures.txt", class_name_id_map)
tda_test_vecs, test_labels = load_tda_features(
    "./dutta_modelnet/test-modelnet40-giottofeatures.txt", class_name_id_map)
tda_dim = len(tda_train_vecs[0])

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
