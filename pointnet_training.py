import warnings
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from constants import data_path
from utils import get_folders
from PointCloudDataset import PointCloudDataset
from PointNet import PointNet2Classification as PointNet

folders = get_folders()

train_dataset = PointCloudDataset(data_path)
val_dataset = PointCloudDataset(data_path, get_testset=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=32,
                          shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(
    dataset=val_dataset, batch_size=64, num_workers=4, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = PointNet(num_classes=40)
model.to(device)
# print(next(model.parameters()).is_cuda)  # Should output True

optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
criterion = nn.CrossEntropyLoss()
# print(torch.cuda.is_available())  # Should print True if GPU is available
# print(torch.cuda.current_device())  # Should print the index of the current GPU
# print(torch.cuda.get_device_name(0))  # Should print the name of the GPU


def train(model, train_loader, val_loader=None, epochs=100):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_loss = 0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(
                device).float(), data['category'].to(device)
            optimizer.zero_grad()
            # outputs = model(inputs.transpose(1,2))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / 5))
                running_loss = 0.0

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss /
              len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

        model.eval()
        correct = total = 0
        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(
                        device).float(), data['category'].to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

    # save the model
    torch.save(model.state_dict(), "0-pointNet-modelNet.pth")


with warnings.catch_warnings():
    # warnings.simplefilter("ignore")
    train(model, train_loader, valid_loader)
