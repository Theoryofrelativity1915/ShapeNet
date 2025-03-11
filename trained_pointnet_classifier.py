import torch
from tqdm import tqdm
from PointNet import PointNet2Classification
from constants import class_name_id_map, DATA_FOLDER
from utils import collate_fn
import torch.nn as nn
from ShapeNetDataset import ShapeNetDataset
criterion = nn.CrossEntropyLoss()
test_set = ShapeNetDataset(root_dir=DATA_FOLDER, split_type='test')
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn)

class_id_name_map = {v: k for k, v in class_name_id_map.items()}
num_classes = len(class_id_name_map.items())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
