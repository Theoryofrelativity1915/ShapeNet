import torch.nn.functional as F
import numpy as np
import tqdm
import torch
from PointNet import PointNet2Classification
from constants import class_name_id_map, DATA_FOLDER, num_classes
from utils import collate_fn
from ShapeNetDataset import ShapeNetDataset

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


test_sample = test_set[np.random.choice(np.arange(len(test_set)))]
batch_dict = collate_fn([test_sample])
x = batch_dict['points'].to(device)

# Get model predictions
model_preds = classifier(x)
predicted_class = torch.argmax(model_preds, axis=1).detach().cpu().numpy()[0]
print(model_preds)
predicted_class_name = class_id_name_map[predicted_class]
pred_class_probs = F.softmax(
    model_preds.flatten(), dim=0).detach().cpu().numpy()
print(class_id_name_map)
title = f"Label = {test_sample['class_name']
                   }, Predicted class = {predicted_class_name}"
print(title)
