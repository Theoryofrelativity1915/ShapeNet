import torch
import torch.nn.functional as F
from utils import get_point_clouds_and_labels, default_transforms, read_off
import numpy as np
from PointNet import PointNet2Classification
from constants import num_classes
import os


def get_modelnet40_class_map(dataset_root):
    class_folders = sorted([
        f for f in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, f))
    ])
    return {cls_name: idx for idx, cls_name in enumerate(class_folders)}


class_name_id_map = get_modelnet40_class_map('./ModelNet40Dataset')


def pad_or_truncate_point_cloud(pc, target_len=1024):
    """
    Ensures a point cloud has exactly `target_len` points by truncating or padding.
    """
    if pc.shape[0] > target_len:
        # Truncate
        idx = np.random.choice(pc.shape[0], target_len, replace=False)
        return pc[idx]
    elif pc.shape[0] < target_len:
        # Pad
        pad_len = target_len - pc.shape[0]
        pad = np.zeros((pad_len, pc.shape[1]))
        return np.vstack([pc, pad])
    else:
        return pc


def pointnet_prediction_wrapper(point_clouds, num_points=1024):
    """
    Takes a list of numpy arrays of shape (N_points, 3),
    and returns softmax class probabilities for each point cloud.
    """
    device = torch.device("cpu")

    # Load model
    model = PointNet2Classification(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('pointnet_pp.pth', map_location=device))
    model.eval()

    # Normalize all point clouds to same size
    uniform_pcs = [pad_or_truncate_point_cloud(
        pc, num_points) for pc in point_clouds]
    batch_tensor = torch.tensor(
        np.stack(uniform_pcs), dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    # Optional: print class predictions
    class_id_name_map = {v: k for k, v in class_name_id_map.items()}
    predicted_classes = probs.argmax(axis=1)
    predicted_class_names = [class_id_name_map[i] for i in predicted_classes]

    for i, (cls, name) in enumerate(zip(predicted_classes, predicted_class_names)):
        # print(f"Sample {i}: Predicted class ID = {cls}, Name = {name}")
        print(f"Predicted class ID = {cls}, Name = {name}")

    return probs


def get_max(preds):
    m = [-1, -1]
    preds = preds[0]
    for i in range(len(preds)):
        if preds[i] > m[0]:
            m[0] = preds[i]
            m[1] = i
    return m


def load_pointcloud_from_off(path):
    verts, faces = read_off(path)
    pointcloud = default_transforms()((verts, faces))
    return pointcloud  # NumPy array of shape (N, 3)


if __name__ == "__main__":
    clouds, labels = get_point_clouds_and_labels('./ModelNet40Dataset')
    print(f'Actual class: {labels[0]}')

    # This is the critical fix: convert .off file to a point cloud
    pc = load_pointcloud_from_off(clouds[0])  # <--- FIXED

    certainty, predicted_class = get_max(
        pointnet_prediction_wrapper([pc]))  # Now a proper input

    # print(f'Predicted class ID: {predicted_class}')
    print(f'Probability of prediction: {certainty:.4f}')
