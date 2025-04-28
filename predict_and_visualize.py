import torch
import numpy as np
import open3d as o3d
import os
from PointNet import PointNet2Classification as PointNet
from constants import class_name_id_map
from utils import read_txt_pointcloud

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_pointcloud_as_image(points, filename="output_nightstand.png"):
    """
    Saves a point cloud visualization to an image file without opening a window.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    width, height = 800, 600
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # White background

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    renderer.scene.add_geometry("pointcloud", pcd, material)

    center = pcd.get_center()
    # Camera positioned above the point cloud
    eye = center + np.array([0, 0, 1])
    up = [0, 1, 0]  # Y-axis up
    renderer.scene.camera.look_at(center, eye, up)

    img = renderer.render_to_image()
    o3d.io.write_image(filename, img)

    print(f"Saved visualization to {filename}")


def predict_single_pointcloud(model, points, tda_features=None):
    """
    Makes a prediction for a single point cloud.
    points: (N, 3) numpy array
    tda_features: (1, tda_dim) torch tensor or None
    Returns predicted class index.
    """
    model.eval()
    with torch.no_grad():
        pc_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)

        if tda_features is not None:
            tda_features = tda_features.to(device).float()

        outputs = model(
            pc_tensor, tda_features) if tda_features is not None else model(pc_tensor)
        _, predicted = outputs.max(1)

    return predicted.item()


def load_model(model_path, use_tda=False, tda_dim=18):
    """
    Loads a trained PointNet model from checkpoint.
    """
    model = PointNet(
        num_classes=40, tda_dim=tda_dim if use_tda else None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_last_tda_vector_for_class(tda_filepath, target_class="airplane"):
    """
    Loads and returns the normalized TDA feature vector
    for the last occurrence of a target class name in the TDA file.
    """
    matching_lines = []

    with open(tda_filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            *features, class_name = parts
            if class_name.lower() == target_class.lower():
                matching_lines.append(features)

    if not matching_lines:
        raise ValueError(f"No lines found for class {
                         target_class} in {tda_filepath}")

    # Take the last matching line
    last_features = np.array(list(map(float, matching_lines[-1])))

    # Normalize between 0 and 1
    min_val = last_features.min()
    max_val = last_features.max()
    range_val = max_val - min_val
    if range_val == 0:
        normalized = np.zeros_like(last_features)
    else:
        normalized = (last_features - min_val) / range_val

    return normalized.reshape(1, -1)


# ----------------- EXAMPLE USAGE -----------------

if __name__ == "__main__":
    model_checkpoint = "pointnet_with_tda.pth"
    pointcloud_file = "dutta_modelnet/xbox/test/xbox_0123.txt"
    tda_file = "dutta_modelnet/test-modelnet40-giottofeatures.txt"
    use_tda = True

    model = load_model(model_checkpoint, use_tda=use_tda, tda_dim=24)
    points = read_txt_pointcloud(pointcloud_file)

    tda_features = None
    if use_tda:
        tda_vector = load_last_tda_vector_for_class(
            tda_file, target_class="xbox")
        tda_features = torch.from_numpy(tda_vector).float()

    # Save visualization instead of opening a window
    save_pointcloud_as_image(points, filename="xbox.png")

    # Predict
    predicted_label_idx = predict_single_pointcloud(
        model, points, tda_features)
    predicted_class_name = [
        name for name, idx in class_name_id_map.items() if idx == predicted_label_idx][0]

    print(f"Predicted Class: {predicted_class_name}")
