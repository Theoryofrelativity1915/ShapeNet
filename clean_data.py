import os
import shutil

# Set your root directory
base_dir = '.'

# Paths to the train/test file lists
train_list_file = 'modelnet40_train.txt'
test_list_file = 'modelnet40_test.txt'

# Helper function to process file list


def get_class_to_filenames(file_path):
    class_to_files = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_name, file_name = line.split('_', 1)
            if class_name not in class_to_files:
                class_to_files[class_name] = []
            class_to_files[class_name].append(f"{class_name}_{file_name}.txt")
    return class_to_files


# Parse file lists
train_files = get_class_to_filenames(train_list_file)
test_files = get_class_to_filenames(test_list_file)

# Create train/test subfolders and move files
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Create train and test subdirectories
    train_dir = os.path.join(class_path, 'train')
    test_dir = os.path.join(class_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move training files
    for file_name in train_files.get(class_name, []):
        src = os.path.join(class_path, file_name)
        dst = os.path.join(train_dir, file_name)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Move testing files
    for file_name in test_files.get(class_name, []):
        src = os.path.join(class_path, file_name)
        dst = os.path.join(test_dir, file_name)
        if os.path.exists(src):
            shutil.move(src, dst)

print("Finished reorganizing the dataset.")
