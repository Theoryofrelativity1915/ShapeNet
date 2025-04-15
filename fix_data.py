import os
import shutil


def fix_data(modelnet_folder_path):
    train_split = modelnet_folder_path + "modelnet40_train.txt"
    file = open(train_split, "r")
    training_files = set()
    for line in file:
        training_file_name = line.split("\n")[0]
        training_files.add(training_file_name)
    for class_folder in os.listdir(modelnet_folder_path):
        class_folder_path = f'{modelnet_folder_path}{class_folder}'
        if os.path.isdir(class_folder_path):
            for point_cloud_file in os.listdir(class_folder_path):
                if point_cloud_file.split(".")[0] in training_files:
                    path_to_point_cloud_file = class_folder_path + "/" + point_cloud_file

                    # print(point_cloud_file)
    # for dr in os.listdir(data_path):
    #     if os.path.isdir(f'{data_path}/{dr}'):
    #         folders.append(dr)


fix_data("./dataset/")
