DATA_FOLDER = './shapenet_data'
N_EPOCHS = 20
num_points = 2500
num_classes = 16
data_path = 'dutta_modelnet/'

# ----- Setup -----
class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
               'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
               'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio',
               'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand',
               'vase', 'wardrobe', 'xbox']
class_name_id_map = {name: idx for idx, name in enumerate(class_names)}
