# training_wrapper.py
from train import train_model

if __name__ == "__main__":
    # Toggle this flag to switch between using TDA features and not using TDA features (generic PointNet implementation)
    use_tda = True  # Set to False to disable TDA
    train_model(
        use_tda=True,
        tda_train_file="./dutta_modelnet/train-modelnet40-giottofeatures.txt",
        tda_test_file="./dutta_modelnet/test-modelnet40-giottofeatures.txt",
        train_split_file="./dutta_modelnet/modelnet40_train.txt",
        test_split_file="./dutta_modelnet/modelnet40_test.txt",
        epochs=100
    )
