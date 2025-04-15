from data.generate_datasets import make_point_clouds

point_clouds_basic, labels_basic = make_point_clouds(
    n_samples_per_shape=10, n_points=20, noise=0.5)
point_clouds_basic.shape, labels_basic.shape
