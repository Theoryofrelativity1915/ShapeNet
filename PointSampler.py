import numpy as np


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, pointcloud):
        """
        Uniformly samples `output_size` points from the input point cloud.
        If the point cloud has fewer points than required, samples with replacement.
        """
        num_points = pointcloud.shape[0]

        if num_points >= self.output_size:
            indices = np.random.choice(num_points, self.output_size, replace=False)
        else:
            indices = np.random.choice(num_points, self.output_size, replace=True)

        return pointcloud[indices]

