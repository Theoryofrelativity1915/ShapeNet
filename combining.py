from utils import get_point_clouds_and_labels
from tda_inferencingV2 import tda_prediction_wrapper
from trained_pointnet_classifierV2 import pointnet_prediction_wrapper

point_clouds, labels = get_point_clouds_and_labels()
pointnet_prediction = pointnet_prediction_wrapper([point_clouds[0]])
tda_prediction = tda_prediction_wrapper(point_clouds[0])


def get_pred(predictions):
    # calc and return prediction
    # could also return [argmax, index of argmax]
    pass


print(tda_prediction)
print(pointnet_prediction)
