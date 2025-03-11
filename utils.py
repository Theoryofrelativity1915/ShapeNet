import torch
import numpy as np


def collate_fn(batch_list):
    ret = {}
    ret['class_id'] = torch.from_numpy(
        np.array([x['class_id'] for x in batch_list])).long()
    ret['class_name'] = np.array([x['class_name'] for x in batch_list])
    ret['points'] = torch.from_numpy(
        np.stack([x['points'] for x in batch_list], axis=0)).float()
    ret['seg_labels'] = torch.from_numpy(
        np.stack([x['seg_labels'] for x in batch_list], axis=0)).long()
    return ret
