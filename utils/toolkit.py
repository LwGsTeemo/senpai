import os
import torch
import numpy as np
import nibabel as nib
from torch import optim

from utils.Loss import DiceBCELoss, DiceCELoss, hybrid_loss, multiLoss
from losses.cross_entropy import weightedCELoss
from losses.diceloss import MulticlassDiceLoss

def save_cache(img, path, single_file=False):
    save_dir = os.path.dirname(path)
    check_file_exists(save_dir)
    if single_file:
        np.save(path.split('.')[0], img)
    else:
        np.save(path + "cache", img)

def save_result(img, path, affine=None, header=None, type="nii"):
    save_dir = os.path.dirname(path)
    check_file_exists(save_dir)
    if type == "nii":
        img = nib.Nifti1Image(img, affine=affine)
        nib.save(img, path.split('.')[0])
    else:   # npy
        np.save(path, img)

def load_path(path):
    data_paths = [
        os.path.join(path, x)
        for x in os.listdir(path)
    ]
    data_paths.sort()
    return data_paths

def check_file_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_loss_function(loss, device):
    if loss == "Dice":
        return MulticlassDiceLoss()
    elif loss == "DiceBCE":
        return DiceBCELoss()
    elif loss == "DiceCE":
        return DiceCELoss()
    elif loss == "DiceFocal":
        return multiLoss()
    elif loss == "SSIM":
        return hybrid_loss()
    elif loss == "CE":
        return weightedCELoss()
    elif loss == "weightedCE":
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
        return weightedCELoss(weight=weight)
    else:
        raise ValueError("Wrong Loss Function Name.")

def get_optimizer(cfg, params):
    if cfg.optimizer == "Adam":
        return optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "AdamW":
        return optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "SGD":
        return optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum)
    else:
        raise ValueError("Wrong Optimzer Name.")

def iou_3d(box1, box2):
    ''' bbx [xmin, xmax, ymin, ymax, zmin, zmax] '''
    area1 = (box1[1] - box1[0]) * (box1[3] - box1[2]) * (box1[5] - box1[4])
    area2 = (box2[1] - box2[0]) * (box2[3] - box2[2]) * (box2[5] - box2[4])
    area_sum = area1 + area2

    x1 = max(box1[0], box2[0])
    y1 = max(box1[2], box2[2])
    z1 = max(box1[4], box2[4])
    x2 = min(box1[1], box2[1])
    y2 = min(box1[3], box2[3])
    z2 = min(box1[5], box2[5])
    if x1 >= x2 or y1 >= y2 or z1 >= z2:
        return 0
    else:
        inter_area = (x2 - x1) * (y2 - y1) * (z2 - z1)
    return inter_area / (area_sum - inter_area)