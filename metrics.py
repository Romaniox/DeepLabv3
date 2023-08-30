import torch
import numpy as np


def pix_acc(target, outputs, num_classes):
    """
    Calculates pixel accuracy, given target and output tensors 
    and number of classes.
    """
    labeled = (target > 0) * (target <= num_classes)
    _, preds = torch.max(outputs.data, 1)
    correct = ((preds == target) * labeled).sum().item()
    return labeled, correct


def get_tp_tn_fp_fn(target, outputs):
    _, preds = torch.max(outputs.data, 1)

    preds = preds.cpu().numpy()
    target = target.cpu().numpy()

    TP = np.sum(np.logical_and(preds == 1, target == 1))
    TN = np.sum(np.logical_and(preds == 0, target == 0))
    FP = np.sum(np.logical_and(preds == 1, target == 0))
    FN = np.sum(np.logical_and(preds == 0, target == 1))

    return TP, TN, FP, FN


def get_metrics(TP, TN, FP, FN):
    iou = TP / (TP + FP + FN)
    dice = 2 * TP / (2 * TP + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (np.spacing(1) + TP + FN)

    return iou, dice, precision, recall
