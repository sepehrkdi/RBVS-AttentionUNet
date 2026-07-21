"""Segmentation metrics (accuracy, IoU, F1), faithful to CV_Project.ipynb cell 74.

NumPy, computed on binary masks over ALL pixels (no field-of-view masking, matching
the original evaluation). F1 here equals the Dice coefficient of the binary masks.
"""
from __future__ import annotations

import numpy as np

_SMOOTH = 1e-7


def compute_metrics(y_true, y_pred):
    """Return (accuracy, iou, f1) for binary masks of identical shape."""
    y_true_f = np.asarray(y_true).flatten()
    y_pred_f = np.asarray(y_pred).flatten()

    accuracy = np.sum(y_true_f == y_pred_f) / len(y_true_f)

    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    iou = (intersection + _SMOOTH) / (union + _SMOOTH)

    f1 = (2 * intersection + _SMOOTH) / (np.sum(y_true_f) + np.sum(y_pred_f) + _SMOOTH)
    return accuracy, iou, f1
