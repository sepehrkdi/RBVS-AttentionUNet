"""Focal + Dice loss, faithful to CV_Project.ipynb cells 40 & 42.

The training objective is binary_focal_loss(alpha=0.9, gamma=7) applied to the
sigmoid probability, plus (1 - dice_coef). Operating on probabilities (not
logits) with the notebook's explicit clipping keeps the loss numerically
identical to the original run.
"""
from __future__ import annotations

import torch

_EPS = 1e-6
_SMOOTH = 1e-6


def binary_focal_loss(alpha: float = 0.9, gamma: float = 7.0):
    """Return loss_fn(y_pred_prob, y_true) computing the mean binary focal loss.

    FL = -alpha*y*(1-p)^gamma*log(p) - (1-alpha)*(1-y)*p^gamma*log(1-p)
    """
    def loss_fn(y_pred, y_true):
        p = torch.clamp(y_pred, _EPS, 1.0 - _EPS)
        ce = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
        p_t = torch.where(y_true == 1.0, p, 1.0 - p)
        alpha_factor = torch.where(
            y_true == 1.0,
            torch.as_tensor(alpha, dtype=p.dtype, device=p.device),
            torch.as_tensor(1.0 - alpha, dtype=p.dtype, device=p.device),
        )
        modulating = torch.pow(1.0 - p_t, gamma)
        loss = alpha_factor * modulating * ce
        return loss.mean()
    return loss_fn


def dice_coef(y_true, y_pred):
    """Soft Dice coefficient over the whole batch (flattened), as in the notebook."""
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    inter = torch.sum(y_true_f * y_pred_f)
    return (2.0 * inter + _SMOOTH) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + _SMOOTH)


def combined_loss(y_pred, y_true, alpha: float = 0.9, gamma: float = 7.0):
    """focal(alpha, gamma) + (1 - dice). Matches the notebook's compile lambda
    (cell 46): loss = binary_focal_loss(0.9, 7)(yt, yp) + (1 - dice_coef(yt, yp))."""
    focal = binary_focal_loss(alpha, gamma)(y_pred, y_true)
    return focal + (1.0 - dice_coef(y_true, y_pred))
