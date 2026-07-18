"""Thin-vessel width stratification.

For each test image: skeletonize the GT mask, measure vessel half-width as the
Euclidean distance transform (EDT) at skeleton points (width = 2*EDT), propagate
that width to every pixel via its nearest skeleton point, bin pixels by width,
and pool TP/FN/FP per bin across all images. Recall is expected to increase with
vessel width — thin vessels are the hard case — which analyze.py asserts.

Predicted-positive background pixels (false positives) are attributed to the
width bin of their nearest vessel, so precision/F1 are well-defined per bin.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

# Width bins in pixels: [lo, hi).
BINS = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, np.inf)]
BIN_LABELS = ["[1,2)", "[2,3)", "[3,5)", "[5,inf)"]


def width_map_full(gt_mask):
    """Per-pixel vessel width (float32), assigned to EVERY pixel via its nearest
    skeleton point (width = 2*EDT(GT) sampled at the skeleton centerline)."""
    gt = (np.asarray(gt_mask).squeeze() > 0).astype(np.uint8)
    if gt.sum() == 0:
        return np.zeros(gt.shape, dtype=np.float32)
    edt = ndimage.distance_transform_edt(gt)
    skel = skeletonize(gt.astype(bool))
    if skel.sum() == 0:
        return np.zeros(gt.shape, dtype=np.float32)
    width_skel = np.zeros(gt.shape, dtype=np.float32)
    width_skel[skel] = 2.0 * edt[skel]
    # EDT of the skeleton complement returns, per pixel, the index of the nearest
    # skeleton point -> propagate its centerline width to the whole image.
    _, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
    return width_skel[iy, ix].astype(np.float32)


def stratified_counts(gt_mask, pred_mask):
    """Per-bin TP/FN/FP counts for one image, keyed by BIN_LABELS.

    TP: pred=1 & gt=1 in bin; FN: pred=0 & gt=1 in bin;
    FP: pred=1 & gt=0, attributed to the nearest-vessel width bin.
    """
    gt = (np.asarray(gt_mask).squeeze() > 0).astype(np.uint8)
    pred = (np.asarray(pred_mask).squeeze() > 0).astype(np.uint8)
    width = width_map_full(gt)
    out = {}
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        in_bin = (width >= lo) & (width < hi)
        gt_bin = in_bin & (gt == 1)
        tp = int(np.sum(gt_bin & (pred == 1)))
        fn = int(np.sum(gt_bin & (pred == 0)))
        fp = int(np.sum(in_bin & (gt == 0) & (pred == 1)))
        out[label] = {"tp": tp, "fn": fn, "fp": fp}
    return out


def pool_scores(per_image_counts):
    """Pool a list of per-image count dicts into per-bin recall/precision/F1.

    Returns an ordered list of dicts (one per bin) with keys:
    label, tp, fn, fp, gt, recall, precision, f1.
    """
    rows = []
    for label in BIN_LABELS:
        tp = sum(c[label]["tp"] for c in per_image_counts)
        fn = sum(c[label]["fn"] for c in per_image_counts)
        fp = sum(c[label]["fp"] for c in per_image_counts)
        gt = tp + fn
        recall = tp / gt if gt > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")
        rows.append({"label": label, "tp": tp, "fn": fn, "fp": fp, "gt": gt,
                     "recall": recall, "precision": precision, "f1": f1})
    return rows
