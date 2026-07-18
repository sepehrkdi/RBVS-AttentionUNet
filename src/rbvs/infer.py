"""Full-image inference by reflect-pad tile-and-stitch with count-map averaging.

Standardized eval config: patch 128, stride 64, threshold 0.5 — the config that
produced the notebook's reported 96.4/63.8/77.9 (cell 74). Patches are predicted
in batches for speed; the arithmetic (overlap-average of probabilities, then
threshold) is identical to the notebook's per-patch loop (cell 59).
"""
from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def predict_prob_full(image, model, device, patch_size=128, stride=64, batch_size=64):
    """Return the averaged probability map (H, W) float32 for a full image.

    image: (H, W) or (H, W, 1) array, uint8 [0,255] or float [0,1].
    """
    img = np.asarray(image).astype("float32")
    if img.ndim == 3:
        img = img[..., 0]
    if img.max() > 1.0:
        img /= 255.0
    H, W = img.shape[:2]

    # Pad so an integer number of strided steps covers the image (reflect avoids
    # zero-border artifacts).
    n_h = math.ceil((H - patch_size) / stride) + 1
    n_w = math.ceil((W - patch_size) / stride) + 1
    H_pad = (n_h - 1) * stride + patch_size
    W_pad = (n_w - 1) * stride + patch_size
    img_p = np.pad(img, ((0, H_pad - H), (0, W_pad - W)), mode="reflect")

    ys = range(0, H_pad - patch_size + 1, stride)
    xs = range(0, W_pad - patch_size + 1, stride)
    coords = [(y, x) for y in ys for x in xs]

    sum_probs = np.zeros((H_pad, W_pad), dtype=np.float32)
    count_map = np.zeros((H_pad, W_pad), dtype=np.float32)

    model.eval()
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = np.stack([img_p[y:y + patch_size, x:x + patch_size] for (y, x) in chunk])
        t = torch.from_numpy(batch).unsqueeze(1).to(device)   # (B,1,ps,ps)
        prob = model(t).squeeze(1).cpu().numpy()              # (B,ps,ps)
        for (y, x), pm in zip(chunk, prob):
            sum_probs[y:y + patch_size, x:x + patch_size] += pm
            count_map[y:y + patch_size, x:x + patch_size] += 1.0

    avg = sum_probs / count_map
    return avg[:H, :W]


def predict_full(image, model, device, patch_size=128, stride=64, threshold=0.5,
                 batch_size=64):
    """Binary mask (H, W) uint8 at the given threshold."""
    prob = predict_prob_full(image, model, device, patch_size, stride, batch_size)
    return (prob >= threshold).astype(np.uint8)
