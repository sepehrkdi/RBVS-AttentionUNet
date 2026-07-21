"""Explainability: attention-gate heatmaps and Grad-CAM for the Attention U-Net.

Full-fundus single pass: reflect-pad the image up to a multiple of 16 (so the 4
downsamplings are clean), forward once, crop back. Attention maps come straight
from the model's gates (self.alpha); Grad-CAM targets the mean vessel logit over
predicted-vessel pixels and back-props to the last decoder feature.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch


def _prep(image, device, multiple=16):
    """Return (input tensor (1,1,Hp,Wp), (H,W), (Hp,Wp)); image reflect-padded to
    a multiple of `multiple` in each dim."""
    img = np.asarray(image).astype("float32")
    if img.ndim == 3:
        img = img[..., 0]
    if img.max() > 1.0:
        img /= 255.0
    H, W = img.shape[:2]
    Hp = math.ceil(H / multiple) * multiple
    Wp = math.ceil(W / multiple) * multiple
    padded = np.pad(img, ((0, Hp - H), (0, Wp - W)), mode="reflect")
    t = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
    return t, (H, W), (Hp, Wp)


def _minmax(a, eps=1e-8):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + eps)


@torch.no_grad()
def attention_maps(image, model, device):
    """Return (prob HxW float, maps) where maps is a list of 4 attention maps,
    each resized to (H, W) and min-max normalized to [0,1]. Order is gate 1
    (deepest) -> gate 4 (shallowest)."""
    t, (H, W), (Hp, Wp) = _prep(image, device)
    model.eval()
    out = model(t, return_features=True)
    prob = out["prob"][0, 0, :H, :W].cpu().numpy()
    maps = []
    for a in out["attn"]:
        m = a[0, 0].cpu().numpy()                                  # (hp_i, wp_i)
        m = cv2.resize(m, (Wp, Hp), interpolation=cv2.INTER_LINEAR)  # -> padded full
        maps.append(_minmax(m[:H, :W]))
    return prob, maps


def grad_cam(image, model, device, thresh=0.5, topk=4096):
    """Return (cam HxW in [0,1], prob HxW). Target = mean vessel_logits over
    pixels with prob>thresh (top-k highest logits if none), back-propagated to
    decoder_last_conv, GAP-weighted ReLU CAM."""
    t, (H, W), (Hp, Wp) = _prep(image, device)
    model.eval()
    out = model(t, return_features=True)
    logits, prob, feat = out["logits"], out["prob"], out["feat"]
    feat.retain_grad()                                    # non-leaf: keep its grad

    mask = prob > thresh
    if bool(mask.any()):
        target = logits[mask].mean()
    else:
        flat = logits.flatten()
        k = min(topk, flat.numel())
        target = torch.topk(flat, k).values.mean()

    model.zero_grad(set_to_none=True)
    target.backward()

    grads = feat.grad                                     # (1,C,Hp,Wp)
    weights = grads.mean(dim=(2, 3), keepdim=True)        # GAP over spatial
    cam = torch.relu((weights * feat).sum(dim=1, keepdim=True))  # (1,1,Hp,Wp)
    cam = cam[0, 0].detach().cpu().numpy()[:H, :W]
    prob_np = prob[0, 0, :H, :W].detach().cpu().numpy()
    return _minmax(cam), prob_np


def overlay_heatmap(gray_image, heat, alpha=0.55):
    """JET-colormap the [0,1] heat and alpha-blend over the grayscale image.
    Returns an (H, W, 3) float RGB image in [0,1]."""
    g = np.asarray(gray_image).astype("float32")
    if g.ndim == 3:
        g = g[..., 0]
    if g.max() > 1.0:
        g /= 255.0
    g_rgb = np.stack([g, g, g], axis=-1)
    heat_u8 = np.uint8(np.clip(heat, 0, 1) * 255)
    jet = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    jet = jet.astype("float32") / 255.0
    blend = alpha * jet + (1.0 - alpha) * g_rgb
    return np.clip(blend, 0.0, 1.0)
