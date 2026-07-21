#!/usr/bin/env python
"""Explainability figures for the trained Attention U-Net.

For the best / median / worst-IoU test images (or explicit IDs) writes:
  * Figures/attention_<tag>_<ID>.png  — raw + 4 per-gate attention overlays + mean
  * Figures/gradcam_<tag>_<ID>.png    — raw + predicted prob + Grad-CAM overlay
Optional (--occlusion): a coarse occlusion-sensitivity map per image.

Attention gate 1 is deepest (coarse, vessel-tree localization); gate 4 is
shallowest (fine, thin-vessel detail). Grad-CAM targets the mean vessel logit
over predicted-vessel pixels, back-propagated to the last decoder feature.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rbvs import data as D
from rbvs.explain import attention_maps, grad_cam, overlay_heatmap
from rbvs.infer import predict_prob_full
from rbvs.metrics import compute_metrics
from rbvs.model import build_attention_unet

REPO = pathlib.Path(__file__).resolve().parent.parent
FIGURES = REPO / "Figures"
CKPT_DIR = REPO / "checkpoints"


def load_model(weights, device):
    model = build_attention_unet((128, 128, 1)).to(device)
    obj = torch.load(weights, map_location=device, weights_only=False)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state)
    model.eval()
    return model


def select_images(spec):
    """Map a spec like 'best,median,worst' or '01,08' to [(tag, id, img_fp, msk_fp)]."""
    pairs = {img.split("/")[-1].split("_")[0]: (img, msk) for img, msk in D.test_pairs()}
    keywords = {"best", "median", "worst"}
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if not (set(tokens) & keywords):
        return [(iid, iid, *pairs[iid]) for iid in tokens if iid in pairs]

    # rank by IoU using the eval CSV if present, else compute
    csv = FIGURES / "metrics_per_image.csv"
    ious = {}
    if csv.exists():
        for line in csv.read_text().splitlines()[1:]:
            parts = line.split(",")
            if parts[0] in pairs:
                ious[parts[0]] = float(parts[2])
    if not ious:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(CKPT_DIR / "attention_unet.weights.pt", device)
        for iid, (img, msk) in pairs.items():
            prob = predict_prob_full(D.load_image(img), model, device, stride=64)
            gt = (D.load_mask(msk).squeeze() > 0).astype(np.uint8)
            _, iou, _ = compute_metrics(gt, (prob >= 0.5).astype(np.uint8))
            ious[iid] = iou
    ranked = sorted(ious, key=ious.get)
    picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}
    return [(tag, picks[tag], *pairs[picks[tag]]) for tag in ("best", "median", "worst")
            if tag in tokens]


def attention_figure(raw, maps, tag, iid):
    titles = ["gate 1 (deepest)", "gate 2", "gate 3", "gate 4 (shallowest)"]
    mean_map = np.mean(np.stack(maps, 0), 0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(raw, cmap="gray"); axes[0, 0].set_title(f"{tag} #{iid}: input")
    for k in range(4):
        ax = axes.flat[k + 1]
        ax.imshow(overlay_heatmap(raw, maps[k]))
        ax.set_title(f"attention {titles[k]}")
    axes[1, 2].imshow(overlay_heatmap(raw, mean_map)); axes[1, 2].set_title("mean attention")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    out = FIGURES / f"attention_{tag}_{iid}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def gradcam_figure(raw, cam, prob, tag, iid):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(raw, cmap="gray"); axes[0].set_title(f"{tag} #{iid}: input")
    axes[1].imshow(prob, cmap="magma"); axes[1].set_title("predicted P(vessel)")
    axes[2].imshow(overlay_heatmap(raw, cam)); axes[2].set_title("Grad-CAM")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = FIGURES / f"gradcam_{tag}_{iid}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


@torch.no_grad()
def occlusion_map(raw, model, device, window=48, stride=48):
    """Coarse occlusion sensitivity: drop in mean vessel prob when a gray window
    covers each grid cell (full-fundus single pass). Opt-in; O(grid) forwards."""
    from rbvs.explain import _prep
    base = predict_prob_full(raw, model, device, stride=64).mean()
    H, W = np.asarray(raw).shape[:2]
    heat = np.zeros((H, W), np.float32)
    img = np.asarray(raw).astype("float32")
    if img.max() > 1:
        img = img / 255.0
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occ = img.copy()
            occ[y:y + window, x:x + window] = 0.5
            t, (h, w), _ = _prep(occ, device)
            p = model(t)[0, 0, :h, :w].mean().item()
            heat[y:y + window, x:x + window] = base - p
    heat = np.maximum(heat, 0)
    return heat / (heat.max() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(CKPT_DIR / "attention_unet.weights.pt"))
    ap.add_argument("--images", default="best,median,worst")
    ap.add_argument("--occlusion", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = pathlib.Path(args.weights)
    if not weights.exists():
        sys.exit(f"ABORT: weights not found at {weights}. Train first.")
    model = load_model(weights, device)
    FIGURES.mkdir(parents=True, exist_ok=True)

    written = []
    for tag, iid, img_fp, msk_fp in select_images(args.images):
        raw = D.load_image(img_fp)
        prob, maps = attention_maps(raw, model, device)
        written.append(attention_figure(raw, maps, tag, iid))
        cam, prob2 = grad_cam(raw, model, device)
        written.append(gradcam_figure(raw, cam, prob2, tag, iid))
        if args.occlusion:
            occ = occlusion_map(raw, model, device)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(overlay_heatmap(raw, occ)); ax.axis("off")
            ax.set_title(f"occlusion {tag} #{iid}")
            out = FIGURES / f"occlusion_{tag}_{iid}.png"
            fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
            written.append(out)
        print(f"[explain] {tag} #{iid} done", flush=True)

    print("[explain] wrote:")
    for w in written:
        print("  ", w.relative_to(REPO))


if __name__ == "__main__":
    main()
