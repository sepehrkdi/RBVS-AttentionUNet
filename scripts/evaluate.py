#!/usr/bin/env python
"""Evaluate the trained Attention U-Net on the DRIVE test split.

Standardized protocol (cell 74): full-image predict via reflect-pad tile-and-stitch
at stride 64, threshold 0.5. Writes results.md + Figures/metrics_per_image.csv and
caches the averaged probability maps to artifacts/prob_<ID>.npy for reuse by
analyze.py / explain.py. Expected ~96.4 acc / 63.8 IoU / 77.9 F1 on the original
run; the retrained numbers will drift and become the committed truth.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from rbvs import data as D
from rbvs.infer import predict_prob_full
from rbvs.metrics import compute_metrics
from rbvs.model import build_attention_unet

REPO = pathlib.Path(__file__).resolve().parent.parent
FIGURES = REPO / "Figures"
ARTIFACTS = REPO / "artifacts"
CKPT_DIR = REPO / "checkpoints"


def image_id(path: str) -> str:
    """'.../01_test.tif' -> '01'."""
    return os.path.basename(path).split("_")[0]


def load_model(weights: pathlib.Path, device):
    model = build_attention_unet((128, 128, 1)).to(device)
    obj = torch.load(weights, map_location=device, weights_only=False)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(CKPT_DIR / "attention_unet.weights.pt"))
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = pathlib.Path(args.weights)
    if not weights.exists():
        sys.exit(f"ABORT: weights not found at {weights}. Train first (scripts/train.py).")
    model = load_model(weights, device)
    print(f"[eval] device={device} stride={args.stride} threshold={args.threshold}")
    print(f"[eval] weights={weights.name}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_fp, msk_fp in D.test_pairs():
        iid = image_id(img_fp)
        raw = D.load_image(img_fp)                       # (H,W) uint8 grayscale
        gt = (D.load_mask(msk_fp).squeeze() > 0).astype(np.uint8)
        prob = predict_prob_full(raw, model, device, patch_size=128,
                                 stride=args.stride)
        np.save(ARTIFACTS / f"prob_{iid}.npy", prob.astype(np.float32))
        pred = (prob >= args.threshold).astype(np.uint8)
        acc, iou, f1 = compute_metrics(gt, pred)
        rows.append((iid, acc, iou, f1))
        print(f"[eval] image {iid}: acc={acc:.4f} iou={iou:.4f} f1={f1:.4f}", flush=True)

    accs = np.array([r[1] for r in rows])
    ious = np.array([r[2] for r in rows])
    f1s = np.array([r[3] for r in rows])
    mean = (accs.mean(), ious.mean(), f1s.mean())
    std = (accs.std(), ious.std(), f1s.std())

    # per-image CSV
    csv = ["image_id,accuracy,iou,f1"]
    csv += [f"{iid},{a:.6f},{i:.6f},{f:.6f}" for (iid, a, i, f) in rows]
    csv.append(f"mean,{mean[0]:.6f},{mean[1]:.6f},{mean[2]:.6f}")
    csv.append(f"std,{std[0]:.6f},{std[1]:.6f},{std[2]:.6f}")
    (FIGURES / "metrics_per_image.csv").write_text("\n".join(csv) + "\n")

    # results.md (evaluation section)
    md = [
        "# Results",
        "",
        f"Attention U-Net on the DRIVE test split (20 images). Protocol: full-image "
        f"reflect-pad tile-and-stitch, patch 128, stride {args.stride}, threshold "
        f"{args.threshold}; metrics over all pixels (no FOV masking).",
        "",
        f"Weights: `{weights.name}`.",
        "",
        "| Metric | Mean | Std |",
        "|---|---|---|",
        f"| Accuracy | {mean[0]:.4f} | {std[0]:.4f} |",
        f"| IoU | {mean[1]:.4f} | {std[1]:.4f} |",
        f"| F1 (Dice) | {mean[2]:.4f} | {std[2]:.4f} |",
        "",
        "Per-image numbers: [Figures/metrics_per_image.csv](Figures/metrics_per_image.csv).",
        "",
        "_Original TF/Keras run reported 0.9639 / 0.6383 / 0.7787 (acc/IoU/F1). "
        "The original was unseeded and its checkpoint was lost; these retrained "
        "numbers are the committed truth and may drift from the originals._",
        "",
    ]
    (REPO / "results.md").write_text("\n".join(md) + "\n")

    print(f"\n[eval] MEAN  acc={mean[0]:.4f}  iou={mean[1]:.4f}  f1={mean[2]:.4f}")
    print(f"[eval] STD   acc={std[0]:.4f}  iou={std[1]:.4f}  f1={std[2]:.4f}")
    print("[eval] wrote results.md, Figures/metrics_per_image.csv, artifacts/prob_*.npy")


if __name__ == "__main__":
    main()
