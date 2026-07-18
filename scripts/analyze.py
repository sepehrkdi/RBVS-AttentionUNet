#!/usr/bin/env python
"""Thin-vessel width-stratified analysis on the DRIVE test split.

Reuses the cached probability maps from evaluate.py (artifacts/prob_<ID>.npy),
thresholds at 0.5, and pools recall / precision / F1 by GT vessel-width bin
([1,2), [2,3), [3,5), [5,inf) px). Writes Figures/thin_vessel_table.csv, a
recall-vs-width bar plot, and a markdown section appended to results.md.

Asserts recall increases with vessel width (thin vessels are the hard case).
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
from rbvs.analysis import BIN_LABELS, pool_scores, stratified_counts
from rbvs.infer import predict_prob_full

REPO = pathlib.Path(__file__).resolve().parent.parent
FIGURES = REPO / "Figures"
ARTIFACTS = REPO / "artifacts"
CKPT_DIR = REPO / "checkpoints"


def _load_or_predict(iid, raw, weights, threshold):
    cache = ARTIFACTS / f"prob_{iid}.npy"
    if cache.exists():
        prob = np.load(cache)
    else:  # fall back to a fresh prediction if the cache is absent
        from rbvs.model import build_attention_unet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_attention_unet((128, 128, 1)).to(device)
        obj = torch.load(weights, map_location=device, weights_only=False)
        state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
        model.load_state_dict(state)
        prob = predict_prob_full(raw, model, device, stride=64)
        np.save(cache, prob.astype(np.float32))
    return (prob >= threshold).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(CKPT_DIR / "attention_unet.weights.pt"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    per_image = []
    for img_fp, msk_fp in D.test_pairs():
        iid = img_fp.split("/")[-1].split("_")[0]
        raw = D.load_image(img_fp)
        gt = (D.load_mask(msk_fp).squeeze() > 0).astype(np.uint8)
        pred = _load_or_predict(iid, raw, args.weights, args.threshold)
        per_image.append(stratified_counts(gt, pred))
        print(f"[analyze] image {iid} done", flush=True)

    rows = pool_scores(per_image)

    # CSV
    csv = ["width_bin,gt_pixels,tp,fn,fp,recall,precision,f1"]
    for r in rows:
        csv.append(f"{r['label']},{r['gt']},{r['tp']},{r['fn']},{r['fp']},"
                   f"{r['recall']:.6f},{r['precision']:.6f},{r['f1']:.6f}")
    (FIGURES / "thin_vessel_table.csv").write_text("\n".join(csv) + "\n")

    # bar plot: recall vs width bin
    recalls = [r["recall"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(BIN_LABELS, recalls, color="#2b7bba")
    ax.set_xlabel("Vessel width (px)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.set_title("Recall vs vessel width (thin-vessel analysis)")
    for i, v in enumerate(recalls):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "thin_vessel_recall.png", dpi=150)
    plt.close(fig)

    # markdown section appended to results.md
    md = ["", "## Thin-vessel analysis", "",
          "Recall / precision / F1 pooled over the 20 test images, stratified by "
          "ground-truth vessel width. False positives are attributed to the "
          "nearest vessel's width bin.", "",
          "| Width (px) | GT px | Recall | Precision | F1 |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['gt']} | {r['recall']:.4f} | "
                  f"{r['precision']:.4f} | {r['f1']:.4f} |")
    md += ["", "![Recall vs vessel width](Figures/thin_vessel_recall.png)", ""]
    results = REPO / "results.md"
    prev = results.read_text() if results.exists() else "# Results\n"
    results.write_text(prev + "\n".join(md) + "\n")

    print("[analyze] recall by bin:", [f"{r:.4f}" for r in recalls])

    # assertion: recall increases with vessel width (allow tiny numerical ties)
    clean = [r for r in recalls if not np.isnan(r)]
    for a, b in zip(clean, clean[1:]):
        assert b >= a - 1e-6, (
            f"recall not non-decreasing across width bins: {recalls}")
    print("[analyze] OK: recall is non-decreasing with vessel width.")


if __name__ == "__main__":
    main()
