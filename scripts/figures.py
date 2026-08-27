#!/usr/bin/env python
"""Generate the README figures that come from committed CSVs rather than from the model.

Writes to Figures/:
  training_curves.png         - train vs validation loss per epoch, marking the kept checkpoint
  comparison_thin_vessel.png  - recall by vessel width, this repo vs the sibling ALoG pipeline

Neither needs the DRIVE dataset or the trained weights: both read committed CSVs.
The attention / Grad-CAM figures come from scripts/explain.py, and the thin-vessel
plot from scripts/analyze.py.
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
FIGURES = REPO / "Figures"

# Sibling repo's committed thin-vessel recall, for the cross-repo comparison figure.
# Source: retinal-vessel-segmentation-alog/Figures/thin_vessel_table.csv
# Both tables are pooled over identical GT pixel counts, so the bins are directly comparable.
SIBLING_RECALL = {"[1,2)": 0.553680, "[2,4)": 0.832624, "[4,6)": 0.791851, "[6,inf)": 0.776492}


def read_thin_table():
    """Rows of Figures/thin_vessel_table.csv. The width_bin field contains a comma,
    so split from the right rather than with csv.reader."""
    f = FIGURES / "thin_vessel_table.csv"
    if not f.exists():
        return []
    cols = ["bin", "gt", "tp", "fn", "fp", "recall", "precision", "f1"]
    return [dict(zip(cols, line.rsplit(",", 7))) for line in f.read_text().splitlines()[1:]]


def fig_training_curves():
    """Train and validation loss per epoch, with the retained checkpoint marked.

    The validation curve is deliberately not smoothed: its jaggedness is the visible
    cost of the original run's 1% validation split, and it is why this repo monitors
    val_loss rather than val_accuracy.
    """
    f = FIGURES / "training_history.csv"
    if not f.exists():
        return
    rows = list(csv.DictReader(f.open()))
    ep = [int(r["epoch"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    vloss = [float(r["val_loss"]) for r in rows]
    best = min(range(len(rows)), key=lambda i: vloss[i])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(ep, loss, "-o", ms=4, color="#2b7bba", label="training loss")
    ax.plot(ep, vloss, "-o", ms=4, color="#c0392b", label="validation loss (1% of patches)")
    ax.axvline(ep[best], color="#444", ls="--", lw=1.2, zorder=0)
    ax.annotate(f"checkpoint kept:\nepoch {ep[best]}, best val_loss",
                (ep[best], vloss[best]), textcoords="offset points", xytext=(-14, 58),
                ha="right", fontsize=9.5, weight="bold", color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.3, shrinkB=7))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("focal(0.9, 7) + (1 - Dice)")
    ax.set_title(f"Training the Attention U-Net on 63,800 patches (seed 42, {len(ep)} epochs)")
    ax.set_xticks(ep)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png", dpi=140)
    plt.close(fig)


def fig_comparison():
    """This repo vs the sibling ALoG pipeline: recall by ground-truth vessel width."""
    rows = read_thin_table()
    if not rows:
        return
    labels = ["1-2 px\n(capillaries)", "2-4 px", "4-6 px", "6+ px\n(major vessels)"]
    mine = [float(r["recall"]) for r in rows]
    theirs = [SIBLING_RECALL[r["bin"]] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    b1 = ax.bar(x - w / 2, theirs, w, label="ALoG: 3 hand-set numbers, no training", color="#e08a3c")
    b2 = ax.bar(x + w / 2, mine, w, label="Attention U-Net: 40.4 M learned weights", color="#2b7bba")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                    f"{r.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(rows)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall  (fraction of true vessel pixels found)")
    ax.set_title("Same dataset, same test split: what 40 million parameters actually buy")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "comparison_thin_vessel.png", dpi=140)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    fig_training_curves()
    fig_comparison()
    print("[figures] wrote training_curves, comparison_thin_vessel to Figures/")


if __name__ == "__main__":
    main()
