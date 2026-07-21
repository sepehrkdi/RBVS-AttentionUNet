# Retinal Blood-Vessel Segmentation with an Attention U-Net (PyTorch)

Patch-based **Attention U-Net** for retinal blood-vessel segmentation on the
**DRIVE** dataset. Originally a University of Genoa Computer Vision course project
(2025); this is a clean, reproducible **PyTorch** re-implementation with a `src/`
package, seeded training, explainability artifacts, and honest reporting.

> **Reproducibility note.** The original model was trained in TensorFlow/Keras on
> Colab; that run was **unseeded** and its checkpoint was **lost**. This repo
> re-implements the exact architecture and hyperparameters in PyTorch and
> **retrains from scratch** on a local GPU. The numbers reported in
> [results.md](results.md) come from that seeded retrain and are the committed
> truth — they may drift slightly from the originally reported
> 0.9639 / 0.6383 / 0.7787 (accuracy / IoU / F1). The original TF notebook is kept
> verbatim as [CV_Project.ipynb](CV_Project.ipynb) for provenance.

## Repository structure

```
src/rbvs/            # library
  seed.py            # single-seed reproducibility
  data.py            # DRIVE loading + 128px patch extraction (63,800 patches)
  model.py           # AttentionUNet + BaselineUNet, split logit/prob head
  losses.py          # binary focal (alpha=0.9, gamma=7) + Dice
  metrics.py         # accuracy / IoU / F1
  infer.py           # reflect-pad tile-and-stitch full-image prediction
  explain.py         # attention-gate maps + Grad-CAM
  analysis.py        # thin-vessel width stratification
scripts/
  train.py           # local-GPU retrain (primary path)
  evaluate.py        # test-set metrics -> results.md + CSV
  explain.py         # attention / Grad-CAM figures
  analyze.py         # thin-vessel table + plot
  fetch_weights.py   # download weights from the GitHub Release
notebooks/
  demo.ipynb         # minimal inference demo
  colab_train.ipynb  # no-local-GPU training path (Colab)
Figures/             # committed figures (paper evidence)
results.md           # generated metrics (do not hand-edit)
requirements.txt  LICENSE  .gitignore
CV_Project.ipynb     # original TF/Keras notebook (historical record, untouched)
```

## Dataset

**DRIVE** (Digital Retinal Images for Vessel Extraction): 20 training + 20 test
fundus images at 584×565, with first-observer manual annotations. Images are read
as **grayscale**; masks are read from the original `.gif` in memory (the original
notebook's destructive in-place GIF→TIF conversion is **not** used). Set the
dataset root in [src/rbvs/data.py](src/rbvs/data.py) (`DRIVE_ROOT`).

## Method

- **Architecture.** Attention U-Net: 4-level encoder (64/128/256/512), 1024
  bottleneck, attention-gated skip connections, ~40.4 M parameters. The output
  head is split into `vessel_logits` (Conv 1×1, no activation) → `vessel_prob`
  (sigmoid); numerically identical to a single sigmoid conv but it exposes a logit
  tensor for Grad-CAM. A plain `BaselineUNet` is also provided (see below).
- **Patch sampling.** Overlapping 128×128 patches at stride 8 → **63,800** patches
  from the 20 training images (vessel structure is preserved instead of resizing).
- **Loss.** `binary_focal_loss(alpha=0.9, gamma=7)` on the probability + `(1 - Dice)`.
- **Inference.** Reflect-pad tile-and-stitch at stride 64 with overlap averaging,
  then threshold 0.5.

### Hyperparameters (faithful to the original run)

| Setting | Value | Note |
|---|---|---|
| Optimizer | Adam | Keras default learning rate **1e-3** |
| Batch size | 8 | |
| Epochs | 20 | |
| Early stopping | **none** | the notebook defined `EarlyStopping` but never passed it to `fit`; we match that |
| Validation split | **1%** | last 1% of patches (Keras-equivalent), fixed before shuffling |
| Loss | focal(0.9, 7) + (1−Dice) | |
| Seed | 42 | original run was unseeded |
| Checkpoint monitor | `val_loss` | notebook used `val_accuracy`; `val_loss` is the better signal on this imbalanced task |

These correct several inaccuracies in the original project write-up (which listed
Adam 1e-4, batch 4, patience-5 early stopping, and a 95/5 split — none of which
match the code that produced the reported numbers).

## Results

DRIVE test split (20 images), patch 128 / stride 64 / threshold 0.5, metrics over
all pixels (no field-of-view masking), mean ± std across the 20 images:

| | Accuracy | IoU | F1 (Dice) |
|---|---|---|---|
| **This PyTorch retrain** (seed 42, epoch 16) | **0.9617** ± 0.0032 | **0.6337** ± 0.0286 | **0.7754** ± 0.0215 |
| Original TF/Keras run (unseeded, checkpoint lost) | 0.9639 | 0.6383 | 0.7787 |

All three metrics land within 0.5 pp of the originally reported numbers, across
both a framework port and a seeded re-run. Full table and per-image numbers:
**[results.md](results.md)** and `Figures/metrics_per_image.csv`, both regenerated
by `scripts/evaluate.py` — do not hand-edit.

Training used the best-`val_loss` checkpoint (epoch 16 of 20). Note that eval-mode
metrics are meaningless until the BatchNorm running statistics converge: at epoch 1
the model scores IoU 0.09, and by epoch 2 it is already at 0.63.

The `BaselineUNet` (plain U-Net, no attention) is **provided but not trained** —
there is no measured baseline row here, so no baseline-vs-attention ablation is
claimed. Training it is left as a one-command exercise.

## Explainability

- **Attention gates.** Each decoder level has an additive attention gate.
  In the trained model these do *not* split into the tidy coarse-to-fine
  hierarchy one might expect: gates 1 and 2 (deepest) are close to spatially
  uniform, gate 3 carries almost all of the structure and responds *inversely*
  (low on vessels, high on background), and gate 4 (shallowest) is fine-grained
  and noisy. `scripts/explain.py` writes per-gate and mean-attention overlays for
  the best/median/worst-IoU test images so this is inspectable rather than
  asserted — see `Figures/attention_*.png`.
- **Grad-CAM.** Target = mean `vessel_logits` over predicted-vessel pixels
  (`prob > 0.5`; top-k fallback), back-propagated to the last decoder feature,
  GAP-weighted ReLU CAM.
- **Thin-vessel analysis.** `scripts/analyze.py` skeletonizes each GT mask, assigns
  a vessel width to every pixel, and pools recall/precision/F1 by width bin
  ([1,2), [2,4), [4,6), [6,∞) px). Width is `2×EDT − 1` at the skeleton
  centerline, which inverts the distance transform correctly (a 1-pixel vessel
  measures 1, not 2); the bin edges are chosen so all four bins are populated
  given the grid quantization. Measured: recall climbs steeply from **0.590** on
  the thinnest vessels to **0.844**, then plateaus (dipping marginally to 0.836 on
  the widest bin), while precision and F1 rise monotonically with width. On wide
  vessels the residual false negatives sit at vessel *boundaries*, not centrelines
  (55.8% fall in the outer third, vs 12.6% of true positives), so the error there
  is ordinary boundary under-segmentation rather than a central-reflex effect.

## How to run

### Local GPU (primary path)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
# set DRIVE_ROOT in src/rbvs/data.py, then:
env PYTHONHASHSEED=42 python -u scripts/train.py         # ~2-5 h on an RTX 3060
python scripts/evaluate.py                                # -> results.md, CSV
python scripts/analyze.py                                 # -> thin-vessel table
python scripts/explain.py --images best,median,worst      # -> attention/Grad-CAM PNGs
```

### No local GPU

Use [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) (Colab T4), which
mirrors `scripts/train.py`.

### Weights

Trained weights are **not committed**; they are published as a GitHub Release
asset. After the release exists:

```bash
python scripts/fetch_weights.py     # -> checkpoints/attention_unet.weights.pt
```

## Limitations

- **Threshold sensitivity.** Metrics depend on the 0.5 binarization threshold.
- **1% validation.** A very small, fixed validation set (matching the original).
- **Single annotator.** Only the first-observer masks are used.
- **Retrain drift.** Because the original was unseeded and its checkpoint lost,
  the numbers here are a faithful re-run, not a bit-exact reproduction.

## Dependencies

PyTorch (CUDA 12.8 build), NumPy, OpenCV (headless), scikit-image, SciPy,
Matplotlib, Pillow, tqdm. See [requirements.txt](requirements.txt).

## License and citation

MIT — see [LICENSE](LICENSE).

```bibtex
@techreport{khodadadi2025rbvs,
  title  = {Retinal Blood-Vessel Segmentation with Attention U-Net and Patch Sampling},
  author = {Khodadadi, Sepehr},
  institution = {University of Genoa},
  year   = {2025},
  type   = {Computer Vision Course Project}
}
```
