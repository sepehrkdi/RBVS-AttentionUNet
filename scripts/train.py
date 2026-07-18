#!/usr/bin/env python
"""Train the Attention U-Net on DRIVE (local GPU).

Primary retrain path. Faithful to CV_Project.ipynb (cells 46/48): Adam @ default
1e-3, batch 8, validation_split=0.01, 20 epochs, loss = focal(0.9, 7) + (1-Dice),
no early stopping (the notebook defined EarlyStopping but never passed it to fit).

Deviations from the notebook, all deliberate and documented:
  * seeded (PYTHONHASHSEED + set_seeds) — the original was unseeded, so its exact
    numbers are unrecoverable; these become the committed truth.
  * ModelCheckpoint monitors val_loss (the notebook used val_accuracy; val_loss is
    the better early-signal for a heavily class-imbalanced segmentation task).
  * training history is persisted (CSV + JSON) so the loss curve is never lost
    again (the notebook lost its history and hand-parsed it from the log).

Usage:
  env PYTHONHASHSEED=42 python -u scripts/train.py            # full 20-epoch run
  env PYTHONHASHSEED=42 python -u scripts/train.py --smoke    # 1-epoch mini-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rbvs import data as D
from rbvs.losses import combined_loss
from rbvs.model import build_attention_unet
from rbvs.seed import set_seeds

REPO = pathlib.Path(__file__).resolve().parent.parent
FIGURES = REPO / "Figures"
CKPT_DIR = REPO / "checkpoints"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nhwc_to_tensor(a: np.ndarray) -> torch.Tensor:
    """(N,H,W,1) float32 -> (N,1,H,W) contiguous tensor."""
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def evaluate_split(model, loader, device):
    """Mean combined loss and pixel accuracy (threshold 0.5) over a loader."""
    model.eval()
    tot_loss, tot_correct, tot_px = 0.0, 0, 0
    n_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        prob = model(xb)
        tot_loss += float(combined_loss(prob, yb))
        n_batches += 1
        tot_correct += int(((prob >= 0.5).float() == yb).sum())
        tot_px += yb.numel()
    return tot_loss / max(n_batches, 1), tot_correct / max(tot_px, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="1-epoch mini-run on a 2000-patch subset (GPU/pipeline gate)")
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap number of training patches (0 = all); implied by --smoke")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 1)
        if args.max_train == 0:
            args.max_train = 2000

    set_seeds(args.seed)

    # --- GPU-detect gate -----------------------------------------------------
    if not torch.cuda.is_available():
        sys.exit("ABORT: no CUDA GPU visible to torch. This is the local-GPU "
                 "training path; check the driver / torch install.")
    device = torch.device("cuda")
    print(f"[train] device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"[train] config: epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"val_split={args.val_split} seed={args.seed} smoke={args.smoke}")

    # --- data ----------------------------------------------------------------
    assert_shape = not (args.smoke or args.max_train)
    X, y = D.build_training_set(assert_shape=assert_shape)
    print(f"[train] built training set: X={X.shape} y={y.shape}")
    # Convert and free the NumPy copies incrementally: each array is ~4.2 GB at
    # full size, so holding both representations at once would peak ~16.7 GB.
    X_t = nhwc_to_tensor(X); del X
    y_t = nhwc_to_tensor(y); del y

    # Keras-equivalent validation_split: the LAST val_split fraction is the fixed
    # validation set (chosen before shuffling); only the train part is shuffled.
    n = X_t.shape[0]
    split_at = int(n * (1.0 - args.val_split))
    X_tr, y_tr = X_t[:split_at], y_t[:split_at]
    X_va, y_va = X_t[split_at:], y_t[split_at:]
    if args.max_train:
        X_tr, y_tr = X_tr[:args.max_train], y_tr[:args.max_train]
    print(f"[train] train={X_tr.shape[0]} patches | val={X_va.shape[0]} patches")

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=max(args.batch_size, 32),
                            shuffle=False, num_workers=0, pin_memory=True)

    # --- model / optim -------------------------------------------------------
    model = build_attention_unet((128, 128, 1)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    FIGURES.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    ckpt_full = CKPT_DIR / "attention_unet.pt"
    ckpt_weights = CKPT_DIR / "attention_unet.weights.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, run_correct, run_px, n_batches = 0.0, 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prob = model(xb)
            loss = combined_loss(prob, yb)
            loss.backward()
            optimizer.step()
            run_loss += float(loss.detach())
            n_batches += 1
            with torch.no_grad():
                run_correct += int(((prob >= 0.5).float() == yb).sum())
                run_px += yb.numel()
        tr_loss = run_loss / max(n_batches, 1)
        tr_acc = run_correct / max(run_px, 1)
        va_loss, va_acc = evaluate_split(model, val_loader, device)
        dt = time.time() - t0
        print(f"[epoch {epoch:02d}/{args.epochs}] "
              f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | {dt:.0f}s", flush=True)
        history.append({"epoch": epoch, "loss": tr_loss, "accuracy": tr_acc,
                        "val_loss": va_loss, "val_accuracy": va_acc, "seconds": dt})

        # persist history every epoch (never lose the curve again)
        _dump_history(history)

        # ModelCheckpoint(monitor=val_loss, save_best_only, mode=min)
        if va_loss < best_val:
            best_val = va_loss
            config = {"arch": "attention_unet", "input_shape": [128, 128, 1],
                      "epoch": epoch, "val_loss": va_loss, "seed": args.seed,
                      "loss": "focal(0.9,7)+(1-dice)", "optimizer": "adam",
                      "lr": args.lr, "batch_size": args.batch_size}
            torch.save({"model_state_dict": model.state_dict(), **config}, ckpt_full)
            torch.save(model.state_dict(), ckpt_weights)
            print(f"           checkpoint improved (val_loss={va_loss:.4f}) -> saved", flush=True)

    if not ckpt_full.exists():
        # degenerate (e.g. 0 epochs); still emit a checkpoint
        torch.save({"model_state_dict": model.state_dict()}, ckpt_full)
        torch.save(model.state_dict(), ckpt_weights)

    # --- SHA256 provenance ---------------------------------------------------
    sums = {ckpt_full.name: sha256_file(ckpt_full),
            ckpt_weights.name: sha256_file(ckpt_weights)}
    (CKPT_DIR / "attention_unet.sha256.txt").write_text(
        "\n".join(f"{v}  {k}" for k, v in sums.items()) + "\n")
    print("[train] best val_loss:", f"{best_val:.4f}")
    for k, v in sums.items():
        print(f"[train] SHA256 {k}: {v}")
    print("[train] done.")


def _dump_history(history):
    (FIGURES / "training_history.json").write_text(json.dumps(history, indent=2))
    cols = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "seconds"]
    lines = [",".join(cols)]
    for h in history:
        lines.append(",".join(str(h[c]) for c in cols))
    (FIGURES / "training_history.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
