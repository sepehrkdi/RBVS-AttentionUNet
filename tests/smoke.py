#!/usr/bin/env python
"""Standalone CPU smoke tests for the rbvs library (no dataset/weights needed).

Run:  PYTHONPATH=src python tests/smoke.py
Covers the Phase-A verification checklist: named submodules present, attention
extractor returns 4 maps, dynamic H/W build, random-weights full-image inference
shape, and a clean import of the whole stack. A separate GPU assert is included.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch


def test_named_modules_and_head():
    from rbvs.model import build_attention_unet
    m = build_attention_unet((128, 128, 1))
    # attention gates named up1..up4 each with an AttentionGate; split logit head
    for name in ("up1", "up2", "up3", "up4", "vessel_logits"):
        assert hasattr(m, name), f"missing submodule {name}"
    for i in (1, 2, 3, 4):
        assert hasattr(getattr(m, f"up{i}"), "attn"), f"up{i} has no attention gate"
    out = m(torch.randn(1, 1, 128, 128), return_features=True)
    assert set(out) == {"logits", "prob", "feat", "attn"}
    assert out["logits"].shape == (1, 1, 128, 128)
    assert out["feat"].shape[1] == 64, "decoder_last_conv should be 64-channel"
    print("  [1] named modules + split logit/prob head: OK")


def test_extractor_four_maps():
    from rbvs.model import build_attention_unet, build_attention_extractor
    m = build_attention_unet((128, 128, 1))
    maps = build_attention_extractor(m)(torch.randn(2, 1, 128, 128))
    assert len(maps) == 4, f"expected 4 attention maps, got {len(maps)}"
    res = [tuple(a.shape[-2:]) for a in maps]
    assert res == [(16, 16), (32, 32), (64, 64), (128, 128)], res
    print("  [2] attention extractor returns 4 maps (16/32/64/128): OK")


def test_dynamic_shape_build():
    from rbvs.model import build_attention_unet
    m = build_attention_unet((128, 128, 1)).eval()
    for (h, w) in [(128, 128), (256, 256), (592, 576)]:
        y = m(torch.randn(1, 1, h, w))
        assert y.shape == (1, 1, h, w), (h, w, tuple(y.shape))
    print("  [3] dynamic (H,W) build incl. 592x576 full-fundus: OK")


def test_random_weights_full_image_shape():
    from rbvs.model import build_attention_unet
    from rbvs.infer import predict_prob_full
    m = build_attention_unet((128, 128, 1)).eval()
    raw = (np.random.rand(584, 565) * 255).astype(np.uint8)   # DRIVE full size
    prob = predict_prob_full(raw, m, torch.device("cpu"), stride=64)
    assert prob.shape == (584, 565), prob.shape
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    print("  [4] random-weights predict_prob_full -> (584,565): OK")


def test_clean_imports():
    import cv2  # noqa: F401
    import skimage  # noqa: F401
    from skimage.morphology import skeletonize  # noqa: F401
    from scipy import ndimage  # noqa: F401
    import matplotlib  # noqa: F401
    from rbvs import data, model, losses, metrics, infer, explain, analysis, seed  # noqa: F401
    print("  [5] clean import of full stack (torch/cv2/skimage/rbvs.*): OK")


def test_gpu_available():
    assert torch.cuda.is_available(), "no CUDA GPU visible to torch"
    print(f"  [GPU] cuda available: {torch.cuda.get_device_name(0)}: OK")


def main():
    print("rbvs smoke tests")
    test_named_modules_and_head()
    test_extractor_four_maps()
    test_dynamic_shape_build()
    test_random_weights_full_image_shape()
    test_clean_imports()
    test_gpu_available()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
