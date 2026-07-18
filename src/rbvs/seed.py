"""Centralized, explicit seeding for reproducible runs.

All randomness routes through set_seeds(seed); no module sets a global seed at
import time. Entry-point scripts call set_seeds() once near the top with an
explicit seed (default 42) and set PYTHONHASHSEED in the environment before
Python starts (see scripts/train.py). The original notebook was unseeded, which
is why its exact numbers are unrecoverable; this makes future runs reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) from a single value.

    deterministic=True configures cuDNN for reproducible convolutions (benchmark
    off, deterministic on). Bitwise cross-machine reproducibility is still not
    guaranteed on GPU, but same-machine runs match.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
