#!/usr/bin/env python
"""Download the trained Attention U-Net weights from the GitHub Release asset.

Weights are never committed to the repo (see .gitignore); they are distributed as
a Release asset. Set the asset URL below (or pass --url) after creating the
release, then:

    python scripts/fetch_weights.py

Verifies the SHA256 against checkpoints/attention_unet.sha256.txt when present.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
CKPT_DIR = REPO / "checkpoints"

# Fill in after publishing the release (tag/vX + asset filename):
DEFAULT_URL = ("https://github.com/sepehrkdi/RBVS-AttentionUNet/releases/"
               "download/v0.1/attention_unet.weights.pt")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_sha(name: str):
    f = CKPT_DIR / "attention_unet.sha256.txt"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        digest, _, fname = line.partition("  ")
        if fname.strip() == name:
            return digest.strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default=str(CKPT_DIR / "attention_unet.weights.pt"))
    args = ap.parse_args()

    if "releases/download/v0.1" in args.url and "sepehrkdi" in args.url:
        print("[fetch] NOTE: using the placeholder Release URL. Publish the "
              "release and/or pass --url if this 404s.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] downloading {args.url}\n        -> {out}")
    try:
        urllib.request.urlretrieve(args.url, out)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ABORT: download failed ({e}). Check the URL / release.")

    exp = expected_sha(out.name)
    if exp:
        got = sha256_file(out)
        status = "OK" if got == exp else "MISMATCH"
        print(f"[fetch] SHA256 {status}: {got}")
        if got != exp:
            sys.exit("ABORT: checksum mismatch — the asset may be corrupt or stale.")
    print("[fetch] done.")


if __name__ == "__main__":
    main()
