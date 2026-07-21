"""DRIVE dataset loading, patch extraction, and training-set assembly.

Framework-agnostic (returns NumPy arrays in NHWC); the training script converts
to torch tensors. Faithful to CV_Project.ipynb: images are read as grayscale
(the notebook's green-channel/CLAHE paths were commented out) and masks are
thresholded at 128. Crucially, masks are read from the original .gif files IN
MEMORY via PIL — we never run the notebook's destructive in-place gif->tif
conversion (convert_gif_to_tif, cells 9/17), which corrupted the annotations.
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
from PIL import Image

# DRIVE root on this machine. Single-level nesting (no doubled DRIVE/DRIVE):
#   DRIVE/{training,test}/images/*_{training,test}.tif
#   DRIVE/{training,test}/1st_manual/*_manual1.gif
# The sibling mask/ (FOV) and test/2nd_manual/ dirs are intentionally ignored,
# and the stray training/t.txt is skipped by the *.tif suffix glob.
DRIVE_ROOT = "/home/msc_student/sepehr/DRIVE"

TRAIN_IMAGE_DIR = os.path.join(DRIVE_ROOT, "training", "images")
TRAIN_MASK_DIR = os.path.join(DRIVE_ROOT, "training", "1st_manual")
TEST_IMAGE_DIR = os.path.join(DRIVE_ROOT, "test", "images")
TEST_MASK_DIR = os.path.join(DRIVE_ROOT, "test", "1st_manual")

PATCH_SIZE = 128
STRIDE = 8
# 20 training images of 584x565 -> 58 x 55 patch grid at stride 8 -> 3190 each.
EXPECTED_PATCHES = 63800


def load_image(filepath: str) -> np.ndarray:
    """Load a fundus image as single-channel grayscale, shape (H, W) uint8.

    Matches the notebook's load_image (cv2.IMREAD_GRAYSCALE).
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {filepath}")
    return img


def load_mask(filepath: str) -> np.ndarray:
    """Load a manual annotation as a binary mask, shape (H, W, 1) uint8 in {0,1}.

    Reads the original .gif in memory with PIL (OpenCV handles GIF poorly, which
    is why the notebook pre-converted to TIF on disk). We never convert on disk.
    """
    with Image.open(filepath) as im:
        arr = np.array(im.convert("L"))
    mask = (arr > 128).astype(np.uint8)
    return mask[..., np.newaxis]


def load_and_prep_image(filepath: str) -> np.ndarray:
    """(H, W, 1) uint8 image."""
    img = load_image(filepath)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    return img.astype("uint8")


def load_and_prep_mask(filepath: str) -> np.ndarray:
    """(H, W, 1) uint8 mask in {0,1}."""
    msk = load_mask(filepath)
    if msk.ndim == 2:
        msk = msk[..., np.newaxis]
    return msk


def extract_patches(img, mask, patch_size=PATCH_SIZE, stride=STRIDE):
    """Slide a patch_size window with the given stride over (H,W,1) img/mask.

    Verbatim from the notebook (cell 32): returns (N,128,128,1) arrays.
    """
    H, W = img.shape[:2]
    patches_img, patches_msk = [], []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch_img = img[y:y + patch_size, x:x + patch_size]
            patch_msk = mask[y:y + patch_size, x:x + patch_size]
            assert patch_img.shape == (patch_size, patch_size, 1)
            assert patch_msk.shape == (patch_size, patch_size, 1)
            patches_img.append(patch_img)
            patches_msk.append(patch_msk)
    return np.stack(patches_img, axis=0), np.stack(patches_msk, axis=0)


def _sorted_pairs(image_dir, mask_dir, image_glob, mask_glob="*_manual1.gif"):
    """Sorted (image_path, mask_path) pairs. The suffix globs skip stray files
    (e.g. training/t.txt) and the sibling FOV mask/ and 2nd_manual/ dirs."""
    imgs = sorted(glob.glob(os.path.join(image_dir, image_glob)))
    msks = sorted(glob.glob(os.path.join(mask_dir, mask_glob)))
    if not imgs:
        raise RuntimeError(f"no images matched {image_glob} in {image_dir}")
    if len(imgs) != len(msks):
        raise RuntimeError(
            f"image/mask count mismatch: {len(imgs)} images vs {len(msks)} masks "
            f"({image_dir} / {mask_dir})"
        )
    return list(zip(imgs, msks))


def train_pairs():
    """20 (image, mask) path pairs for the DRIVE training split."""
    return _sorted_pairs(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, "*_training.tif")


def test_pairs():
    """20 (image, mask) path pairs for the DRIVE test split."""
    return _sorted_pairs(TEST_IMAGE_DIR, TEST_MASK_DIR, "*_test.tif")


def build_training_set(patch_size=PATCH_SIZE, stride=STRIDE, assert_shape=True):
    """Assemble the full patch training set from the 20 DRIVE training images.

    Returns (X, y) as float32 NHWC arrays: X normalized to [0,1], y in {0,1}.
    Asserts X.shape == (63800, 128, 128, 1) unless assert_shape=False.
    """
    all_X, all_y = [], []
    for img_fp, msk_fp in train_pairs():
        img = load_and_prep_image(img_fp)
        msk = load_and_prep_mask(msk_fp)
        Xp, yp = extract_patches(img, msk, patch_size, stride)
        all_X.append(Xp)
        all_y.append(yp)
    X = np.concatenate(all_X, axis=0).astype("float32") / 255.0
    y = np.concatenate(all_y, axis=0).astype("float32")
    if assert_shape:
        expected = (EXPECTED_PATCHES, patch_size, patch_size, 1)
        assert X.shape == expected, f"unexpected training shape {X.shape}, expected {expected}"
    return X, y
