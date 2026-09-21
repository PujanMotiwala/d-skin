"""Load an image to LINEAR RGB in [0,1].

Every photometric number downstream assumes linear reflectance. sRGB values are
gamma-encoded; averaging them is a silent systematic error, so linearisation
happens here, once, and nothing downstream touches gamma again.
"""
from __future__ import annotations
import os
import numpy as np
import cv2

RAW_EXT = {".dng", ".arw", ".cr2", ".cr3", ".nef", ".raf", ".rw2", ".orf"}


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * x ** (1 / 2.4) - a)


def load_linear_rgb(path: str) -> tuple[np.ndarray, dict]:
    """Return (H,W,3) float32 linear RGB in [0,1], plus provenance metadata."""
    ext = os.path.splitext(path)[1].lower()
    meta: dict = {"path": path, "source": "raw" if ext in RAW_EXT else "ldr"}

    if ext in RAW_EXT:
        import rawpy
        with rawpy.imread(path) as raw:
            # No auto-brightness, no camera WB, linear output: we want measurement,
            # not a pleasant picture. gamma=(1,1) keeps it linear.
            rgb16 = raw.postprocess(
                gamma=(1, 1), no_auto_bright=True, use_camera_wb=False,
                use_auto_wb=False, output_bps=16, user_flip=0,
            )
        lin = rgb16.astype(np.float32) / 65535.0
        meta["bit_depth"] = 16
    else:
        bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise ValueError(f"could not read image: {path}")
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        bgr = bgr[:, :, :3]
        depth = 65535.0 if bgr.dtype == np.uint16 else 255.0
        meta["bit_depth"] = 16 if bgr.dtype == np.uint16 else 8
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / depth
        lin = srgb_to_linear(rgb)

    meta["height"], meta["width"] = lin.shape[:2]
    return np.clip(lin, 0.0, 1.0).astype(np.float32), meta


def to_display_u8(lin: np.ndarray) -> np.ndarray:
    """Linear RGB -> 8-bit sRGB, for overlays and contact sheets only."""
    return (linear_to_srgb(lin) * 255).round().astype(np.uint8)
