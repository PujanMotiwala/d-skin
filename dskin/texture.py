"""Texture metrics on NATIVE pixels, with filter scale tied to face size."""
from __future__ import annotations
import numpy as np
import cv2
from . import config as C


def texture_metrics(L: np.ndarray, mask: np.ndarray, iod_px: float) -> dict:
    """Band-pass energy at scales defined in units of inter-ocular distance.

    Adapting the FILTER to apparent face size avoids resampling the image, which
    would blur by a pose-dependent amount. iod_px is still stored as a covariate:
    if a metric tracks it, the metric is measuring your seating position.
    """
    out: dict[str, float] = {}
    if mask.sum() < 50:
        return {f"tex_dog_{i}": float("nan") for i in range(len(C.TEXTURE_SIGMAS_IOD))}

    Lf = np.where(mask, L, np.nan).astype(np.float32)
    filled = np.where(mask, L, float(np.nanmean(Lf))).astype(np.float32)

    for i, k in enumerate(C.TEXTURE_SIGMAS_IOD):
        s1 = max(0.6, k * iod_px)
        s2 = s1 * 1.6
        dog = cv2.GaussianBlur(filled, (0, 0), s1) - cv2.GaussianBlur(filled, (0, 0), s2)
        # Erode the mask by the filter support so the ROI border does not leak in.
        pad = int(np.ceil(3 * s2))
        inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool)
        use = inner if inner.sum() > 50 else mask
        out[f"tex_dog_{i}"] = float(np.sqrt(np.mean(dog[use] ** 2)))

    # Local contrast at a fixed fraction of face size.
    win = max(3, int(0.012 * iod_px) * 2 + 1)
    mu = cv2.blur(filled, (win, win))
    var = cv2.blur(filled * filled, (win, win)) - mu * mu
    out["tex_local_std"] = float(np.sqrt(np.clip(var[mask], 0, None)).mean())

    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    out["tex_grad_mag"] = float(np.hypot(gx, gy)[mask].mean())
    return out


def sharpness_tenengrad(L: np.ndarray, mask: np.ndarray) -> float:
    """Focus measure evaluated on the ROI only -- a sharp background is irrelevant."""
    if mask.sum() < 50:
        return 0.0
    gx = cv2.Sobel(L.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return float((gx[mask] ** 2 + gy[mask] ** 2).mean())
