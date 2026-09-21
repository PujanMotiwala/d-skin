"""Reference-card normalisation and skin colour metrics on LINEAR RGB."""
from __future__ import annotations
import numpy as np
import cv2
from . import config as C

# --- reference card -------------------------------------------------------

def find_gray_card(lin: np.ndarray, exclude_box=None) -> dict | None:
    """Locate a neutral grey card: a large, flat, unsaturated quad.

    Returns dict with bbox and the card's mean linear RGB, or None. The card's
    error is independent of your skin, which is exactly why it beats estimating
    the illuminant from the face itself.
    """
    h, w = lin.shape[:2]
    disp = np.clip(lin ** (1 / 2.2), 0, 1)
    hsv = cv2.cvtColor((disp * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    gray = cv2.cvtColor((disp * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flat = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    flat = cv2.blur(np.abs(flat) / 255.0, (15, 15))

    mask = ((sat < C.CARD_MAX_SAT) & (flat < C.CARD_MAX_TEXTURE)
            & (val > 0.12) & (val < 0.97)).astype(np.uint8)
    if exclude_box is not None:
        x, y, bw, bh = exclude_box
        pad = int(0.15 * max(bw, bh))
        cv2.rectangle(mask, (max(0, x - pad), max(0, y - pad)),
                      (min(w, x + bw + pad), min(h, y + bh + pad)), 0, -1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = None, C.CARD_MIN_AREA_FRAC * h * w
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= best_area:
            continue
        bw_, bh_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area / float(bw_ * bh_) < 0.6:   # must actually fill its bbox
            continue
        best_area, best = area, i
    if best is None:
        return None

    sel = lab == best
    # Erode so the card's edge and any shadow on it stay out of the average.
    sel = cv2.erode(sel.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    if sel.sum() < 200:
        return None
    return {
        "bbox": [int(stats[best, cv2.CC_STAT_LEFT]), int(stats[best, cv2.CC_STAT_TOP]),
                 int(stats[best, cv2.CC_STAT_WIDTH]), int(stats[best, cv2.CC_STAT_HEIGHT])],
        "rgb": lin[sel].mean(axis=0).astype(float).tolist(),
        "n_px": int(sel.sum()),
    }


def normalize_with_card(lin: np.ndarray, card_rgb, card_reflectance: float = 0.18):
    """White-balance AND exposure-normalise so the card reads its true reflectance.

    This is the step a colour-constancy model cannot do: AWB recovers illuminant
    chromaticity only and is scale-invariant, so it leaves absolute level — and
    therefore melanin index and L* — uncorrected.
    """
    card = np.asarray(card_rgb, dtype=np.float64)
    if np.any(card <= 1e-6):
        raise ValueError("card patch too dark to normalise against")
    gain = card_reflectance / card
    return (lin * gain[None, None, :]).astype(np.float32), gain.tolist()


def normalize_gray_world(lin: np.ndarray, mask: np.ndarray | None = None):
    """Fallback illuminant estimate when no card is present.

    Chromaticity only, level left alone — matching what a learned colour-constancy
    model gives you. Included so the experiment can measure what you lose.
    """
    px = lin[mask] if mask is not None else lin.reshape(-1, 3)
    m = px.reshape(-1, 3).mean(axis=0).astype(np.float64)
    if np.any(m <= 1e-6):
        return lin.copy(), [1.0, 1.0, 1.0]
    gain = m.mean() / m
    return (lin * gain[None, None, :]).astype(np.float32), gain.tolist()


# --- colour metrics -------------------------------------------------------

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.00000, 1.08883])


def linear_rgb_to_lab(lin: np.ndarray) -> np.ndarray:
    xyz = lin.reshape(-1, 3) @ _M_RGB2XYZ.T
    t = xyz / _WHITE[None, :]
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(np.clip(t, 1e-12, None)),
                 t / (3 * d ** 2) + 4.0 / 29.0)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1).reshape(lin.shape).astype(np.float32)


def ita_degrees(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Individual Typology Angle, the standard skin-tone measure."""
    return np.degrees(np.arctan2(L - 50.0, np.maximum(b, 1e-6)))


def melanin_erythema(lin_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dawson/Takiwaki-style log-reflectance indices. Requires LINEAR, calibrated
    reflectance -- on uncalibrated sRGB these numbers are meaningless."""
    r = np.clip(lin_px[:, 0], 1e-4, None)
    g = np.clip(lin_px[:, 1], 1e-4, None)
    mi = 100.0 * np.log10(1.0 / r)
    ei = 100.0 * (np.log10(1.0 / g) - np.log10(1.0 / r))
    return mi, ei
