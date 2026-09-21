"""Build a synthetic experiment from one portrait, with KNOWN perturbations.

Self-test: you know the ground truth here, so you can confirm the pipeline
recovers it before you point the tool at your own face. Run this after install.

  python tools/make_synthetic_experiment.py <portrait.jpg> out/synthetic
  python -m dskin experiment out/synthetic
"""
from __future__ import annotations
import os
import sys
import numpy as np
import cv2

rng = np.random.default_rng(7)


def add_card(bgr: np.ndarray, reflectance: float = 0.18) -> np.ndarray:
    """Paste a neutral grey card into a corner, in LINEAR terms."""
    out = bgr.copy()
    h, w = out.shape[:2]
    ch, cw = int(0.16 * h), int(0.16 * h)
    y0, x0 = int(0.72 * h), int(0.04 * w)
    srgb = (reflectance ** (1 / 2.4) * (1.055) - 0.055)
    val = np.clip(srgb, 0, 1) * 255
    out[y0:y0 + ch, x0:x0 + cw] = val
    return out


def perturb(bgr: np.ndarray, kind: str, i: int) -> np.ndarray:
    """Apply a perturbation in LINEAR light, then re-encode."""
    lin = (bgr.astype(np.float32) / 255.0) ** 2.2

    if kind == "baseline":
        lin *= 1.0 + rng.normal(0, 0.004)
        lin += rng.normal(0, 0.0015, lin.shape).astype(np.float32)

    elif kind == "lighting":
        # Warm/cool cast PLUS an exposure change. The exposure part is what a
        # colour-constancy model cannot recover: AWB is scale-invariant.
        warm = rng.uniform(-1, 1)
        gains = np.array([1 - 0.16 * warm, 1.0, 1 + 0.16 * warm], np.float32)  # BGR
        lin *= gains[None, None, :] * rng.uniform(0.75, 1.30)

    elif kind == "posedist":
        h, w = lin.shape[:2]
        ang, scale = rng.uniform(-6, 6), rng.uniform(0.90, 1.10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
        lin = cv2.warpAffine(lin, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        # distance change also shifts illumination: inverse-square falloff
        lin *= scale ** 2

    lin = np.clip(lin, 0, 1)
    return (np.clip(lin ** (1 / 2.2), 0, 1) * 255).astype(np.uint8)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, outdir = sys.argv[1], sys.argv[2]
    base = cv2.imread(src)
    if base is None:
        print(f"cannot read {src}")
        return 1
    base = add_card(base)

    for kind in ("baseline", "lighting", "posedist"):
        d = os.path.join(outdir, kind)
        os.makedirs(d, exist_ok=True)
        for i in range(10):
            cv2.imwrite(os.path.join(d, f"{kind}_{i:02d}.png"), perturb(base, kind, i))
        print(f"  {kind}: 10 images -> {d}")
    print(f"\nnow run:  python -m dskin experiment {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
