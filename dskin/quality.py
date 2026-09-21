"""The capture gate.

What makes Face ID reliable is not the depth sensor, it is that it REFUSES to
proceed until conditions are met. That is a gate, not a model -- and a gate is
cheap. Every rejected frame is a data point you do not have to distrust later.
"""
from __future__ import annotations
import numpy as np
from . import config as C


def gate(face: dict | None, lin, roi_union, sharpness: float, valid_frac: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if face is None:
        return False, ["no_face_detected"]

    for name, val, lim in (("yaw", face["yaw"], C.MAX_YAW_DEG),
                           ("pitch", face["pitch"], C.MAX_PITCH_DEG),
                           ("roll", face["roll"], C.MAX_ROLL_DEG)):
        if not np.isfinite(val):
            reasons.append(f"{name}_unavailable")
        elif abs(val) > lim:
            reasons.append(f"{name}_out_of_range({val:+.1f}deg>{lim})")

    if sharpness < C.MIN_SHARPNESS:
        reasons.append(f"blurry(tenengrad={sharpness:.0f}<{C.MIN_SHARPNESS})")

    if roi_union is not None and roi_union.sum() > 0:
        px = lin[roi_union]
        clip = float((px.max(axis=1) >= 0.995).mean())
        mean = float(px.mean())
        if clip > C.MAX_CLIP_FRAC:
            reasons.append(f"clipped({clip:.3f}>{C.MAX_CLIP_FRAC})")
        if mean < C.MIN_ROI_MEAN:
            reasons.append(f"underexposed(mean={mean:.3f})")
        if mean > C.MAX_ROI_MEAN:
            reasons.append(f"overexposed(mean={mean:.3f})")
    else:
        reasons.append("no_roi_pixels")

    if valid_frac < C.MIN_VALID_FRAC:
        reasons.append(f"too_few_valid_px({valid_frac:.2f}<{C.MIN_VALID_FRAC})")

    return (len(reasons) == 0), reasons
