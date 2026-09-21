"""Learn gate thresholds from your own baseline shoot.

The shipped thresholds are guesses made on a stock portrait. Yours are
measurements made on your face, your lamp, your phone. Run this once on a
baseline folder and the gate starts rejecting what is actually bad FOR YOU,
rather than what looked bad to me.
"""
from __future__ import annotations
import json
import os
import numpy as np
from . import config as C, pipeline, imageio


def _pct(vals: list[float], q: float, default: float) -> float:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.percentile(v, q)) if len(v) >= 4 else default


def calibrate(rows: list[dict]) -> tuple[dict, list[str]]:
    """Derive thresholds + report on whether your camera settings held."""
    notes: list[str] = []
    ok = [r for r in rows if r.get("iod_px")]
    if len(ok) < 5:
        return {}, ["Need at least 5 analysable photos; got %d." % len(ok)]

    sharp = [r.get("sharpness") for r in ok]
    # 10th percentile: the blurriest frame you would still accept.
    thr = {"MIN_SHARPNESS": round(_pct(sharp, 10, C.MIN_SHARPNESS) * 0.9, 1)}

    for name, key, lo, hi in (("yaw", "yaw", 5.0, 12.0), ("pitch", "pitch", 5.0, 12.0),
                              ("roll", "roll", 5.0, 12.0)):
        vals = [abs(r[key]) for r in ok if r.get(key) is not None and np.isfinite(r[key])]
        if not vals:
            continue
        # 90th percentile of your own steadiness, floored at 3 deg and capped:
        # the gate should be tight enough to matter, loose enough to pass.
        thr[f"MAX_{name.upper()}_DEG"] = round(min(hi, max(3.0, _pct(vals, 90, lo) * 1.2)), 1)

    iod = [r["iod_px"] for r in ok]
    cv = float(np.std(iod) / np.mean(iod)) if np.mean(iod) else 0.0
    notes.append(f"Apparent face size varies {cv*100:.1f}% across the shoot "
                 f"(iod {np.min(iod):.0f}-{np.max(iod):.0f} px).")
    if cv > 0.05:
        notes.append("  >5% distance variation. Illumination falls off inverse-square, so "
                     "this alone is a ~10% brightness swing. Mark a floor spot and a "
                     "phone position, or brace your elbows on a fixed surface.")

    found = sum(1 for r in ok if r.get("card_found"))
    notes.append(f"Reference target found in {found}/{len(ok)} photos "
                 f"(configured: {C.REFERENCE_TARGET}).")
    if found < len(ok):
        notes.append("  Keep the target fully in frame, flat to the camera, unshadowed.")
    clipped = [r.get("card_clip_frac", 0) or 0 for r in ok]
    if clipped and max(clipped) > C.REFERENCE_MAX_CLIP_FRAC:
        notes.append("  Reference is CLIPPING in at least one frame. Expose down one stop, "
                     "or switch to a darker target (an 18% grey card).")

    const = imageio.settings_constancy(ok)
    if const:
        drift = [k for k, v in const.items() if not v["constant"]]
        if drift:
            notes.append("CAMERA SETTINGS DRIFTED across this shoot: " + ", ".join(drift))
            for k in drift:
                notes.append(f"  {k}: {const[k]['values']}")
            notes.append("  Auto-exposure or auto-WB is still on. Lock them; this is the "
                         "single cheapest variance reduction available to you.")
        else:
            notes.append("Camera settings held constant across the shoot. Good.")
    else:
        notes.append("No EXIF found (stripped, or RAW). Cannot verify your settings were "
                     "locked -- confirm manually in your camera app.")
    return thr, notes


def run(directory: str, arm: str = "card", write: bool = True) -> int:
    import glob
    from .cli import _find_images
    imgs = _find_images(directory)
    if not imgs:
        print(f"no images in {directory}")
        return 1
    rows = []
    for p in imgs:
        try:
            rows.append(pipeline.analyze(p, arm=arm))
        except Exception as e:
            print(f"  skipped {os.path.basename(p)}: {e}")
    thr, notes = calibrate(rows)

    print(f"\nCalibrated on {len(rows)} photos from {directory}\n")
    for n in notes:
        print(" ", n)
    if not thr:
        return 1
    print("\nSuggested thresholds:")
    for k, v in thr.items():
        print(f"  {k:<18s} {v!r:>8}   (was {getattr(C, k)!r})")

    if write:
        os.makedirs(os.path.dirname(C.THRESHOLDS_PATH) or ".", exist_ok=True)
        with open(C.THRESHOLDS_PATH, "w") as f:
            json.dump(thr, f, indent=2)
        print(f"\nWritten to {C.THRESHOLDS_PATH} (delete it to revert to defaults).")
    return 0
