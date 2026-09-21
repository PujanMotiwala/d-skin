"""One image in, one row of measurements out."""
from __future__ import annotations
import os
import numpy as np
import cv2
from . import config as C
from . import imageio, color, faces, texture, quality

NORM_ARMS = ("none", "grayworld", "card")


def _exclude_pixels(L: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop shine, cast shadow and hair. Acne lesions are deliberately KEPT --
    they are signal, not outliers."""
    if mask.sum() < 20:
        return mask, {"excl_specular": 0.0, "excl_shadow": 0.0, "excl_dark": 0.0}
    v = L[mask]
    hi = np.percentile(v, C.SPECULAR_PCT)
    lo = np.percentile(v, C.SHADOW_PCT)
    med, mad = np.median(v), np.median(np.abs(v - np.median(v))) * 1.4826
    dark_cut = med - C.DARK_OUTLIER_SIGMA * max(mad, 1e-3)

    spec = mask & (L >= hi)
    shad = mask & (L <= lo)
    dark = mask & (L < dark_cut)
    keep = mask & ~(spec | shad | dark)
    n = float(mask.sum())
    return keep, {"excl_specular": float(spec.sum() / n),
                  "excl_shadow": float(shad.sum() / n),
                  "excl_dark": float(dark.sum() / n)}


def analyze(path: str, arm: str = "card", card_reflectance: float | None = None,
            overlay_dir: str | None = None) -> dict:
    """Full measurement for one photo under one normalisation arm."""
    if arm not in NORM_ARMS:
        raise ValueError(f"arm must be one of {NORM_ARMS}")

    lin, meta = imageio.load_linear_rgb(path)
    rgb_u8 = imageio.to_display_u8(lin)
    face = faces.detect(rgb_u8)

    row: dict = {
        "path": path, "image_id": os.path.basename(path), "arm": arm,
        "width": meta["width"], "height": meta["height"],
        "source": meta["source"], "bit_depth": meta["bit_depth"],
        "reference_target": C.REFERENCE_TARGET,
        **imageio.read_capture_settings(path),
    }
    if face is None:
        row.update(passed=0, reject_reasons="no_face_detected")
        return row

    row.update(yaw=face["yaw"], pitch=face["pitch"], roll=face["roll"],
               iod_px=face["iod_px"])

    masks = faces.roi_masks(face, lin.shape)
    union = np.zeros(lin.shape[:2], bool)
    for m in masks.values():
        union |= m

    # --- normalisation arm ------------------------------------------------
    card = color.find_reference(lin, exclude_box=face["bbox"])
    row["card_found"] = int(card is not None)
    if card:
        row["card_rgb"] = ",".join(f"{v:.5f}" for v in card["rgb"])
        row["card_clip_frac"] = card["clip_frac"]

    if arm == "card":
        if card is None:
            row.update(passed=0, reject_reasons="reference_target_not_found")
            return row
        # A blown-out reference carries no information: the gain it implies is
        # a floor, not a measurement, and silently under-corrects every frame.
        if card["clip_frac"] > C.REFERENCE_MAX_CLIP_FRAC:
            row.update(passed=0, reject_reasons=(
                f"reference_clipped({card['clip_frac']:.3f}>"
                f"{C.REFERENCE_MAX_CLIP_FRAC}); expose down or use a darker target"))
            return row
        work, gain = color.normalize_with_reference(lin, card["rgb"], card_reflectance)
    elif arm == "grayworld":
        work, gain = color.normalize_gray_world(lin, union)
    else:
        work, gain = lin, [1.0, 1.0, 1.0]
    row["gain_r"], row["gain_g"], row["gain_b"] = gain

    # --- metrics on native, un-resampled pixels ---------------------------
    lab = color.linear_rgb_to_lab(work)
    L = lab[:, :, 0]
    sharp = texture.sharpness_tenengrad(L, union)
    row["sharpness"] = sharp

    valid_total, kept_total = 0, 0
    for name, m in masks.items():
        keep, excl = _exclude_pixels(L, m)
        valid_total += int(m.sum())
        kept_total += int(keep.sum())
        if keep.sum() < 50:
            continue
        px = work[keep]
        lb = lab[keep]
        mi, ei = color.melanin_erythema(px)
        ita = color.ita_degrees(lb[:, 0], lb[:, 2])
        p = f"{name}_"
        row.update({
            p + "n_px": int(keep.sum()),
            p + "L_mean": float(lb[:, 0].mean()), p + "L_std": float(lb[:, 0].std()),
            p + "a_mean": float(lb[:, 1].mean()), p + "b_mean": float(lb[:, 2].mean()),
            p + "ita": float(np.median(ita)),
            p + "melanin": float(mi.mean()), p + "melanin_std": float(mi.std()),
            p + "erythema": float(ei.mean()),
            **{p + k: v for k, v in excl.items()},
            **{p + k: v for k, v in texture.texture_metrics(L, keep, face["iod_px"]).items()},
        })

    valid_frac = kept_total / max(valid_total, 1)
    row["valid_frac"] = valid_frac
    ok, reasons = quality.gate(face, work, union, sharp, valid_frac)
    row["passed"] = int(ok)
    row["reject_reasons"] = ";".join(reasons)

    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        vis = faces.draw_overlay(rgb_u8, face, masks, card)
        tag = "PASS" if ok else "FAIL"
        out = os.path.join(overlay_dir, f"{tag}_{os.path.splitext(row['image_id'])[0]}.jpg")
        cv2.imwrite(out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        row["overlay"] = out
    return row
