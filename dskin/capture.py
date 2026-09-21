"""Live gated capture: the shutter fires only when the frame is actually good.

What makes Face ID reliable is not its depth sensor -- it is that it REFUSES to
proceed until conditions are met. That is a gate, not a model. This is that gate:
the camera runs continuously, the frame is scored every tick, and a photo is
saved only after the conditions hold STEADY for several frames in a row.

A frame you never took is a frame you never have to distrust later.
"""
from __future__ import annotations
import os
import time
import datetime as dt
import numpy as np
import cv2
from . import config as C, imageio, faces, color, texture

IPD_MM = 63.0   # population mean interpupillary distance: our physical anchor


def _has_gui() -> bool:
    """Actually probe it. opencv-python exposes imshow even where no display
    exists, and the Qt plugin aborts the process rather than raising politely."""
    import sys
    if not hasattr(cv2, "imshow"):
        return False
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        cv2.namedWindow("__probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe")
        return True
    except Exception:
        return False


def px_per_mm(iod_px: float) -> float:
    return iod_px / IPD_MM


def score_frame(bgr: np.ndarray, ref_landmarks=None) -> dict:
    """Evaluate one live frame. Returns per-condition pass/fail plus diagnostics."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    face = faces.detect(rgb)
    st: dict = {"ok": False, "face": face, "checks": {}, "masks": None, "ref": None}
    if face is None:
        st["checks"]["face"] = (False, "no face")
        return st

    lin = imageio.srgb_to_linear(rgb.astype(np.float32) / 255.0)
    masks = faces.roi_masks(face, lin.shape)
    union = np.zeros(lin.shape[:2], bool)
    for m in masks.values():
        union |= m
    st["masks"] = masks

    lab = color.linear_rgb_to_lab(lin)
    sharp = texture.sharpness_tenengrad(lab[:, :, 0], union)
    ref = color.find_reference(lin, exclude_box=face["bbox"])
    st["ref"] = ref

    ch = st["checks"]
    ch["face"] = (True, "found")
    for nm, val, lim in (("yaw", face["yaw"], C.MAX_YAW_DEG),
                         ("pitch", face["pitch"], C.MAX_PITCH_DEG),
                         ("roll", face["roll"], C.MAX_ROLL_DEG)):
        good = np.isfinite(val) and abs(val) <= lim
        ch[nm] = (good, f"{val:+.1f} deg" if np.isfinite(val) else "n/a")
    ch["sharp"] = (sharp >= C.MIN_SHARPNESS, f"{sharp:.0f}")

    if union.sum():
        mean = float(lin[union].mean())
        clip = float((lin[union].max(axis=1) >= 0.995).mean())
        ch["exposure"] = (C.MIN_ROI_MEAN <= mean <= C.MAX_ROI_MEAN and clip <= C.MAX_CLIP_FRAC,
                          f"{mean:.2f}")
    else:
        ch["exposure"] = (False, "no roi")

    if ref is None:
        ch["reference"] = (False, "not found")
    elif ref["clip_frac"] > C.REFERENCE_MAX_CLIP_FRAC:
        ch["reference"] = (False, f"CLIPPED {ref['clip_frac']:.2f}")
    else:
        ch["reference"] = (True, f"{ref['n_px']}px")

    # Alignment against the enrolled reference frame: keeps day 200 framed like
    # day 1, which is the whole point of a longitudinal series.
    if ref_landmarks is not None:
        d = float(np.linalg.norm(face["landmarks"] - ref_landmarks, axis=1).mean())
        tol = 0.12 * face["iod_px"]
        ch["alignment"] = (d <= tol, f"{d:.0f}px (tol {tol:.0f})")

    st["ok"] = all(v[0] for v in ch.values())
    return st


def _draw_hud(bgr, st, streak, need, saved, ghost=None):
    vis = bgr.copy()
    h, w = vis.shape[:2]
    if ghost is not None:
        for (x, y) in ghost.astype(int):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(vis, (x, y), 1, (120, 120, 120), -1)
    if st.get("masks"):
        for m in st["masks"].values():
            e = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            vis[e.astype(bool)] = (0, 255, 255)
    if st.get("ref"):
        x, y, rw, rh = st["ref"]["bbox"]
        cv2.rectangle(vis, (x, y), (x + rw, y + rh), (255, 255, 255), 2)

    panel = max(210, int(0.26 * w))
    cv2.rectangle(vis, (0, 0), (panel, 26 * (len(st["checks"]) + 2)), (0, 0, 0), -1)
    for i, (k, (good, why)) in enumerate(st["checks"].items()):
        col = (0, 220, 0) if good else (0, 0, 255)
        cv2.putText(vis, f"{'OK ' if good else 'NO '}{k:<10s}{why}", (8, 22 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
    y0 = 22 + 24 * len(st["checks"])
    if st["face"]:
        pm = px_per_mm(st["face"]["iod_px"])
        cv2.putText(vis, f"   {pm:.1f} px/mm   saved {saved}", (8, y0 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    if st["ok"]:
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 220, 0), 6)
        cv2.putText(vis, f"HOLD {streak}/{need}", (w // 2 - 80, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 3, cv2.LINE_AA)
    return vis


def list_cameras(max_index: int = 4) -> list[dict]:
    """Probe camera indices so you can pick the right one."""
    import sys
    # Probing absent indices is noisy on every backend; the result is the report.
    try:
        prev = cv2.getLogLevel()
        cv2.setLogLevel(0)
    except Exception:
        prev = None
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" \
            else cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append({"index": i,
                              "size": f"{frame.shape[1]}x{frame.shape[0]}",
                              "black": bool(frame.max() < 8)})
        cap.release()
    if prev is not None:
        try:
            cv2.setLogLevel(prev)
        except Exception:
            pass
    return found


def _open(source, width, height):
    """Open a camera, preferring AVFoundation on macOS."""
    import sys
    idx = int(source) if str(source).isdigit() else source
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if (
        sys.platform == "darwin" and str(source).isdigit()) else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(idx, be)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def _frames(source, width, height):
    """Yield BGR frames from a camera index, a video file, or a folder of images."""
    if isinstance(source, str) and os.path.isdir(source):
        from .cli import _find_images
        for p in _find_images(source):
            img = cv2.imread(p)
            if img is not None:
                yield img
        return
    cap = _open(source, width, height)
    if cap is None:
        import sys
        msg = [f"cannot open camera/source {source!r}"]
        if sys.platform == "darwin":
            msg += ["On macOS the terminal app needs camera access:",
                    "  System Settings > Privacy & Security > Camera > enable your",
                    "  terminal (Terminal / iTerm / VS Code), then restart it.",
                    "Run `python -m dskin capture --list` to see available cameras."]
        raise RuntimeError("\n  ".join(msg))
    if str(source).isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Best effort: many built-in Mac cameras ignore these and stay on auto.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        got = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  camera opened at {int(got[0])}x{int(got[1])}")
    try:
        blackrun = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # A camera that opens but returns black frames is the classic macOS
            # permission failure -- it fails silently rather than erroring.
            if frame.max() < 8:
                blackrun += 1
                if blackrun == 15:
                    print("\n  Camera is returning BLACK frames. On macOS this almost"
                          "\n  always means the terminal lacks camera permission:"
                          "\n  System Settings > Privacy & Security > Camera, enable"
                          "\n  your terminal app, then restart it.\n")
            else:
                blackrun = 0
            yield frame
    finally:
        cap.release()


def run(source="0", outdir="data/images", n_shots=3, hold_frames=8,
        cooldown_s=2.0, width=3840, height=2160, enroll: str | None = None) -> int:
    os.makedirs(outdir, exist_ok=True)

    ghost = None
    if enroll and os.path.exists(enroll):
        img = cv2.imread(enroll)
        f = faces.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img is not None else None
        if f is not None:
            ghost = f["landmarks"]
            print(f"  enrolled reference frame: {enroll}")
        else:
            print(f"  WARNING: no face in {enroll}; running without alignment check")

    gui = _has_gui()
    if not gui:
        print("  (no GUI build of OpenCV -- running headless. `pip install opencv-python`")
        print("   instead of opencv-python-headless to get the live preview.)")

    saved, streak, last_save, warned_res = 0, 0, 0.0, False
    print(f"\n  Gate: hold steady {hold_frames} frames. Need {n_shots} shots. "
          f"{'q to quit.' if gui else 'Ctrl-C to quit.'}\n")

    for frame in _frames(source, width, height):
        st = score_frame(frame, ghost)

        if st["face"] and not warned_res:
            warned_res = True
            pm = px_per_mm(st["face"]["iod_px"])
            print(f"  scale: {pm:.1f} px/mm (IOD {st['face']['iod_px']:.0f} px)")
            if pm < 10:
                print(f"  NOTE: below ~10 px/mm. Colour metrics (melanin, erythema, ITA)")
                print(f"  are fine; TEXTURE metrics are not resolvable at this scale.")
                print(f"  Move closer, or shoot on a phone, if you want texture.")

        streak = streak + 1 if st["ok"] else 0
        if streak >= hold_frames and (time.time() - last_save) > cooldown_s:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            # PNG, never JPEG: compression artifacts are indistinguishable from texture.
            path = os.path.join(outdir, f"{ts}_{saved:02d}.png")
            cv2.imwrite(path, frame)
            saved += 1
            streak = 0
            last_save = time.time()
            print(f"  [{saved}/{n_shots}] saved {path}")
            if ghost is None and st["face"] is not None:
                ghost = st["face"]["landmarks"]   # first good shot becomes the anchor
            if saved >= n_shots:
                break

        if gui:
            cv2.imshow("d-skin capture", _draw_hud(frame, st, streak, hold_frames, saved, ghost))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        elif not st["ok"]:
            bad = [k for k, v in st["checks"].items() if not v[0]]
            print(f"  waiting: {', '.join(bad)}          ", end="\r")

    if gui:
        cv2.destroyAllWindows()
    print(f"\n  {saved} photo(s) -> {outdir}")
    if saved:
        print(f"  next:  python -m dskin check {outdir}/<file>   # verify the overlay")
    return 0 if saved else 1
