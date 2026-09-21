"""Face landmarks, head pose, and ROI masks in ORIGINAL image coordinates.

The warp is used only to LOCATE regions, never to produce the pixels we measure.
Resampling is a spatially varying low-pass filter whose strength depends on the
local warp Jacobian, which depends on head pose -- so measuring warped pixels
makes texture correlate with how straight you sat.
"""
from __future__ import annotations
import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from . import config as C

_LANDMARKER = None


def ensure_model(path: str = C.MODEL_PATH) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(C.MODEL_URL, path)
    return path


def get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(opts)
    return _LANDMARKER


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from the 4x4 facial transformation matrix."""
    r = m[:3, :3]
    sy = float(np.hypot(r[0, 0], r[1, 0]))
    if sy > 1e-6:
        pitch = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = np.arctan2(r[1, 0], r[0, 0])
    else:
        pitch, yaw, roll = np.arctan2(-r[1, 2], r[1, 1]), np.arctan2(-r[2, 0], sy), 0.0
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def detect(rgb_u8: np.ndarray) -> dict | None:
    """Run the landmarker. Input must be 8-bit sRGB (display-encoded)."""
    res = get_landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_u8))
    )
    if not res.face_landmarks:
        return None
    h, w = rgb_u8.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]], np.float32)

    if res.facial_transformation_matrixes:
        yaw, pitch, roll = _euler_from_matrix(np.array(res.facial_transformation_matrixes[0]))
    else:
        yaw = pitch = roll = float("nan")

    iod = float(np.linalg.norm(pts[C.IRIS_L] - pts[C.IRIS_R])) if len(pts) > C.IRIS_R else \
          float(np.linalg.norm(pts[33] - pts[263]))
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return {
        "landmarks": pts, "yaw": yaw, "pitch": pitch, "roll": roll,
        "iod_px": iod,
        "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
    }


def roi_masks(face: dict, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Boolean mask per ROI: discs anchored to landmarks, scaled by IOD.

    Eyes, brows, lips and nostrils are cut out as dilated convex hulls of their
    contour rings -- tight to the actual feature, so the cheek ROI survives.
    """
    h, w = shape[:2]
    pts, iod = face["landmarks"], face["iod_px"]
    pad = max(1, int(C.EXCLUDE_DILATE_IOD * iod))

    hull = cv2.convexHull(pts.astype(np.int32))
    face_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 1)
    face_mask = cv2.erode(face_mask, np.ones((9, 9), np.uint8))

    excl = np.zeros((h, w), np.uint8)
    for ring in C.EXCLUDE_RINGS.values():
        idx = [i for i in ring if i < len(pts)]
        if len(idx) < 3:
            continue
        cv2.fillConvexPoly(excl, cv2.convexHull(pts[idx].astype(np.int32)), 1)
    if pad > 0:
        excl = cv2.dilate(excl, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))

    out: dict[str, np.ndarray] = {}
    for name, (anchors, k) in C.ROIS.items():
        anchors = [(i, wt) for i, wt in anchors if i < len(pts)]
        if not anchors:
            continue
        tot = sum(wt for _, wt in anchors)
        c = sum(pts[i] * wt for i, wt in anchors) / tot
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, tuple(np.round(c).astype(int)), max(3, int(k * iod)), 1, -1)
        out[name] = ((m & face_mask) & (1 - excl)).astype(bool)
    return out


def draw_overlay(rgb_u8: np.ndarray, face: dict, masks: dict, card=None) -> np.ndarray:
    """Contact-sheet overlay. Eyeball this on day one before trusting any number."""
    vis = rgb_u8.copy()
    colors = {"forehead": (255, 90, 90), "cheek_left": (90, 200, 255),
              "cheek_right": (120, 255, 140), "chin": (255, 210, 80)}
    for name, m in masks.items():
        col = colors.get(name, (255, 255, 255))
        edges = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
        vis[edges.astype(bool)] = col
        vis[m] = (0.82 * vis[m] + 0.18 * np.array(col)).astype(np.uint8)
    if card:
        x, y, w_, h_ = card["bbox"]
        cv2.rectangle(vis, (x, y), (x + w_, y + h_), (255, 255, 255), 3)
        cv2.putText(vis, "CARD", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"yaw {face['yaw']:+.1f} pitch {face['pitch']:+.1f} "
                     f"roll {face['roll']:+.1f} iod {face['iod_px']:.0f}px",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return vis
