"""Tunable constants. Everything that encodes a judgement call lives here."""
from dataclasses import dataclass, field

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "data/models/face_landmarker.task"
DB_PATH = "data/skin.db"

# MediaPipe FaceMesh landmark indices (478-point model, iris included).
IRIS_L, IRIS_R = 468, 473

# Canonical FaceMesh contour rings (mediapipe >=1.0 removed mp.solutions, so these
# are inlined). Used as EXCLUSION zones -- convex hulls, not fat circles, so they
# cut out the eye without eating half the cheek.
EXCLUDE_RINGS: dict[str, list[int]] = {
    "eye_r": [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
    "eye_l": [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
    "brow_r": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "brow_l": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
             0, 37, 39, 40, 185],
    "nostrils": [98, 327, 2, 94, 1, 4],
}
EXCLUDE_DILATE_IOD = 0.07   # pad each exclusion hull, in units of inter-ocular distance

# Each ROI is a disc: centre = WEIGHTED mean of anchor landmarks, radius = k * IOD.
# Weights were fitted against measured landmark geometry so cheeks land mid-cheek
# rather than under the eye. Discs (not hand-drawn polygons) so the geometry scales
# with the face and stays easy to eyeball on the overlay.
#
# VERIFY THESE ON YOUR OWN OVERLAY BEFORE TRUSTING A SINGLE NUMBER. Faces differ;
# these are the two knobs you are most likely to need to tune.
ROIS: dict[str, tuple[list[tuple[int, float]], float]] = {
    # forehead: between hairline (10) and glabella (9), biased low to dodge hair
    "forehead":    ([(10, 0.35), (9, 0.65)], 0.22),
    # cheek: eye outer corner -> mouth corner, pushed outward toward the jaw edge
    "cheek_right": ([(33, 0.40), (61, 0.35), (234, 0.25)], 0.22),
    "cheek_left":  ([(263, 0.40), (291, 0.35), (454, 0.25)], 0.22),
    # chin: below the lower lip (17), above the chin edge (152)
    "chin":        ([(17, 0.45), (152, 0.55)], 0.20),
}

# --- quality gate ---------------------------------------------------------
MAX_YAW_DEG = 5.0
MAX_PITCH_DEG = 5.0
MAX_ROLL_DEG = 5.0
MIN_SHARPNESS = 120.0      # Tenengrad on the ROI union; calibrate on your own rig
MAX_CLIP_FRAC = 0.005      # fraction of ROI pixels at/near sensor clipping
MIN_ROI_MEAN = 0.04        # linear reflectance; below this you are underexposed
MAX_ROI_MEAN = 0.75
MIN_VALID_FRAC = 0.55      # usable pixels remaining in a ROI after masking

# --- pixel exclusion within a ROI ----------------------------------------
SPECULAR_PCT = 98.0        # drop the top N-th percentile of luminance (shine)
SHADOW_PCT = 2.0
DARK_OUTLIER_SIGMA = 3.0   # drop hair/stubble: very dark vs ROI median.
                           # Deliberately generous so acne lesions are KEPT.

# --- texture --------------------------------------------------------------
# Filter scale is set in units of inter-ocular distance, so it adapts to apparent
# face size WITHOUT resampling the image. Resampling would low-pass the texture by
# a pose-dependent amount and make "texture" correlate with how straight you sat.
TEXTURE_SIGMAS_IOD = (0.004, 0.008, 0.016)

# --- reference target -----------------------------------------------------
# Any STABLE neutral object works. Absolute accuracy is not what longitudinal
# self-comparison needs -- consistency is. White printer paper is a legitimate
# stopgap: its true reflectance is unknown, so absolute melanin/ITA values are
# offset, but the offset is CONSTANT, so trends stay valid.
#
# Caveats for paper: most printer paper contains optical brighteners that
# fluoresce under UV-rich light (some LEDs), so it is not spectrally neutral;
# and paper yellows with age. Use ONE sheet, keep it in a drawer, replace it
# never. Switch to "gray_card_18" when the card arrives -- and re-ingest the
# whole history, because absolute values are not comparable across targets.
REFERENCE_TARGETS = {
    "gray_card_18": 0.18,
    "white_balance_card_90": 0.90,
    "white_paper": 0.85,
}
REFERENCE_TARGET = "white_paper"

REFERENCE_MIN_AREA_FRAC = 0.004   # of full frame
REFERENCE_MAX_SAT = 0.18          # HSV saturation: a neutral target is unsaturated
REFERENCE_MAX_TEXTURE = 0.02      # and flat
REFERENCE_MAX_CLIP_FRAC = 0.01    # a blown-out reference carries NO information
# Back-compat aliases
CARD_MIN_AREA_FRAC = REFERENCE_MIN_AREA_FRAC
CARD_MAX_SAT = REFERENCE_MAX_SAT
CARD_MAX_TEXTURE = REFERENCE_MAX_TEXTURE


# --- gate thresholds learned from your own baseline -----------------------
# `python -m dskin calibrate <baseline_dir>` writes this file; it overrides the
# defaults above. Shipped thresholds are guesses -- yours are measurements.
THRESHOLDS_PATH = "data/thresholds.json"


def load_learned_thresholds() -> dict:
    import json, os
    if not os.path.exists(THRESHOLDS_PATH):
        return {}
    try:
        with open(THRESHOLDS_PATH) as f:
            d = json.load(f)
    except Exception:
        return {}
    g = globals()
    for k, v in d.items():
        if k in g and isinstance(v, (int, float)):
            g[k] = v
    return d


LEARNED = load_learned_thresholds()
