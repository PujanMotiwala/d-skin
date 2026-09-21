# d-skin — project handoff / brainstorm log

Paste this file into a new Claude Code session (ideally one running **locally on
your Mac**, since that's what this project needs from here on — a camera, a
filesystem it can write photos to, and the ability to open a live preview
window). Tell the new session: *"Read HANDOFF.md, then continue from where this
left off."*

Repo: `https://github.com/PujanMotiwala/d-skin`
Branch: `claude/laughing-lovelace-88uovl` (already pushed, ~2000 lines, working)

---

## 1. What this project is

Daily/regular skin photos → consistent, comparable, longitudinal **measurements**
(pigmentation, erythema, texture) — not a diagnosis tool, not "AI dermatologist."
Track your own skin against your own baseline over weeks/months, e.g. to see
whether a skincare routine is doing anything.

Original inspiration: a ChatGPT brainstorm proposing `google/derm-foundation`
(Google's dermatology embedding model) as the core. That idea was interrogated
hard across this session and substantially revised — see §3.

**Core philosophy: measurement first, models later.** Do not point a foundation
model at your face until you've proven your own capture pipeline can measure
anything at all. Most of the difficulty is metrology (controlling lighting,
pose, distance, exposure), not modeling.

---

## 2. Where things stand right now

- **V1 is fully built and pushed** to `claude/laughing-lovelace-88uovl`.
- It has **never been run on a real camera or a real face** — only validated on
  a synthetic dataset built from a stock portrait with injected, known
  perturbations (lighting, pose, distance).
- The blocker: this Claude Code session runs in a **cloud container**, not on
  your Mac. It has no camera, and no access to your Mac's filesystem. Everything
  built so far is code + a synthetic self-test; the real-world run has to happen
  in a session that's actually on your machine (local Claude Code, or you
  running the commands yourself in Terminal).
- **Next concrete step**, once you're in a local session: clone the repo,
  `pip install -r requirements.txt && pip install opencv-python`, grant camera
  permission (System Settings → Privacy & Security → Camera → your terminal
  app → restart terminal), then `python -m dskin capture --list` to confirm the
  Mac's camera is visible, then run the actual capture protocol in §6.

---

## 3. Key decisions and the reasoning behind them

These are the things worth NOT re-deriving from scratch.

### 3a. Don't feed a whole face to Derm Foundation
It's trained on dermatology images framed on a condition/region, not portraits.
Full-face at 448×448 gives ~3px/mm — discards texture, out of distribution.
**Decision:** if/when a foundation model is used, crop fixed anatomical ROIs
(forehead, cheeks, chin) at constant mm-per-pixel, not the whole face.

### 3b. Raw cosine-similarity-to-baseline is a weak/dangerous metric
It's unsigned (can't tell "better" from "different"), and — this is the
important part — **Derm Foundation's own documented downstream tasks include
predicting body part and image quality from the embedding.** That means those
nuisance factors are *linearly encoded* in the embedding space, not discarded.
A linear probe can read them off; cosine distance will ride those directions
preferentially. This was checked against the actual model card / HF page /
papers, not assumed. Also: Derm Foundation is now **legacy**; Google recommends
**MedSigLIP** for new work. Verify this is still current before building on it.
**Decision:** don't build V1 around embeddings/cosine similarity. If used at
all, it's V3, after a supervised head is trained on real labels — and even then,
compare against MedSigLIP.

### 3c. Face-landmark models (MediaPipe) solve geometry, not photometry
A 3D face mesh gives real, stable, pose-invariant ROI localization — that part
of the "replace the rig with software" idea is correct and implemented.
**But it cannot fix white balance, exposure, or ISP tone-mapping.** Geometry
and photometry are different problems with different fixes.

### 3d. A physical reference target beats a color-constancy model — and this was proven, not assumed
Built a synthetic experiment with **known ground truth** (injected exposure +
white-balance perturbations) and ran three correction arms through the actual
pipeline:

| arm | worst-case nuisance error |
|---|---|
| card (physical reference) | **3.5%** |
| grayworld (stand-in for learned color constancy) | 20.8% |
| none | 21.8% |

Why gray-world/AWB-style correction fails here: it estimates illuminant
*chromaticity* only and is scale-invariant by construction — it cannot recover
**exposure**, but melanin index / ITA° / L* are absolute-reflectance
quantities. Also, an illuminant estimator reads the illuminant partly from
image content, and skin is a strong cue — if your face genuinely reddens, the
model may "correct" that away. **Its error is correlated with the very signal
you're trying to measure.** A physical card's error is not.
**Decision:** use a physical reference target. Started with **white printer
paper** (free, today) as a stopgap — true reflectance unknown so absolute
values are offset, but the offset is *constant*, so trends stay valid. Swap to
an **18% gray card** (~₹500–1500, cheaper than a full ColorChecker Passport)
when it arrives; requires re-ingesting history since absolute values aren't
comparable across targets (`REFERENCE_TARGET` in `dskin/config.py`).

### 3e. The resampling/pose confound (the subtle one — catch this if nobody mentions it again)
Warping a photo to a canonical/UV frame for measurement **resamples pixels**,
and resampling is a spatially-varying low-pass filter whose strength depends on
the local warp Jacobian, which depends on head pose. So a more-rotated face
gets more smoothing in the warped output — meaning "texture improved" can
secretly mean "sat straighter this month."
**Decision:** the face model is used ONLY to *locate* ROIs. All metrics (color
AND texture) are computed on **native, un-resampled pixels** at their
back-projected location, never on warped/UV pixels. Texture filter scale is
expressed in units of inter-ocular distance (adapts to apparent face size
without resampling). `dskin/experiment.py::pose_confound()` checks metrics for
correlation with `iod_px`/yaw/pitch as a standing sanity check — a metric that
correlates with pose is broken, not interesting.

### 3f. The capture gate is a gate, not a model (Face ID analogy)
What makes Face ID reliable isn't the depth sensor, it's that it refuses to
proceed until conditions are met. That's cheap: pose thresholds, sharpness,
exposure, reference-target-found, all checked live, shutter fires only after N
consecutive good frames. Implemented in `dskin/capture.py`.

### 3g. Camera choice: any phone/laptop works IF manual settings are locked
Resolution was never the bottleneck; the ISP is. HDR/beauty/scene-optimizer
literally smooth pigmentation and pores — must be off. RAW/DNG preferred, PNG
otherwise, **never JPEG** (compression artifacts ≈ texture noise).

**Physical scale (px/mm) determines which metrics are even possible:**
| setup | px/mm | color metrics | texture metrics |
|---|---|---|---|
| MacBook webcam @ 50cm | 2.5 | yes | no |
| MacBook webcam @ 30cm | 4.2 | yes | no |
| Phone, face fills frame @ 30cm | 29 | yes | yes |

Thresholds: ~2 px/mm for color, ~10 for mid-band texture, ~25 for pore-scale.
**MacBook webcam decision: fine for pigmentation/erythema/ITA, useless for
texture.** Also: macOS generally won't let you lock exposure/WB on the
built-in camera — the reference target matters *more* here, not less, though it
can't undo the ISP's nonlinear tone curve (a gain correction is linear).
`dskin capture` and `dskin calibrate` print actual px/mm measured from your
setup and tell you which metrics are resolvable — don't have to trust the table.

### 3h. Statistical traps to guard against once data collection starts
- **Multiple comparisons**: 4 regions × 6 metrics × 3 windows ≈ 70 chances to
  find noise. Pre-register 2-3 primary endpoints *before* collecting.
- **Regression to the mean**: you start tracking when skin looks bad; it
  improves regardless of any product.
- **Temporal autocorrelation**: daily measurements aren't independent; read
  trends from weekly means, not daily points (the dashboard defaults to this).
- **No control arm** — the single highest-value free fix: **split-face**
  design. Apply the active to one side only, use the other as a same-day,
  same-light, same-physiology control.
- Physiological confounds that dwarf a 12-week treatment effect: hot
  shower/exercise/alcohol (transient erythema, wait 60min), sleep
  creases/edema (wait 30-45min after waking), shaving (strong erythema —
  either always pre-shave or restrict analysis to forehead/upper cheek).
- Expect **8-12 weeks minimum** before any trend is readable above noise.
  Acne lesion counts move 40-60% over 12 weeks — much bigger/faster signal
  than pigmentation (single-digit %) if acne is the actual target.

---

## 4. What's built (V1) — file by file

```
dskin/config.py      thresholds, ROI landmark anchors (disc anchors + radius in
                      units of inter-ocular distance), reference-target registry
dskin/imageio.py     RAW(rawpy)/LDR -> linear RGB (sRGB gamma removed FIRST,
                      before any arithmetic — averaging gamma-encoded values is
                      a silent systematic error); EXIF capture-settings reader
dskin/faces.py       MediaPipe FaceLandmarker wrapper: pose (yaw/pitch/roll from
                      facial transformation matrix), ROI masks as landmark-
                      anchored discs with eye/brow/lip/nostril exclusion hulls,
                      debug overlay renderer
dskin/color.py       reference-target detection (flat/unsaturated/large region,
                      works for gray card OR white paper), white-balance +
                      EXPOSURE normalization against it, linear RGB -> CIELAB,
                      ITA°, melanin/erythema log-reflectance indices
dskin/texture.py     scale-adaptive band-pass texture metrics (DoG at multiple
                      scales in units of IOD), local contrast, gradient
                      magnitude, Tenengrad sharpness — all on native pixels
dskin/quality.py     post-hoc capture gate (pose/sharpness/exposure/clip/
                      reference-found/valid-pixel-fraction thresholds)
dskin/capture.py     LIVE gated capture: opens camera, scores every frame,
                      fires shutter after N consecutive good frames, live HUD
                      overlay, --enroll for cross-day alignment check, macOS
                      camera-permission diagnostics (detects black-frame
                      failure mode), --list to probe available cameras
dskin/calibrate.py   learns YOUR OWN gate thresholds from a baseline shoot
                      (percentile-based), checks EXIF for settings drift
                      (auto-exposure secretly still on), flags distance
                      variation (inverse-square -> brightness swings), reports
                      measured px/mm and which metric classes it supports
dskin/pipeline.py    one image -> one measurement row (the full chain)
dskin/experiment.py  the go/no-go analysis: stability() [noise floor vs
                      nuisance-induced bias/variance per arm], probe() and
                      probe_pairwise() [logistic-regression batch-effect probe:
                      can a linear model read the nuisance condition off the
                      metrics?], pose_confound() [correlation with iod_px/yaw/
                      pitch], verdict() [plain-English go/no-go]
dskin/db.py          sqlite, dynamic schema (new metric columns auto-added)
dskin/cli.py         `python -m dskin {check,capture,calibrate,ingest,experiment}`
app.py               Streamlit dashboard: per-ROI trend lines (weekly-mean
                      default), noise-floor sanity table, pose-correlation
                      sanity table, rejected-photos log
tools/make_synthetic_experiment.py
                      builds a synthetic 3-condition dataset from ANY portrait
                      with known injected perturbations — self-test with
                      ground truth, run this FIRST after install
requirements.txt, .gitignore (data/images, data/*.db, data/models — faces never
                      go in git), README.md (full protocol + step-by-step)
```

Dataviz note: the Streamlit dashboard's color choices were validated with the
`dataviz` skill's palette validator (categorical hues assigned by ROI, in
fixed order, contrast-checked) — if a new session rebuilds any charting, reuse
that skill rather than picking colors by eye.

---

## 5. Environment gotchas already fixed (don't re-debug these)

- Linux needs `libegl1 libgles2 libgl1 libglib2.0-0t64` installed for
  MediaPipe's GPU delegate to load at all (`OSError: libEGL.so.1`).
- `mediapipe>=1.0` **removed `mp.solutions`** — landmark connection sets
  (eyes/brows/lips) are hardcoded as index lists in `config.py` now, not
  pulled from the old solutions API.
- Default ROI landmark anchors needed hand-tuning against actual measured
  landmark coordinates (not guessed) — first attempt put cheek ROIs on the
  eyes. Fixed with weighted multi-anchor centroids; **verify on your own face's
  overlay before trusting numbers** — anchors were only validated on one stock
  portrait.
- `cv2.setLogLevel` does **not** suppress OpenCV videoio backend errors — they
  write straight to file descriptor 2. Camera probing (`--list`) mutes fd 2
  directly around the probe.
- OpenCV's `imshow`/`namedWindow` existing doesn't mean a display exists;
  `_has_gui()` actually opens/closes a probe window rather than checking
  `hasattr`.
- macOS-specific: camera permission failures are **silent** — camera opens
  fine, frames come back solid black. `capture.py` detects a run of black
  frames and prints the System Settings remedy instead of silently producing
  garbage.

---

## 6. The actual protocol to run once on a real camera (unexecuted so far)

```bash
# once, sanity-check the install with synthetic ground truth
python tools/make_synthetic_experiment.py <any_portrait.jpg> out/synthetic
python -m dskin experiment out/synthetic
# expect: card ~3-4% worst-case nuisance, grayworld/none ~20%+

# confirm the real camera is visible (macOS: grant camera permission first —
# System Settings > Privacy & Security > Camera > your terminal > restart it)
python -m dskin capture --list

# real baseline shoot: white paper in frame, held beside cheek, flat to camera
python -m dskin capture --shots 10 --out myshoot/baseline

# CHECK THE OVERLAY BEFORE TRUSTING ANYTHING
python -m dskin check myshoot/baseline/<file>.png
# -> look at out/overlays/*.jpg: do the 4 discs land on forehead/cheeks/chin,
#    clear of eyes/brows/lips/hair? if not, tune ROIS in dskin/config.py

python -m dskin calibrate myshoot/baseline
# -> learns your real gate thresholds, reports real px/mm, flags EXIF drift
#    (auto-exposure secretly on?), flags distance variation

# then the two perturbation folders (loosen --hold and pose thresholds
# temporarily — the gate will otherwise correctly refuse the "bad" frames
# you're trying to deliberately collect)
python -m dskin capture --shots 10 --hold 2 --out myshoot/lighting     # vary the lamp
python -m dskin capture --shots 10 --hold 2 --out myshoot/posedist     # vary tilt/distance

python -m dskin experiment myshoot
# -> read the verdict: worst-case nuisance %, compare to expected effect size
#    (pigmentation ~single-digit % over 12wk; acne lesion counts 40-60%)
#    <2%: good, start daily collection
#    2-5%: OK for acne, marginal for pigmentation
#    >5%: DO NOT start 12-week collection yet, fix the dominant nuisance first
```

Only after a passing verdict: start daily `dskin capture` + `dskin ingest`,
watch trends in `streamlit run app.py`, expect 8-12 weeks before anything is
statistically readable.

---

## 7. Not yet built (deliberately deferred, in order)

- **V2**: acne lesion detection/counting — better primary endpoint than
  pigmentation for acne specifically (bigger, faster-moving effect size).
- **V3**: foundation-model embeddings (MedSigLIP, not Derm Foundation — check
  currency) per ROI, only after V1's measurement is proven trustworthy and
  ideally after enough labeled days exist to train a supervised head rather
  than trusting raw embedding distance.

Neither is worth starting until the Step-6 experiment above comes back green
on a real face with a real camera.
