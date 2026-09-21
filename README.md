# d-skin

Longitudinal skin-image measurement. Measurement first, models later.

**This is not a diagnosis tool.** It produces consistent image-derived numbers and
their noise floor, so you can tell a real change from a change in your bathroom.

---

## WHAT YOU DO — the short version

### Step 0 — grab a sheet of white printer paper (free, today)

Any stable neutral object works as the reference. Paper's true reflectance is
unknown, so your *absolute* melanin/ITA values will be offset — but the offset is
**constant**, so trends stay valid. That is all longitudinal self-comparison needs.

Use **one** sheet. Keep it in a drawer. Never swap it.

Two caveats, which is why this is a stopgap and not the answer: most printer paper
contains optical brighteners that fluoresce under UV-rich light (some LEDs), so it
is not truly spectrally neutral; and paper yellows with age.

**When your grey card arrives**, set `REFERENCE_TARGET = "gray_card_18"` in
`dskin/config.py` and **re-ingest the whole history** — absolute values are not
comparable across reference targets.

Why a reference at all, given colour-constancy models exist: those models estimate
*illuminant chromaticity* only and are scale-invariant by construction, so they
cannot recover **exposure** — and melanin index, ITA and L\* are absolute-reflectance
quantities. Worse, illuminant estimators read the illuminant from image content, and
skin is a strong cue: if your face genuinely reddens, the model may "correct" that
away. Its error is correlated with your signal. A physical reference's is not.

### Step 1 — lock your camera (free, 2 minutes)

Your phone is fine. The sensor was never the limit — the **ISP** is.

- Manual / Pro mode. **Lock ISO, shutter, white balance and focus.** Write the
  settings down; use the same ones forever.
- **RAW/DNG if available.** Otherwise PNG. Never JPEG.
- **HDR off. Beauty/skin-smoothing off. Scene optimiser off.** Phone ISPs literally
  even out pigmentation and smooth pores — they erase what you are measuring.

If your stock camera app cannot lock white balance, use **Open Camera** (Android,
free, shoots DNG) or **Halide / ProCam** (iOS).

You do not have to take this on trust: `dskin calibrate` reads EXIF and tells you
whether your settings actually held constant across the shoot.

### Step 2 — take 30 photos, once (~20 minutes)

Windowless room, at night, one lamp. Grey card held beside your cheek, in frame,
facing the camera, in **every** shot.

| Folder | What to do | How many |
|---|---|---|
| `baseline/` | Identical conditions. Stand up and sit back down between each. | 10 |
| `lighting/` | Vary the light: move the lamp, dim it, add a second, change bulb. | 10 |
| `posedist/` | Vary yourself: tilt ±8°, turn, move 3–5 cm closer/further. | 10 |

Keep the paper in frame beside your cheek, flat to the camera, unshadowed, in
**every** shot — including the `lighting/` and `posedist/` ones.

```
myshoot/
├── baseline/
├── lighting/
└── posedist/
```

### Step 3 — run two commands

```bash
python -m dskin calibrate myshoot/baseline   # learn YOUR thresholds, check EXIF
python -m dskin experiment myshoot           # the go/no-go test
```

`calibrate` writes `data/thresholds.json`, which overrides the shipped gate
thresholds. The defaults were guessed on a stock portrait; yours are measured on
your face, your lamp, your phone. It also tells you if auto-exposure was secretly
still on, and if your reference is clipping.

### Step 4 — read three numbers

1. **Look at `out/overlays/*.jpg` first.** Are the four coloured discs on actual
   forehead / cheeks / chin? If not, adjust `ROIS` in `dskin/config.py`. Every
   number is worthless until this looks right.

2. **The verdict block** prints worst-case nuisance error as % for each arm
   (`none` / `grayworld` / `card`). Compare against a realistic 12-week effect:
   pigmentation moves single-digit %, acne lesion counts move far more.
   - **under ~2%** → good, start collecting.
   - **2–5%** → fine for acne, not for pigmentation. Tighten and re-run.
   - **over ~5%** → **do not start a 12-week collection.** You would be recording
     a random walk. The per-condition bias columns tell you which nuisance to fix.

3. **`grayworld` vs `card`.** If they are close, drop the card — you proved you
   don't need it. If `card` is much better, no AWB model will substitute for it.

### Step 5 — then, and only then, collect daily

```bash
python -m dskin ingest data/images --arm card    # measure + store
streamlit run app.py                             # look at trends
```

Expect **8–12 weeks** before any trend is readable. Capture daily; read weekly.

---

## Before you start collecting: pick your endpoints

Write down **2–3 primary metrics** now and treat everything else as exploratory.
Four regions × six metrics × three windows is ~70 chances to find a trend in pure
noise. Pre-registering is the whole difference between a measurement and a
fishing expedition.

**The strongest upgrade available, and it is free: split-face.** Apply the active
to one side only, use the other as a within-subject control. Same lighting, same
day, same physiology. This does more for your inference than the entire model stack.

## Capture protocol (once you are collecting)

Same time of day. Fixed interval after washing. Nothing applied to the face.
No shower or exercise for 60 min (transient erythema). ≥30 min after waking
(sleep creases, overnight oedema). Photograph before shaving, or restrict analysis
to forehead and upper cheek.

---

## Install

```bash
pip install -r requirements.txt
# Linux also needs: sudo apt-get install libegl1 libgles2 libgl1 libglib2.0-0t64
python tools/make_synthetic_experiment.py <any_portrait.jpg> out/synthetic
python -m dskin experiment out/synthetic    # self-test with known ground truth
```

The self-test injects known perturbations, so you can confirm the pipeline recovers
them before pointing it at your own face.

## Commands

```bash
python -m dskin check     <image>   # one photo: pose, gate result, overlay
python -m dskin calibrate <dir>     # learn gate thresholds + verify EXIF held
python -m dskin ingest    <dir>     # measure a folder into sqlite
python -m dskin experiment <dir>    # the go/no-go test over condition subfolders
streamlit run app.py                # trends + sanity checks
```

## How it works

```
photo → linearise → face landmarks → pose gate → grey-card normalise
      → ROI masks (native pixels) → colour + texture metrics → sqlite
```

Three design choices that matter:

**Metrics are computed on native, un-resampled pixels.** The face model is used only
to *locate* regions, never to produce the pixels measured. Warping resamples, and
resampling is a low-pass filter whose strength varies with the local warp Jacobian,
which varies with head pose — so measuring warped pixels makes "texture" correlate
with how straight you sat. `pose_confound` in the experiment output exists to catch
exactly this; a metric correlating with `iod_px` is broken, not interesting.

**Texture filter scale is set in units of inter-ocular distance**, so it adapts to
apparent face size without resampling the image.

**Everything is linearised first.** sRGB values are gamma-encoded; averaging them is
a silent systematic error. The melanin/erythema log-reflectance indices are
meaningless on uncalibrated sRGB.

## Layout

```
dskin/config.py      thresholds, ROI anchors  <- the knobs you will tune
dskin/imageio.py     RAW/LDR -> linear RGB
dskin/faces.py       landmarks, pose, ROI masks, overlay
dskin/color.py       grey-card normalisation, LAB, melanin/erythema/ITA
dskin/texture.py     scale-adaptive texture metrics
dskin/quality.py     the capture gate
dskin/calibrate.py   learn thresholds from your own shoot; EXIF constancy check
dskin/pipeline.py    one image -> one row
dskin/experiment.py  stability / probe / pose-confound / verdict
dskin/db.py          sqlite
app.py               dashboard
```

## Privacy

`data/images/` and `data/*.db` are gitignored. Keep it local, on an encrypted disk.
Do not send your face to a cloud API to "just try it".

## Not yet built (deliberately)

V2: acne lesion detection and counting — for acne this is a far better primary
endpoint than embeddings (lesion counts move 40–60% over 12 weeks; pigmentation
moves single digits). V3: foundation-model embeddings per ROI. Neither is worth
building until Step 4 comes back green.

Note on embeddings: Google's Derm Foundation lists "identifying the body part" and
"determining image quality" among its supported downstream tasks — which means those
nuisance factors are *encoded* in its embeddings, not discarded. It is also now
legacy (MedSigLIP is the recommended successor). Raw cosine-similarity-to-baseline
is unsigned and will ride the nuisance directions; that is why it is not in V1.
