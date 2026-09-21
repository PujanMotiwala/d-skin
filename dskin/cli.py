"""d-skin command line.

  python -m dskin ingest  <dir>           daily use: measure and store
  python -m dskin experiment <dir>        the go/no-go test, all three arms
  python -m dskin check <image>           one photo, print gate result + overlay
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import datetime as dt
import pandas as pd
from . import db, pipeline, experiment, calibrate as _calibrate, capture as _capture

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.dng", "*.DNG",
           "*.arw", "*.ARW", "*.cr2", "*.CR2", "*.nef", "*.NEF")


def _find_images(d: str) -> list[str]:
    out: list[str] = []
    for e in IMG_EXT:
        out += glob.glob(os.path.join(d, e))
    return sorted(set(out))


def _captured_at(path: str) -> str:
    return dt.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")


def cmd_check(a) -> int:
    row = pipeline.analyze(a.image, arm=a.arm, overlay_dir=a.overlay_dir)
    print(f"\n{row['image_id']}  arm={row['arm']}")
    if "yaw" in row:
        print(f"  pose      yaw {row['yaw']:+.1f}  pitch {row['pitch']:+.1f}  roll {row['roll']:+.1f}")
        print(f"  scale     iod {row['iod_px']:.0f} px     sharpness {row.get('sharpness', 0):.0f}")
        print(f"  card      {'FOUND' if row.get('card_found') else 'not found'}")
    print(f"  GATE      {'PASS' if row['passed'] else 'FAIL'}")
    if row.get("reject_reasons"):
        for r in row["reject_reasons"].split(";"):
            print(f"              - {r}")
    if row.get("overlay"):
        print(f"  overlay   {row['overlay']}   <- LOOK AT THIS before trusting numbers")
    return 0 if row["passed"] else 1


def cmd_ingest(a) -> int:
    imgs = _find_images(a.directory)
    if not imgs:
        print(f"no images in {a.directory}", file=sys.stderr)
        return 1
    con = db.connect(a.db)
    npass = 0
    for p in imgs:
        row = pipeline.analyze(p, arm=a.arm, overlay_dir=a.overlay_dir)
        row["captured_at"] = _captured_at(p)
        row["condition"] = a.condition
        db.upsert(con, row)
        npass += row["passed"]
        status = "PASS" if row["passed"] else f"FAIL  {row.get('reject_reasons','')}"
        print(f"  {os.path.basename(p):<40s} {status}")
    con.close()
    print(f"\n{npass}/{len(imgs)} passed the gate -> {a.db}")
    return 0


def cmd_experiment(a) -> int:
    """<dir> holds one SUBFOLDER PER CONDITION, e.g. baseline/ lighting/ posedist/."""
    groups = [d for d in sorted(os.listdir(a.directory))
              if os.path.isdir(os.path.join(a.directory, d))]
    if not groups:
        print(f"{a.directory} needs one subfolder per condition "
              f"(baseline/, lighting/, posedist/)", file=sys.stderr)
        return 1

    rows = []
    for cond in groups:
        imgs = _find_images(os.path.join(a.directory, cond))
        print(f"\n[{cond}] {len(imgs)} images")
        for p in imgs:
            for arm in pipeline.NORM_ARMS:
                r = pipeline.analyze(p, arm=arm,
                                     overlay_dir=a.overlay_dir if arm == "card" else None)
                r["condition"] = cond
                r["captured_at"] = _captured_at(p)
                rows.append(r)
            last = rows[-1]
            print(f"  {os.path.basename(p):<36s} "
                  f"{'PASS' if last['passed'] else 'FAIL: ' + last.get('reject_reasons','')}")

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "raw_measurements.csv"), index=False)

    used = df[df.passed == 1] if (df.passed == 1).sum() >= 9 else df
    if len(used) < len(df):
        print(f"\nUsing {len(used)} gate-passing rows.")
    else:
        print("\nWARNING: too few rows passed the gate; analysing ALL rows, including")
        print("bad ones. Fix capture and re-run -- these numbers are optimistic.")

    stab = experiment.stability(used)
    prb = experiment.probe(used)
    pair = experiment.probe_pairwise(used)
    pose = experiment.pose_confound(used)
    for name, d in (("stability", stab), ("probe", prb), ("probe_pairwise", pair), ("pose_confound", pose)):
        d.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)

    if not stab.empty:
        print("\n--- Stability: worst-case nuisance error, %-of-mean (lower better) ---")
        piv = stab.pivot_table(index="metric", columns="arm", values="worst_case_pct")
        print(piv.round(2).to_string())
    if not pose.empty:
        print("\n--- Strongest pose/scale correlations (|r|>0.5 = suspect metric) ---")
        print(pose.head(8).round(3).to_string(index=False))

    txt = experiment.verdict(stab, prb, pair)
    print(txt)
    with open(os.path.join(a.out, "verdict.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nFull results -> {a.out}/")
    return 0


def cmd_capture(a) -> int:
    if a.list:
        cams = _capture.list_cameras()
        if not cams:
            print("No cameras found. On macOS, grant your terminal camera access in")
            print("System Settings > Privacy & Security > Camera, then restart it.")
            return 1
        print("Available cameras:")
        for c_ in cams:
            note = "  (BLACK FRAMES - check camera permission)" if c_["black"] else ""
            print(f"  --source {c_['index']}   {c_['size']}{note}")
        return 0
    return _capture.run(a.source, outdir=a.out, n_shots=a.shots,
                        hold_frames=a.hold, width=a.width, height=a.height,
                        enroll=a.enroll)


def cmd_calibrate(a) -> int:
    return _calibrate.run(a.directory, arm=a.arm, write=not a.dry_run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dskin", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="analyse one photo, print gate result")
    c.add_argument("image")
    c.add_argument("--arm", default="card", choices=pipeline.NORM_ARMS)
    c.add_argument("--overlay-dir", default="out/overlays")
    c.set_defaults(func=cmd_check)

    i = sub.add_parser("ingest", help="measure a folder and store to sqlite")
    i.add_argument("directory")
    i.add_argument("--arm", default="card", choices=pipeline.NORM_ARMS)
    i.add_argument("--condition", default=None)
    i.add_argument("--db", default="data/skin.db")
    i.add_argument("--overlay-dir", default="out/overlays")
    i.set_defaults(func=cmd_ingest)

    e = sub.add_parser("experiment", help="go/no-go test over condition subfolders")
    e.add_argument("directory")
    e.add_argument("--out", default="out/experiment")
    e.add_argument("--overlay-dir", default="out/overlays")
    e.set_defaults(func=cmd_experiment)

    p = sub.add_parser("capture", help="live gated capture: shutter fires only when good")
    p.add_argument("--source", default="0",
                   help="camera index (0), a video file, or a folder of images")
    p.add_argument("--out", default="data/images")
    p.add_argument("--shots", type=int, default=3)
    p.add_argument("--hold", type=int, default=8,
                   help="consecutive good frames required before the shutter fires")
    p.add_argument("--enroll", default=None,
                   help="a previous photo to align against (keeps day 200 framed like day 1)")
    p.add_argument("--list", action="store_true", help="probe and list cameras, then exit")
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    p.set_defaults(func=cmd_capture)

    k = sub.add_parser("calibrate", help="learn gate thresholds from your baseline shoot")
    k.add_argument("directory")
    k.add_argument("--arm", default="card", choices=pipeline.NORM_ARMS)
    k.add_argument("--dry-run", action="store_true", help="print only, do not write")
    k.set_defaults(func=cmd_calibrate)

    a = ap.parse_args(argv)
    return a.func(a)
