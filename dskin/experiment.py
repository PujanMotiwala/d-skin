"""The go/no-go experiment.

Variance alone cannot decide this. A correction can be beautifully CONSISTENT and
systematically WRONG -- a colour-constancy model that partly reads your skin as
the illuminant will quietly suppress the very change you are tracking, while
looking impressively stable. Comparing arms against a physical reference is what
separates "stable" from "stably wrong".
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PRIMARY = ["melanin", "erythema", "ita", "L_mean", "tex_dog_1", "tex_local_std"]


def _metric_cols(df: pd.DataFrame) -> list[str]:
    keep = []
    for c in df.columns:
        if df[c].dtype.kind not in "fi":
            continue
        if any(c.endswith("_" + p) or c == p for p in PRIMARY):
            keep.append(c)
    return sorted(keep)


def stability(df: pd.DataFrame, baseline="baseline") -> pd.DataFrame:
    """Per metric, per arm: noise floor vs nuisance sensitivity, as %-of-mean.

    Compare the numbers against a realistic 12-week effect. Pigmentation moves
    single-digit percent; acne lesion counts move far more.
    """
    rows = []
    for arm, g in df.groupby("arm"):
        conds = [c for c in g["condition"].dropna().unique()]
        if baseline not in conds:
            continue
        base = g[g["condition"] == baseline]
        for m in _metric_cols(g):
            b = base[m].dropna()
            if len(b) < 3 or not np.isfinite(b.mean()) or abs(b.mean()) < 1e-9:
                continue
            scale = abs(b.mean())
            floor = b.std(ddof=1) / scale * 100
            rec = {"arm": arm, "metric": m, "noise_floor_pct": floor,
                   "n_baseline": len(b)}
            worst = floor
            for c in conds:
                if c == baseline:
                    continue
                v = g[g["condition"] == c][m].dropna()
                if len(v) < 3:
                    continue
                s = v.std(ddof=1) / scale * 100
                bias = (v.mean() - b.mean()) / scale * 100
                rec[f"{c}_sd_pct"] = s
                rec[f"{c}_bias_pct"] = bias
                worst = max(worst, abs(bias) + s)
            rec["worst_case_pct"] = worst
            rec["nuisance_over_floor"] = worst / floor if floor > 1e-9 else np.inf
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["arm", "worst_case_pct"])


def probe(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Can a linear model read the CONDITION off the measurements?

    This is the batch-effect probe from the pathology literature. High accuracy
    means nuisance is linearly encoded and will dominate any linear trend model
    you build later. Chance level is the majority-class rate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    out = []
    for arm, g in df.groupby("arm"):
        g = g.dropna(subset=["condition"])
        cols = _metric_cols(g)
        X = g[cols].to_numpy(float)
        ok = np.isfinite(X).all(axis=1)
        X, y = X[ok], g["condition"].to_numpy()[ok]
        if len(np.unique(y)) < 2 or len(y) < 10:
            continue
        counts = pd.Series(y).value_counts()
        k = int(min(n_splits, counts.min()))
        if k < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, C=0.5))
        acc = cross_val_score(clf, X, y,
                              cv=StratifiedKFold(k, shuffle=True, random_state=seed)).mean()
        chance = counts.max() / counts.sum()
        out.append({"arm": arm, "probe_accuracy": acc, "chance": chance,
                    "lift_over_chance": acc - chance, "n": len(y),
                    "n_features": len(cols)})
    return pd.DataFrame(out).sort_values("probe_accuracy")


def probe_pairwise(df: pd.DataFrame, baseline: str = "baseline",
                   n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Probe each nuisance condition AGAINST baseline, separately.

    The pooled probe conflates conditions: an arm that cleans up lighting can end
    up MORE separable overall, because the pose difference it never addressed is
    no longer buried in lighting noise. Pairwise asks the question each arm is
    actually responsible for.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    out = []
    for arm, g in df.groupby("arm"):
        g = g.dropna(subset=["condition"])
        if baseline not in set(g["condition"]):
            continue
        cols = _metric_cols(g)
        for cond in sorted(set(g["condition"]) - {baseline}):
            sub = g[g["condition"].isin([baseline, cond])]
            X = sub[cols].to_numpy(float)
            ok = np.isfinite(X).all(axis=1)
            X, y = X[ok], sub["condition"].to_numpy()[ok]
            if len(np.unique(y)) < 2:
                continue
            counts = pd.Series(y).value_counts()
            k = int(min(n_splits, counts.min()))
            if k < 2:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=4000, C=0.5))
            acc = cross_val_score(clf, X, y,
                                  cv=StratifiedKFold(k, shuffle=True, random_state=seed)).mean()
            chance = counts.max() / counts.sum()
            out.append({"arm": arm, "condition": cond, "probe_accuracy": acc,
                        "chance": chance, "lift_over_chance": acc - chance, "n": len(y)})
    return pd.DataFrame(out).sort_values(["condition", "probe_accuracy"])


def pose_confound(df: pd.DataFrame) -> pd.DataFrame:
    """Does a metric track how you sat rather than your skin?

    A texture metric correlating with iod_px (apparent face size) or |yaw| is the
    resampling/foreshortening confound showing up. Treat a strong correlation as
    a broken metric, not a finding.
    """
    rows = []
    for arm, g in df.groupby("arm"):
        for m in _metric_cols(g):
            for cov in ("iod_px", "yaw", "pitch"):
                if cov not in g:
                    continue
                a, b = g[m], g[cov].abs() if cov != "iod_px" else g[cov]
                ok = a.notna() & b.notna()
                if ok.sum() < 6:
                    continue
                r = float(np.corrcoef(a[ok], b[ok])[0, 1])
                rows.append({"arm": arm, "metric": m, "covariate": cov, "pearson_r": r})
    d = pd.DataFrame(rows)
    return d.reindex(d.pearson_r.abs().sort_values(ascending=False).index) if len(d) else d


def verdict(stab: pd.DataFrame, prb: pd.DataFrame,
            pair: pd.DataFrame | None = None) -> str:
    lines = ["", "=" * 68, "VERDICT", "=" * 68]
    if stab.empty:
        return "\n".join(lines + ["Not enough data. Need >=3 usable photos per condition."])

    best = (stab.groupby("arm")["worst_case_pct"].median().sort_values())
    lines.append("\nMedian worst-case nuisance error, by correction arm")
    lines.append("(lower is better; compare against your expected real effect):")
    for arm, v in best.items():
        lines.append(f"   {arm:<12s} {v:8.2f} %")

    win = best.index[0]
    lines.append(f"\n-> Best arm: '{win}'")
    if "card" in best.index and win != "card":
        lines.append("   NOTE: an arm beat the physical card. Check the card was")
        lines.append("   detected correctly (look at the overlays) before believing it.")
    if "card" in best.index and "grayworld" in best.index:
        gap = best["grayworld"] - best["card"]
        lines.append(f"\n   grayworld - card = {gap:+.2f} pct points")
        lines.append("   If grayworld is close to card, a learned illuminant estimate may")
        lines.append("   be enough and you can drop the card. If it is much worse, the card")
        lines.append("   is doing real work and no AWB model will substitute for it.")

    if pair is not None and not pair.empty:
        lines.append("\nNuisance probe, per condition vs baseline")
        lines.append("(can a linear model still tell them apart after correction?):")
        for cond, gg in pair.groupby("condition"):
            lines.append(f"   [{cond}]")
            for _, r in gg.iterrows():
                flag = "OK" if r.lift_over_chance < 0.15 else "LEAKS"
                lines.append(f"      {r.arm:<12s} acc={r.probe_accuracy:.2f} "
                             f"lift={r.lift_over_chance:+.2f}  {flag}")
        lines.append("   Lift near zero = that nuisance is no longer readable. Good.")
        lines.append("   Judge each arm on the nuisance it is meant to fix: a colour")
        lines.append("   correction owns [lighting], not [posedist].")
    elif not prb.empty:
        lines.append("\nPooled nuisance probe:")
        for _, r in prb.iterrows():
            flag = "OK" if r.lift_over_chance < 0.15 else "LEAKS"
            lines.append(f"   {r.arm:<12s} acc={r.probe_accuracy:.2f} "
                         f"lift={r.lift_over_chance:+.2f}  {flag}")

    w = float(best.iloc[0])
    lines.append("")
    if w < 2:
        lines.append(f"Worst-case nuisance ~{w:.1f}% vs a typical 12-week pigmentation")
        lines.append("effect of single-digit %. Workable. Start collecting.")
    elif w < 5:
        lines.append(f"Worst-case nuisance ~{w:.1f}%. Marginal. Usable for large effects")
        lines.append("(acne lesion counts) but not for subtle pigmentation. Tighten the")
        lines.append("gate, fix the room, and re-run before committing to 12 weeks.")
    else:
        lines.append(f"Worst-case nuisance ~{w:.1f}%, comparable to or larger than the effect")
        lines.append("you want to detect. Do NOT start a 12-week collection yet -- you would")
        lines.append("be recording a random walk. Fix the dominant nuisance first (see the")
        lines.append("per-condition bias columns above to find which one).")
    return "\n".join(lines)
