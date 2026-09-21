"""d-skin dashboard.  Run:  streamlit run app.py

Deliberately plain. The dashboard's job is to show you the measurement AND its
noise floor together, so you never read a wiggle as a result.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dskin import db, config as C

# Categorical slots 1-4, assigned to ROIs in FIXED order and never cycled:
# colour follows the region, so filtering regions never repaints the survivors.
ROI_COLOR = {
    "forehead":    "#2a78d6",
    "cheek_left":  "#eb6834",
    "cheek_right": "#1baf7a",
    "chin":        "#eda100",
}
ROI_ORDER = list(ROI_COLOR)
METRICS = {
    "melanin": "Melanin index", "erythema": "Erythema index",
    "ita": "ITA (skin tone angle)", "L_mean": "Lightness L*",
    "tex_dog_1": "Texture (mid band)", "tex_local_std": "Texture (local contrast)",
}

st.set_page_config(page_title="d-skin", layout="wide")
alt.data_transformers.disable_max_rows()


@st.cache_data(ttl=30)
def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = db.load_df(path)
    if not df.empty and "captured_at" in df:
        df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce")
        df["date"] = df["captured_at"].dt.date
    return df


def tidy(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Wide ROI columns -> long, so one line per region."""
    rows = []
    for roi in ROI_ORDER:
        col = f"{roi}_{metric}"
        if col not in df:
            continue
        sub = df[["captured_at", "date", col, "iod_px", "yaw"]].dropna(subset=[col])
        rows.append(sub.assign(roi=roi).rename(columns={col: "value"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


st.title("d-skin")
st.caption("Image-based skin change tracking. Not a diagnosis.")

dbp = st.sidebar.text_input("Database", C.DB_PATH)
df = load(dbp)
if df.empty:
    st.info(f"No measurements yet. Run  `python -m dskin ingest data/images`  "
            f"to populate `{dbp}`.")
    st.stop()

arms = sorted(df["arm"].dropna().unique())
arm = st.sidebar.selectbox("Correction arm", arms,
                           index=arms.index("card") if "card" in arms else 0)
only_pass = st.sidebar.checkbox("Gate-passing photos only", value=True)
weekly = st.sidebar.checkbox("Weekly means (recommended)", value=True,
                             help="Daily points are autocorrelated; weekly means "
                                  "are what you should actually read trends from.")
rois = st.sidebar.multiselect("Regions", ROI_ORDER, default=ROI_ORDER)
metric = st.sidebar.selectbox("Metric", list(METRICS), format_func=METRICS.get)

d = df[df["arm"] == arm]
if only_pass and "passed" in d:
    d = d[d["passed"] == 1]

# --- headline tiles -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
total = len(df[df["arm"] == arm])
passed = int(df[(df["arm"] == arm)]["passed"].fillna(0).sum()) if "passed" in df else 0
c1.metric("Usable photos", f"{passed}", f"of {total} captured")
ndays = d["date"].nunique() if "date" in d else 0
c2.metric("Days collected", f"{ndays}")
c3.metric("Weeks to a readable trend", f"{max(0, 8 - ndays // 7)}",
          help="Expect 8-12 weeks before any trend is meaningful.")
rate = passed / total if total else 0
c4.metric("Gate pass rate", f"{rate:.0%}",
          help="A falling pass rate means your capture is drifting.")

if ndays < 14:
    st.warning(f"Only {ndays} days of data. Nothing here is a trend yet — "
               f"you are looking at noise. Keep collecting.")

# --- trend ----------------------------------------------------------------
long = tidy(d, metric)
long = long[long["roi"].isin(rois)] if not long.empty else long
if long.empty:
    st.warning("No values for that metric yet.")
    st.stop()

plot = long.copy()
if weekly:
    plot["captured_at"] = pd.to_datetime(plot["captured_at"]).dt.to_period("W").dt.start_time
    plot = (plot.groupby(["captured_at", "roi"], as_index=False)
                .agg(value=("value", "mean"), n=("value", "size")))
else:
    plot["n"] = 1

scale = alt.Scale(domain=ROI_ORDER, range=[ROI_COLOR[r] for r in ROI_ORDER])
base = alt.Chart(plot).encode(
    x=alt.X("captured_at:T", title=None, axis=alt.Axis(grid=False, labelColor="#52514e")),
    y=alt.Y("value:Q", title=METRICS[metric], scale=alt.Scale(zero=False),
            axis=alt.Axis(gridColor="#eceae4", domain=False, labelColor="#52514e")),
    color=alt.Color("roi:N", scale=scale, sort=ROI_ORDER,
                    legend=alt.Legend(title="Region", orient="top", direction="horizontal")),
)
chart = (base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45))
             .encode(tooltip=[alt.Tooltip("captured_at:T", title="Date"),
                              alt.Tooltip("roi:N", title="Region"),
                              alt.Tooltip("value:Q", title=METRICS[metric], format=".2f"),
                              alt.Tooltip("n:Q", title="Photos")]))
st.altair_chart(chart.properties(height=380).configure_view(stroke=None),
                use_container_width=True)

# --- is this metric actually measuring skin? ------------------------------
st.subheader("Sanity checks")
a, b = st.columns(2)
with a:
    st.markdown("**Does this metric track how you sat?**")
    out = []
    for roi in rois:
        s = long[long["roi"] == roi]
        if len(s) < 6 or s["iod_px"].isna().all():
            continue
        r = float(np.corrcoef(s["value"], s["iod_px"])[0, 1])
        out.append({"Region": roi, "r vs face size": round(r, 2),
                    "Verdict": "SUSPECT" if abs(r) > 0.5 else "ok"})
    if out:
        st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
        st.caption("A strong correlation with apparent face size means you are "
                   "measuring your seating position, not your skin.")
with b:
    st.markdown("**Day-to-day spread (your noise floor)**")
    fl = []
    for roi in rois:
        s = long[long["roi"] == roi]["value"]
        if len(s) < 3 or abs(s.mean()) < 1e-9:
            continue
        fl.append({"Region": roi, "SD as % of mean": round(s.std(ddof=1) / abs(s.mean()) * 100, 2)})
    if fl:
        st.dataframe(pd.DataFrame(fl), hide_index=True, use_container_width=True)
        st.caption("Any change smaller than this is not a result.")

# Table view: required relief for the low-contrast categorical slots, and the
# thing you will actually want when a point looks surprising.
with st.expander("Data table"):
    st.dataframe(plot.sort_values("captured_at", ascending=False),
                 hide_index=True, use_container_width=True)

with st.expander("Rejected photos"):
    rej = df[(df["arm"] == arm) & (df["passed"] != 1)]
    if rej.empty:
        st.write("None.")
    else:
        st.dataframe(rej[["image_id", "captured_at", "reject_reasons"]]
                     .sort_values("captured_at", ascending=False),
                     hide_index=True, use_container_width=True)
