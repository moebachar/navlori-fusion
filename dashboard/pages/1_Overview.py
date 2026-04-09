"""System overview — health dashboard, all encoders at a glance."""
import sys
sys.path.insert(0, ".")

import streamlit as st
import plotly.graph_objects as go

from dashboard.core.loader import best_run_per_modality, dataset_stats, MODALITY_META, STAGE_META
from dashboard.core.charts import mae_bar, metric_radar, flow_diagram

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

st.markdown("""
<style>
.metric-card {
    background: #1A1D2E; border-radius: 12px; padding: 16px 20px;
    border: 1px solid #2D3045;
}
.metric-card h3 { margin: 0 0 4px 0; font-size: 13px; color: #94A3B8; }
.metric-card p  { margin: 0; font-size: 26px; font-weight: 700; }
.section-title  { font-size: 18px; font-weight: 600; color: #E2E8F0; margin: 24px 0 12px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 System Overview")
st.caption("Indoor localization · 4-modality fusion · TIAGO++ robot")

# ── Data flow diagram ──────────────────────────────────────────────────────
st.plotly_chart(flow_diagram(STAGE_META, MODALITY_META), use_container_width=True)

# ── Dataset stats ──────────────────────────────────────────────────────────
ds = dataset_stats()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total paths",    len(ds["paths"]))
col2.metric("Total GT samples", f"{ds['total_samples']:,}")
col3.metric("Modalities",     "4")
col4.metric("Current stage",  "A — Encoders ✅")

st.divider()

# ── Stage progress ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Pipeline stages</div>', unsafe_allow_html=True)
cols = st.columns(len(STAGE_META))
for col, stage in zip(cols, STAGE_META):
    icon  = "✅" if stage["status"] == "done" else "🔘"
    color = stage["color"]
    col.markdown(
        f'<div class="metric-card"><h3>Stage {stage["id"]}</h3>'
        f'<p style="color:{color}">{icon} {stage["name"]}</p></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Encoder results ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Encoder performance (best runs)</div>', unsafe_allow_html=True)

best = best_run_per_modality()

if not best:
    st.info("No completed training runs found. Train encoders first: run the `encoder_workbench` notebook or `scripts/profile_training.py`.")
else:
    # Summary cards
    cols = st.columns(4)
    for col, (mod, meta) in zip(cols, MODALITY_META.items()):
        run = best.get(mod)
        if run:
            mae   = run["best_val_mae"]
            epoch = run["best_epoch"]
            col.markdown(
                f'<div class="metric-card">'
                f'<h3>{meta["icon"]} {meta["label"]} · {meta["encoder"]}</h3>'
                f'<p style="color:{meta["color"]}">{mae:.3f} m</p>'
                f'<small style="color:#64748B">MAE · best @ epoch {epoch}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f'<div class="metric-card">'
                f'<h3>{meta["icon"]} {meta["label"]}</h3>'
                f'<p style="color:#4B5563">Not trained</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    lcol, rcol = st.columns(2)
    with lcol:
        st.plotly_chart(mae_bar(best), use_container_width=True)
    with rcol:
        st.plotly_chart(metric_radar(best), use_container_width=True)

    # Full metric table
    st.markdown('<div class="section-title">Full metric table</div>', unsafe_allow_html=True)
    rows = []
    for mod, run in best.items():
        ev = run.get("eval", {})
        lp = ev.get("linear_probe", {})
        kp = ev.get("knn_probe", {})
        au = ev.get("alignment_uniformity", {})
        ed = ev.get("effective_dimensionality", {})
        ts = ev.get("temporal_smoothness", {})
        rows.append({
            "Modality":     mod.upper(),
            "Encoder":      MODALITY_META[mod]["encoder"],
            "Best MAE (m)": f"{run['best_val_mae']:.3f}",
            "Best Epoch":   run["best_epoch"],
            "LP MAE (m)":   f"{lp.get('mae', '-'):.3f}" if lp else "—",
            "KNN MAE (m)":  f"{kp.get('mae', '-'):.3f}" if kp else "—",
            "Alignment ↓":  f"{au.get('alignment', '-'):.4f}" if au else "—",
            "Uniformity ↓": f"{au.get('uniformity', '-'):.4f}" if au else "—",
            "Eff. Dim":     f"{ed.get('participation_ratio', '-'):.1f}/{ed.get('embed_dim', 128)}" if ed else "—",
            "Temporal ↑":   f"{ts.get('correlation', '-'):.3f}" if ts else "—",
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
