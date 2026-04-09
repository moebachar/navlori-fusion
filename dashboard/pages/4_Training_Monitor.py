"""Real-time training monitor — polls JSONL metrics file live."""
import sys
sys.path.insert(0, ".")

import time
import streamlit as st

from dashboard.core.loader import active_run, list_runs, MODALITY_META
from dashboard.core.charts import live_training_chart, training_curves

st.set_page_config(page_title="Training Monitor", page_icon="📈", layout="wide")
st.title("📈 Training Monitor")

# ── Live training ──────────────────────────────────────────────────────────
live = active_run()

if live:
    meta    = live["meta"]
    metrics = live["metrics"]
    mod     = meta.get("modality", "?")
    m_meta  = MODALITY_META.get(mod, {})

    st.success(f"**Live training detected** — {m_meta.get('icon','')} {mod.upper()}")
    st.caption(f"Run: `{meta.get('run_id')}` · Device: `{meta.get('device')}` · "
               f"Train: {meta.get('n_train')} · Val: {meta.get('n_val')}")

    auto_refresh = st.toggle("Auto-refresh (every 3s)", value=True)

    chart_placeholder = st.empty()
    stats_placeholder = st.empty()

    if metrics:
        last = metrics[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epoch",      last["epoch"])
        c2.metric("Train loss", f"{last['train_loss']:.4f}")
        c3.metric("Val loss",   f"{last['val_loss']:.4f}")
        c4.metric("Val MAE",    f"{last['val_mae']:.3f}m")

    chart_placeholder.plotly_chart(
        live_training_chart(metrics), use_container_width=True
    )

    if auto_refresh:
        time.sleep(3)
        st.rerun()

else:
    st.info("No active training run detected. Start training from the notebook or a script.")

st.divider()

# ── Historical runs ────────────────────────────────────────────────────────
st.markdown("### All completed runs")
all_runs = list_runs()

if not all_runs:
    st.warning("No completed runs found in `runs/` directory.")
    st.stop()

# Filter controls
mods_available = sorted(set(r.get("modality", "?") for r in all_runs))
filter_mod = st.multiselect("Filter by modality", mods_available, default=mods_available)
filtered = [r for r in all_runs if r.get("modality") in filter_mod]

# Summary table
import pandas as pd
rows = []
for r in filtered:
    rows.append({
        "Run ID":       r["run_id"],
        "Modality":     r.get("modality", "?").upper(),
        "Best MAE (m)": round(r["best_val_mae"], 3),
        "Best Epoch":   r["best_epoch"],
        "Duration (s)": round(r.get("elapsed_sec", 0), 1),
        "Finished":     r.get("finished_at", "?")[:16] if r.get("finished_at") else "?",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Detailed run viewer
st.markdown("### Training curves — select a run")
run_options = {r["run_id"]: r for r in filtered}
selected_id = st.selectbox("Run", list(run_options.keys()))
selected_run = run_options[selected_id]

st.plotly_chart(training_curves(selected_run, title=selected_id), use_container_width=True)

# Learning rate curve
lr_vals = []
import json
from pathlib import Path
metrics_path = Path(selected_run["run_dir"]) / "metrics.jsonl"
if metrics_path.exists():
    for line in metrics_path.read_text().splitlines():
        try:
            lr_vals.append(json.loads(line).get("lr", 0))
        except Exception:
            pass

if lr_vals:
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        y=lr_vals, mode="lines", line=dict(color="#F59E0B", width=2)
    ))
    fig.update_layout(
        title="Learning rate schedule (OneCycleLR)",
        xaxis_title="Epoch", yaxis_title="LR",
        paper_bgcolor="#0F1117", plot_bgcolor="#1A1D2E",
        font=dict(color="#E2E8F0"),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor="#2D3045"),
        yaxis=dict(gridcolor="#2D3045"),
        height=200,
    )
    st.plotly_chart(fig, use_container_width=True)
