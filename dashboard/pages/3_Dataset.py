"""Dataset explorer — path stats, sample inspector, modality distributions."""
import sys
sys.path.insert(0, ".")

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch

st.set_page_config(page_title="Dataset", page_icon="📁", layout="wide")
st.title("📁 Dataset Explorer")

DATA_DIR = "data/async_collection"
DARK_BG = "#0F1117"
CARD_BG = "#1A1D2E"
GRID    = "#2D3045"
TEXT    = "#E2E8F0"

@st.cache_resource(show_spinner="Loading dataset...")
def load_dm():
    from src.pipeline.data import FusionDataModule
    dm = FusionDataModule(
        data_dir=DATA_DIR,
        train_paths=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        val_paths=[2, 13, 14],
        test_paths=[15, 16, 17],
        modalities=["imu", "odom", "wifi"],
        batch_size=256,
        normalize=False,
        wifi_pca=None,
    )
    dm.setup()
    return dm

dm = load_dm()

# ── Dataset overview ───────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train samples", f"{len(dm.train_ds):,}")
c2.metric("Val samples",   f"{len(dm.val_ds):,}")
c3.metric("Test samples",  f"{len(dm.test_ds):,}")
c4.metric("Total",         f"{len(dm.train_ds)+len(dm.val_ds)+len(dm.test_ds):,}")

st.divider()

# ── Ground truth position distribution ────────────────────────────────────
st.markdown("### Ground truth positions per split")
splits = [("Train", dm.train_ds), ("Val", dm.val_ds), ("Test", dm.test_ds)]
cols = st.columns(3)
for col, (name, ds) in zip(cols, splits):
    targets = ds._targets.numpy()
    fig = go.Figure(go.Scatter(
        x=targets[:, 0], y=targets[:, 1], mode="markers",
        marker=dict(size=3, opacity=0.5, color="#6366F1"),
    ))
    fig.update_layout(
        title=f"{name} ({len(ds):,} samples)",
        xaxis_title="x (m)", yaxis_title="y (m)",
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT), margin=dict(l=30, r=10, t=40, b=30),
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, scaleanchor="x"),
        height=300,
    )
    col.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Sample inspector ───────────────────────────────────────────────────────
st.markdown("### Sample inspector")
split_choice = st.selectbox("Split", ["Train", "Val", "Test"])
ds_map = {"Train": dm.train_ds, "Val": dm.val_ds, "Test": dm.test_ds}
ds = ds_map[split_choice]

idx = st.slider("Sample index", 0, len(ds) - 1, 0)
sample = ds[idx]

s1, s2, s3 = st.columns([1, 2, 2])
with s1:
    row = ds._gt_rows[idx]
    st.markdown("**Sample info**")
    st.json({
        "index":     idx,
        "path_id":   row["path_id"],
        "timestamp": round(float(sample["timestamp"]), 3),
        "target_x":  round(float(sample["target"][0]), 3),
        "target_y":  round(float(sample["target"][1]), 3),
    })

with s2:
    from src.pipeline.data.dataset import IMU_COLS
    imu = sample["imu"].numpy()
    fig = go.Figure()
    for i, col_name in enumerate(IMU_COLS[:6]):
        fig.add_trace(go.Scatter(y=imu[:, i], name=col_name, mode="lines"))
    fig.update_layout(title="IMU window (32 steps)", height=280,
                      paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                      font=dict(color=TEXT), margin=dict(l=30, r=10, t=40, b=30),
                      xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
                      legend=dict(bgcolor=CARD_BG, font=dict(size=9)))
    st.plotly_chart(fig, use_container_width=True)

with s3:
    odom = sample["odom"].numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=odom[:, 0], y=odom[:, 1], mode="lines+markers",
                             marker=dict(size=5), line=dict(color="#F59E0B")))
    fig.add_trace(go.Scatter(x=[odom[0, 0]], y=[odom[0, 1]], mode="markers",
                             marker=dict(size=12, color="green", symbol="square"),
                             name="start"))
    fig.add_trace(go.Scatter(x=[odom[-1, 0]], y=[odom[-1, 1]], mode="markers",
                             marker=dict(size=12, color="red", symbol="triangle-up"),
                             name="end"))
    fig.update_layout(title="Odom trajectory (16 steps)", height=280,
                      paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                      font=dict(color=TEXT), margin=dict(l=30, r=10, t=40, b=30),
                      xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, scaleanchor="x"),
                      legend=dict(bgcolor=CARD_BG, font=dict(size=9)),
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# WiFi fingerprint
wifi = sample["wifi"].numpy().flatten()
fig = go.Figure(go.Bar(y=wifi, marker_color="#3B82F6"))
fig.update_layout(title=f"WiFi fingerprint ({len(wifi)} features — raw RSSI)",
                  height=200, paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                  font=dict(color=TEXT), margin=dict(l=30, r=10, t=40, b=30),
                  xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Feature statistics ─────────────────────────────────────────────────────
st.markdown("### Normalization stats (training set)")
from src.pipeline.data.dataset import IMU_COLS, ODOM_COLS

dm_norm = st.session_state.get("dm_norm")
if dm_norm is None:
    @st.cache_resource(show_spinner="Computing stats...")
    def load_norm_dm():
        from src.pipeline.data import FusionDataModule
        d = FusionDataModule(
            data_dir=DATA_DIR,
            train_paths=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            val_paths=[2, 13, 14],
            test_paths=[15, 16, 17],
            modalities=["imu", "odom", "wifi"],
            normalize=True, wifi_pca=32,
        )
        d.setup()
        return d
    dm_norm = load_norm_dm()

tab1, tab2, tab3 = st.tabs(["IMU", "Odometry", "WiFi (PCA)"])
for tab, (mod, cols) in zip([tab1, tab2, tab3],
                             [("imu", IMU_COLS), ("odom", ODOM_COLS),
                              ("wifi", [f"PC_{i}" for i in range(32)])]):
    with tab:
        stats = dm_norm.train_ds.stats.get(mod, {})
        if stats:
            df = pd.DataFrame({
                "Feature": cols,
                "Mean":    stats["mean"].tolist(),
                "Std":     stats["std"].tolist(),
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Mean", x=df["Feature"], y=df["Mean"], marker_color="#6366F1"))
            fig.add_trace(go.Bar(name="Std",  x=df["Feature"], y=df["Std"],  marker_color="#34D399"))
            fig.update_layout(barmode="group", height=280,
                              paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                              font=dict(color=TEXT), margin=dict(l=30,r=10,t=20,b=60),
                              xaxis=dict(gridcolor=GRID, tickangle=-45),
                              yaxis=dict(gridcolor=GRID),
                              legend=dict(bgcolor=CARD_BG))
            st.plotly_chart(fig, use_container_width=True)
