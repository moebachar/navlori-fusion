"""Per-encoder deep dive — training curves, all metrics, embedding UMAP."""
import sys
sys.path.insert(0, ".")

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from dashboard.core.loader import best_run_per_modality, list_runs, extract_embeddings_from_run, MODALITY_META
from dashboard.core.charts import training_curves, embedding_scatter, position_scatter

st.set_page_config(page_title="Encoders", page_icon="🧠", layout="wide")
st.title("🧠 Encoder Deep Dive")

best = best_run_per_modality()
all_runs = list_runs()

if not best:
    st.info("No completed runs found. Train encoders first.")
    st.stop()

# ── Encoder selector ───────────────────────────────────────────────────────
mod_options = list(best.keys())
mod_labels  = {m: f"{MODALITY_META[m]['icon']} {MODALITY_META[m]['label']} — best MAE: {best[m]['best_val_mae']:.3f}m"
               for m in mod_options}
selected_mod = st.selectbox("Select modality", mod_options,
                             format_func=lambda m: mod_labels[m])

# Allow switching between runs for same modality
runs_for_mod = [r for r in all_runs if r.get("modality") == selected_mod]
run_labels   = {r["run_id"]: f"{r['run_id']}  (MAE={r['best_val_mae']:.3f}m, ep={r['best_epoch']})"
                for r in runs_for_mod}
selected_run_id = st.selectbox("Run", list(run_labels.keys()),
                                format_func=lambda k: run_labels[k])
run = next(r for r in runs_for_mod if r["run_id"] == selected_run_id)

meta = MODALITY_META[selected_mod]
st.markdown(f"### {meta['icon']} {meta['label']} · {meta['encoder']}")

# ── Key metrics row ────────────────────────────────────────────────────────
ev = run.get("eval", {})
lp = ev.get("linear_probe", {})
kp = ev.get("knn_probe", {})
au = ev.get("alignment_uniformity", {})
ed = ev.get("effective_dimensionality", {})
ts = ev.get("temporal_smoothness", {})

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Best MAE",     f"{run['best_val_mae']:.3f}m", help="Direct regression (encoder+head)")
c2.metric("LP MAE",       f"{lp.get('mae', 0):.3f}m",   help="Linear probe — tests if embeddings are linearly decodable")
c3.metric("KNN MAE",      f"{kp.get('mae', 0):.3f}m",   help="5-NN probe — non-parametric structure check")
c4.metric("Alignment ↓",  f"{au.get('alignment', 0):.4f}", help="Nearby positions → close embeddings. Lower=better.")
c5.metric("Eff. Dim",     f"{ed.get('participation_ratio', 0):.1f}/{ed.get('embed_dim', 128)}",
          help="How many of the 128 dims are actually used")
c6.metric("Temporal ↑",   f"{ts.get('correlation', 0):.3f}",
          help="Pearson r between Δembedding and Δposition. Higher=smoother.")

st.divider()

# ── Training curves ────────────────────────────────────────────────────────
lcol, rcol = st.columns([3, 2])
with lcol:
    fig = training_curves(run, title=f"Training — {selected_mod.upper()} ({selected_run_id})")
    st.plotly_chart(fig, use_container_width=True)

with rcol:
    st.markdown("**Run details**")
    st.json({
        "run_id":       run["run_id"],
        "modality":     run["modality"],
        "device":       run.get("device", "?"),
        "n_train":      run.get("n_train", "?"),
        "n_val":        run.get("n_val", "?"),
        "best_epoch":   run["best_epoch"],
        "best_val_mae": round(run["best_val_mae"], 4),
        "elapsed_sec":  round(run.get("elapsed_sec", 0), 1),
        "finished_at":  run.get("finished_at", "?"),
    })

st.divider()

# ── Embedding visualisation ────────────────────────────────────────────────
st.markdown("### Embedding space")
st.caption("UMAP projection of val-set embeddings, coloured by distance from origin (position magnitude).")

if selected_mod == "camera":
    st.info("Camera embedding visualisation requires image loading — not supported live. Run the extraction script instead.")
else:
    with st.spinner("Loading model and extracting embeddings..."):
        try:
            # Build a minimal DataModule for val embeddings
            import sys
            sys.path.insert(0, ".")
            from src.pipeline.data import FusionDataModule

            @st.cache_resource(show_spinner=False)
            def _get_dm(mod):
                dm = FusionDataModule(
                    data_dir="data/async_collection",
                    train_paths=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    val_paths=[2, 13, 14],
                    test_paths=[15, 16, 17],
                    modalities=[mod],
                    batch_size=256,
                    normalize=True,
                    wifi_pca=32 if mod == "wifi" else None,
                )
                dm.setup()
                return dm

            dm = _get_dm(selected_mod)
            z, y = extract_embeddings_from_run(run["run_dir"], dm, selected_mod)

            if z is not None:
                lcol2, rcol2 = st.columns(2)
                with lcol2:
                    st.plotly_chart(embedding_scatter(z, y), use_container_width=True)
                with rcol2:
                    # Predict positions with a quick KNN to show position scatter
                    from sklearn.neighbors import KNeighborsRegressor
                    X_tr, y_tr = dm.train_ds.get_tensors(selected_mod)
                    from src.pipeline.encoders import WiFiNet, IMUCNN, OdomCNN
                    import torch
                    enc_map = {
                        "wifi": WiFiNet(n_aps=32, embed_dim=128, n_anchors=64),
                        "imu":  IMUCNN(in_features=9, embed_dim=128),
                        "odom": OdomCNN(in_features=7, embed_dim=128),
                    }
                    enc = enc_map[selected_mod]
                    import os
                    weights = os.path.join(run["run_dir"], "encoder.pt")
                    enc.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
                    enc.eval()
                    with torch.no_grad():
                        z_tr_all = []
                        for i in range(0, len(X_tr), 512):
                            z_tr_all.append(enc(X_tr[i:i+512]))
                        z_tr = torch.cat(z_tr_all).numpy()
                    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
                    knn.fit(z_tr, y_tr.numpy())
                    y_pred = knn.predict(z)
                    st.plotly_chart(position_scatter(y_pred, y), use_container_width=True)
        except Exception as e:
            st.error(f"Could not extract embeddings: {e}")

# ── All metrics detail ─────────────────────────────────────────────────────
st.divider()
st.markdown("### All 6 evaluation metrics — details")
m1, m2 = st.columns(2)
with m1:
    st.markdown("**Linear Probe** — trains a linear head on frozen embeddings")
    if lp:
        st.json(lp)
    st.markdown("**KNN Probe** — 5-nearest-neighbour position prediction")
    if kp:
        st.json(kp)
    st.markdown("**Temporal Smoothness** — embedding changes ↔ physical displacement")
    if ts:
        st.json(ts)
with m2:
    st.markdown("**Alignment & Uniformity** (Wang & Isola, ICML 2020)")
    if au:
        st.json(au)
    st.markdown("**Effective Dimensionality** — participation ratio")
    if ed:
        st.json(ed)
