"""Reusable Plotly chart builders for the dashboard."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


DARK_BG  = "#0F1117"
CARD_BG  = "#1A1D2E"
GRID_CLR = "#2D3045"
TEXT_CLR = "#E2E8F0"
ACCENT   = "#6366F1"

_LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_CLR, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
)


def training_curves(history: dict, title: str = "") -> go.Figure:
    """Plot train/val loss and val MAE on two y-axes."""
    epochs = list(range(len(history["train_loss"])))
    best   = history.get("best_epoch", 0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
        name="Train loss", line=dict(color="#6366F1", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
        name="Val loss", line=dict(color="#818CF8", width=2, dash="dot")), secondary_y=False)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_mae"],
        name="Val MAE (m)", line=dict(color="#34D399", width=2)), secondary_y=True)

    # Best epoch marker
    if best < len(history["val_mae"]):
        fig.add_vline(x=best, line=dict(color="#F59E0B", dash="dash", width=1.5),
                      annotation_text=f"best", annotation_position="top right")

    fig.update_layout(title=title or "Training curves", **_LAYOUT_BASE,
                      legend=dict(bgcolor=CARD_BG, bordercolor=GRID_CLR))
    fig.update_yaxes(title_text="Huber Loss", secondary_y=False, gridcolor=GRID_CLR)
    fig.update_yaxes(title_text="MAE (m)", secondary_y=True, gridcolor=GRID_CLR)
    return fig


def metric_radar(results_by_modality: dict[str, dict]) -> go.Figure:
    """Radar chart comparing all 6 metrics across modalities (normalized 0-1)."""
    metrics = ["LP MAE↓", "KNN MAE↓", "Alignment↓", "Eff.Dim↑", "Temporal↑", "Uniformity↓"]

    def _extract(ev: dict) -> list[float]:
        lp  = ev.get("linear_probe", {})
        kp  = ev.get("knn_probe", {})
        au  = ev.get("alignment_uniformity", {})
        ed  = ev.get("effective_dimensionality", {})
        ts  = ev.get("temporal_smoothness", {})
        return [
            lp.get("mae", 10),
            kp.get("mae", 10),
            au.get("alignment", 1),
            ed.get("participation_ratio", 0),
            ts.get("correlation", 0),
            abs(au.get("uniformity", 0)),
        ]

    # Collect all raw values for normalisation
    all_raw = {m: _extract(r.get("eval", {})) for m, r in results_by_modality.items()}
    if not all_raw:
        return go.Figure()

    raw_arr = np.array(list(all_raw.values()))
    mins = raw_arr.min(axis=0)
    maxs = raw_arr.max(axis=0)
    ranges = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)

    colors = {"wifi": "#3B82F6", "imu": "#10B981", "odom": "#F59E0B", "camera": "#EF4444"}
    fig = go.Figure()
    for mod, raw in all_raw.items():
        # Normalise so best=1, worst=0 (invert for ↓ metrics)
        normed = []
        for i, v in enumerate(raw):
            n = (v - mins[i]) / ranges[i]
            # For ↓ metrics (LP, KNN, Align, Uniform): invert
            if metrics[i].endswith("↓"):
                n = 1 - n
            normed.append(round(n, 3))
        fig.add_trace(go.Scatterpolar(
            r=normed + [normed[0]],
            theta=metrics + [metrics[0]],
            name=mod.upper(),
            line=dict(color=colors.get(mod, ACCENT), width=2),
            fill="toself",
            fillcolor=colors.get(mod, ACCENT) + "22",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_CLR,
                            tickfont=dict(color=TEXT_CLR)),
            angularaxis=dict(gridcolor=GRID_CLR),
        ),
        showlegend=True,
        title="Metric comparison (normalised, higher=better)",
        **{k: v for k, v in _LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        legend=dict(bgcolor=CARD_BG),
    )
    return fig


def mae_bar(results_by_modality: dict[str, dict]) -> go.Figure:
    """Bar chart of best val MAE per modality."""
    mods, maes, colors = [], [], []
    color_map = {"wifi": "#3B82F6", "imu": "#10B981", "odom": "#F59E0B", "camera": "#EF4444"}
    for mod, run in results_by_modality.items():
        mods.append(mod.upper())
        maes.append(run.get("best_val_mae", 0))
        colors.append(color_map.get(mod, ACCENT))

    fig = go.Figure(go.Bar(
        x=mods, y=maes, marker_color=colors,
        text=[f"{m:.3f}m" for m in maes], textposition="outside",
    ))
    fig.update_layout(title="Best Val MAE per modality", yaxis_title="MAE (m)",
                      **_LAYOUT_BASE)
    return fig


def embedding_scatter(z: np.ndarray, y: np.ndarray, title: str = "Embedding space (UMAP)") -> go.Figure:
    """2D UMAP scatter of embeddings coloured by position magnitude."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        z2d = reducer.fit_transform(z)
    except Exception:
        # Fallback to PCA if umap fails
        from sklearn.decomposition import PCA
        z2d = PCA(n_components=2).fit_transform(z)

    dist = np.sqrt((y ** 2).sum(axis=1))
    fig = go.Figure(go.Scatter(
        x=z2d[:, 0], y=z2d[:, 1],
        mode="markers",
        marker=dict(
            color=dist, colorscale="Viridis", size=4, opacity=0.7,
            colorbar=dict(title="||pos|| (m)", tickfont=dict(color=TEXT_CLR)),
            showscale=True,
        ),
        text=[f"({y[i,0]:.1f}, {y[i,1]:.1f})m" for i in range(len(y))],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(title=title, **_LAYOUT_BASE,
                      xaxis_title="UMAP-1", yaxis_title="UMAP-2")
    return fig


def position_scatter(y_pred: np.ndarray, y_true: np.ndarray) -> go.Figure:
    """Scatter of predicted vs true positions."""
    errors = np.sqrt(((y_pred - y_true) ** 2).sum(axis=1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true[:, 0], y=y_true[:, 1],
        mode="markers", name="Ground truth",
        marker=dict(color="#6B7280", size=4, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=y_pred[:, 0], y=y_pred[:, 1],
        mode="markers", name="Predicted",
        marker=dict(color=errors, colorscale="RdYlGn_r", size=5,
                    showscale=True,
                    colorbar=dict(title="Error (m)", tickfont=dict(color=TEXT_CLR))),
    ))
    fig.update_layout(title="Predicted vs ground truth positions",
                      xaxis_title="x (m)", yaxis_title="y (m)",
                      **_LAYOUT_BASE, legend=dict(bgcolor=CARD_BG))
    return fig


def live_training_chart(metrics: list[dict]) -> go.Figure:
    """Real-time training chart from JSONL rows."""
    if not metrics:
        return go.Figure().update_layout(title="Waiting for training...", **_LAYOUT_BASE)
    epochs     = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss   = [m["val_loss"] for m in metrics]
    val_mae    = [m["val_mae"] for m in metrics]
    best_eps   = [m["epoch"] for m in metrics if m.get("is_best")]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Loss (Huber)", "Val MAE (m)"),
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train",
                             line=dict(color="#6366F1", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val",
                             line=dict(color="#818CF8", dash="dot", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_mae, name="Val MAE",
                             line=dict(color="#34D399", width=2),
                             fill="tozeroy", fillcolor="#34D39922"), row=2, col=1)
    if best_eps:
        for ep in best_eps[-1:]:  # mark only last best
            fig.add_vline(x=ep, line=dict(color="#F59E0B", dash="dash", width=1.5))

    fig.update_layout(**_LAYOUT_BASE, showlegend=True,
                      legend=dict(bgcolor=CARD_BG))
    fig.update_xaxes(gridcolor=GRID_CLR)
    fig.update_yaxes(gridcolor=GRID_CLR)
    return fig


def flow_diagram(stage_meta: list[dict], modality_meta: dict) -> go.Figure:
    """Visual pipeline flow diagram with stages and modalities."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR, family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[0, 7]),
        title=dict(text="Pipeline Data Flow", font=dict(size=18, color=TEXT_CLR)),
        showlegend=False,
        height=420,
    )

    # Modality inputs (left column)
    mod_y = {"wifi": 6, "imu": 4.5, "odom": 3, "camera": 1.5}
    for mod, y in mod_y.items():
        meta = modality_meta[mod]
        color = meta["color"]
        fig.add_shape(type="rect", x0=0.1, x1=1.6, y0=y-0.4, y1=y+0.4,
                      fillcolor=color + "33", line=dict(color=color, width=2))
        fig.add_annotation(x=0.85, y=y, text=f"{meta['icon']} {meta['label']}",
                           showarrow=False, font=dict(size=11, color=color))
        # Arrow to Stage A
        fig.add_annotation(ax=1.6, ay=y, x=2.5, y=3.75,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1,
                           arrowwidth=1.5, arrowcolor=color + "88")

    # Stage boxes
    stage_x = [2.5, 4.2, 5.9, 7.6, 9.0]
    for i, (stage, sx) in enumerate(zip(stage_meta, stage_x)):
        color = stage["color"]
        fig.add_shape(type="rect", x0=sx, x1=sx+1.5, y0=3.1, y1=4.4,
                      fillcolor=color + "33", line=dict(color=color, width=2.5))
        status_icon = "✅" if stage["status"] == "done" else "⏳" if stage["status"] == "active" else "🔘"
        fig.add_annotation(x=sx+0.75, y=3.95, text=f"Stage {stage['id']}",
                           showarrow=False, font=dict(size=12, color=color, family="Inter"))
        fig.add_annotation(x=sx+0.75, y=3.55, text=f"{status_icon} {stage['name']}",
                           showarrow=False, font=dict(size=9, color=TEXT_CLR))
        # Arrow to next stage
        if i < len(stage_meta) - 1:
            nx = stage_x[i+1]
            fig.add_annotation(ax=sx+1.5, ay=3.75, x=nx, y=3.75,
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1.2,
                               arrowwidth=2, arrowcolor="#6B7280")

    # Output label
    fig.add_annotation(x=9.75, y=3.75, text="📍 (x,y)",
                       showarrow=False, font=dict(size=13, color="#F59E0B"))

    return fig
