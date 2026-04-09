"""NavLoRI-Fusion — System Dashboard.

Launch:
    streamlit run dashboard/app.py --server.port 8501
"""
import sys
from pathlib import Path

# Always run from repo root so relative paths work
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import os
os.chdir(repo_root)

import streamlit as st

st.set_page_config(
    page_title="Fusion Dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global dark theme CSS
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0D0F1A; }
[data-testid="stSidebarNav"] a { color: #94A3B8 !important; }
[data-testid="stSidebarNav"] a:hover { color: #E2E8F0 !important; }
.stMetric label { color: #94A3B8; font-size: 12px; }
.stMetric [data-testid="stMetricValue"] { font-size: 22px; font-weight: 700; }
div[data-testid="stHorizontalBlock"] > div { gap: 12px; }
</style>
""", unsafe_allow_html=True)

# Sidebar branding
with st.sidebar:
    st.markdown("## 🧭 Fusion Dashboard")
    st.caption("Indoor localization · TIAGO++ · CESI LINEACT")
    st.divider()
    st.markdown("**Navigate**")

# Landing page content (shown when no sub-page is selected)
st.title("🧭 Fusion System Dashboard")
st.markdown("""
Welcome. Use the sidebar to navigate between views.

| Page | What you get |
|---|---|
| 📊 **Overview** | System health, all encoder MAEs, stage progress, pipeline diagram |
| 🧠 **Encoders** | Per-encoder deep dive: training curves, UMAP, all 6 metrics, position scatter |
| 📁 **Dataset** | Sample inspector, position distributions, normalization stats |
| 📈 **Training Monitor** | Live training curves (polls every 3s), historical run browser |

---

### Quick status
""")

# Mini status from loader
from dashboard.core.loader import best_run_per_modality, MODALITY_META, STAGE_META

best = best_run_per_modality()
cols = st.columns(5)
for col, stage in zip(cols, STAGE_META):
    icon = "✅" if stage["status"] == "done" else "🔘"
    col.markdown(f"**Stage {stage['id']}**  \n{icon} {stage['name']}")

st.divider()
cols = st.columns(4)
for col, (mod, meta) in zip(cols, MODALITY_META.items()):
    run = best.get(mod)
    if run:
        col.metric(f"{meta['icon']} {meta['label']}", f"{run['best_val_mae']:.3f}m",
                   help=f"Best val MAE · {meta['encoder']}")
    else:
        col.metric(f"{meta['icon']} {meta['label']}", "—", help="Not trained yet")
