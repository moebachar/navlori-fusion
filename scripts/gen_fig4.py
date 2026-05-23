"""
Fig 4 — Temporal Smoothness: lollipop with interpretation zones
+ secondary axis showing embedding volatility (std_dz / mean_dz).
"""
import json, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333333', 'axes.linewidth': 0.8,
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'text.color': '#1a1a1a', 'axes.labelcolor': '#1a1a1a',
    'grid.color': '#DDDDDD', 'grid.linewidth': 0.6,
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

colors = {'wifi':'#2166AC','imu':'#D6604D','odom':'#4DAC26','camera':'#762A83'}
mods   = ['wifi','imu','odom','camera']
labels = ['WiFi\n(Anchor2Vec)', 'IMU\n(IMUCNN)', 'Odom\n(OdomCNN)', 'Camera\n(VisionViT)']

evals = {m: json.load(open(glob.glob(f'runs/{m}_*/eval.json')[0])) for m in mods}

r_vals    = [evals[m]['temporal_smoothness']['correlation'] for m in mods]
mean_dz   = [evals[m]['temporal_smoothness']['mean_dz']    for m in mods]
std_dz    = [evals[m]['temporal_smoothness']['std_dz']      for m in mods]
cv        = [s / (mu + 1e-9) for mu, s in zip(mean_dz, std_dz)]   # coefficient of variation

fig, (ax_main, ax_cv) = plt.subplots(
    1, 2, figsize=(10, 4.2),
    gridspec_kw={'width_ratios': [2.2, 1], 'wspace': 0.55}
)

# ── LEFT PANEL: lollipop with interpretation zones ─────────────────────────
y_pos = np.arange(len(mods))

# shaded zones
ax_main.axvspan(0.00, 0.30, color='#FFEBEE', alpha=0.7, zorder=0)
ax_main.axvspan(0.30, 0.55, color='#FFF8E1', alpha=0.7, zorder=0)
ax_main.axvspan(0.55, 1.00, color='#E8F5E9', alpha=0.7, zorder=0)

# zone labels at top
for x, lbl, col in [(0.15, 'weak', '#E57373'),
                     (0.42, 'moderate', '#FFB74D'),
                     (0.77, 'strong', '#66BB6A')]:
    ax_main.text(x, len(mods) - 0.25, lbl, ha='center', va='bottom',
                 fontsize=8.5, color=col, fontweight='bold', style='italic')

# lollipops
for i, (m, r, mu, sig) in enumerate(zip(mods, r_vals, mean_dz, std_dz)):
    c = colors[m]
    # stem
    ax_main.plot([0, r], [i, i], color=c, linewidth=2.4, solid_capstyle='round', zorder=2)
    # head
    ax_main.scatter([r], [i], color=c, s=140, zorder=5, edgecolors='white', linewidths=1.2)
    # r value label
    ax_main.text(r + 0.018, i, f'r = {r:.3f}', va='center',
                 color=c, fontsize=10, fontweight='bold')
    # μ ± σ annotation above stem
    ax_main.text(r / 2, i + 0.28, f'μ={mu:.2f}  σ={sig:.2f}',
                 ha='center', va='bottom', color='#555555', fontsize=7.8, style='italic')

ax_main.set_yticks(y_pos)
ax_main.set_yticklabels(labels, fontsize=10)
ax_main.set_xlim(0, 0.82)
ax_main.set_ylim(-0.6, len(mods) - 0.2)
ax_main.set_xlabel('Pearson r  ( ||Δz||₂  vs  ||Δy||₂ )  ↑  higher = smoother', fontsize=10)
ax_main.set_title('Temporal correlation', fontsize=11, fontweight='bold', loc='left')
ax_main.axvline(0, color='#cccccc', linewidth=0.6)
ax_main.grid(True, axis='x', alpha=0.5)

# ── RIGHT PANEL: embedding volatility (CV = std_dz / mean_dz) ──────────────
# horizontal bars — lower CV = more stable embeddings
ax_cv.barh(y_pos, cv,
           color=[colors[m] for m in mods],
           height=0.45, alpha=0.80, edgecolor='white', linewidth=0.8)

for i, (v, m) in enumerate(zip(cv, mods)):
    ax_cv.text(v + 0.1, i, f'{v:.1f}×', va='center',
               color=colors[m], fontsize=9.5, fontweight='bold')

ax_cv.set_yticks(y_pos)
ax_cv.set_yticklabels([])
ax_cv.set_xlabel('σ / μ  of  ||Δz||₂\n(lower = stable)', fontsize=9.5)
ax_cv.set_title('Volatility', fontsize=11, fontweight='bold', loc='left')
ax_cv.axvline(1.0, color='#aaaaaa', linewidth=0.8, linestyle='--')
ax_cv.text(1.05, -0.5, 'σ = μ', color='#aaaaaa', fontsize=7.5, style='italic')
ax_cv.grid(True, axis='x', alpha=0.4)
ax_cv.set_xlim(0, 12)

# ── shared supertitle ──────────────────────────────────────────────────────
fig.suptitle('Fig. 4 — Temporal embedding smoothness\n'
             'Left: correlation between embedding steps and physical displacement  '
             '·  Right: embedding step volatility',
             fontsize=10.5, fontweight='bold', ha='left', x=0.02, y=1.02)

fig.tight_layout()
fig.savefig('docs/plots/fig4_temporal_smoothness.png', dpi=160, bbox_inches='tight')
print('Fig 4 saved.')
