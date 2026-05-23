"""
4 academic-style plots covering all evaluation metrics.
White background, muted ColorBrewer palette, no bar charts.
"""
import json, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba
from scipy.stats import norm
import numpy as np

# ── Academic palette (ColorBrewer) ─────────────────────────────────────────
colors = {
    'wifi':   '#2166AC',   # steel blue
    'imu':    '#D6604D',   # salmon red
    'odom':   '#4DAC26',   # forest green
    'camera': '#762A83',   # purple
}
mods   = ['wifi', 'imu', 'odom', 'camera']
labels = ['WiFi\n(Anchor2Vec)', 'IMU\n(IMUCNN)', 'Odom\n(OdomCNN)', 'Camera\n(VisionViT)']
short  = ['WiFi', 'IMU', 'Odom', 'Camera']

runs  = {m: json.load(open(glob.glob(f'runs/{m}_*/history.json')[0])) for m in mods}
evals = {m: json.load(open(glob.glob(f'runs/{m}_*/eval.json')[0]))   for m in mods}

# shared rcParams
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   '#333333',
    'axes.linewidth':   0.8,
    'xtick.color':      '#333333',
    'ytick.color':      '#333333',
    'text.color':       '#1a1a1a',
    'axes.labelcolor':  '#1a1a1a',
    'grid.color':       '#DDDDDD',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Learning Curves  (log-y, shaded confidence band via rolling window)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

for mod in mods:
    h   = runs[mod]
    y   = np.array(h['val_mae'])
    x   = np.arange(len(y))
    c   = colors[mod]
    lbl = short[mods.index(mod)]

    # smooth curve
    w = 5
    smooth = np.convolve(y, np.ones(w)/w, mode='same')
    ax.plot(x, y, color=c, alpha=0.18, linewidth=1.0)
    ax.plot(x, smooth, color=c, linewidth=2.0, label=lbl)

    # best epoch dot + annotation
    be = h['best_epoch']
    bv = h['best_val_mae']
    ax.scatter([be], [bv], color=c, s=60, zorder=6, edgecolors='white', linewidths=0.8)
    ax.annotate(f'{bv:.2f} m',
                xy=(be, bv), xytext=(be + 3, bv * 1.08),
                color=c, fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=c, lw=0.6))

ax.set_yscale('log')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Val MAE (m, log scale)', fontsize=11)
ax.set_title('Fig. 1 — Validation MAE Convergence', fontsize=12, fontweight='bold', loc='left')
ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=10, ncol=4,
          loc='upper right', handlelength=1.5)
ax.grid(True, axis='y', alpha=0.5)
ax.grid(True, axis='x', alpha=0.2)
fig.tight_layout()
fig.savefig('docs/plots/fig1_training_curves.png', dpi=160, bbox_inches='tight')
plt.close()
print('Fig 1 done')

# ══════════════════════════════════════════════════════════════════════════
# PLOT 2 — Dumbbell Chart: Best MAE ↔ KNN MAE ↔ LP MAE
# shows the "gap" between probing methods per encoder
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 4.5))

best_vals = [runs[m]['best_val_mae']        for m in mods]
knn_vals  = [evals[m]['knn_probe']['mae']   for m in mods]
lp_vals   = [evals[m]['linear_probe']['mae']for m in mods]

y_pos = np.arange(len(mods))
gap   = 0.22   # vertical spacing between sub-rows

for i, (m, b, k, lp) in enumerate(zip(mods, best_vals, knn_vals, lp_vals)):
    c = colors[m]

    # ── row: connect all three with a thin line ──────────────────────────
    mn_val, mx_val = min(b, k, lp), max(b, k, lp)
    ax.plot([mn_val, mx_val], [i, i], color='#aaaaaa', linewidth=1.0, zorder=1)

    # ── three markers ────────────────────────────────────────────────────
    ax.scatter([b],  [i], color=c,       marker='o', s=110, zorder=4,
               edgecolors='white', linewidths=0.8, label='Best MAE'  if i == 0 else '')
    ax.scatter([k],  [i], color=c,       marker='D', s=80,  zorder=4,
               edgecolors='white', linewidths=0.8, label='KNN MAE'   if i == 0 else '')
    ax.scatter([lp], [i], color='white', marker='s', s=90,  zorder=4,
               edgecolors=c, linewidths=1.5,           label='LP MAE'    if i == 0 else '')

    # ── value labels ─────────────────────────────────────────────────────
    for v, off in [(b, -0.22), (k, 0.22), (lp, 0.22)]:
        side = -1 if v == b else 1
        ax.text(v + side * 0.04, i + off, f'{v:.2f}', ha='center',
                va='bottom', fontsize=7.5, color=c, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('MAE (metres)', fontsize=11)
ax.set_title('Fig. 2 — MAE under three evaluation regimes per encoder\n'
             '● Best (trainable head)   ◆ KNN probe   □ Linear probe',
             fontsize=11, fontweight='bold', loc='left')
ax.axvline(0, color='#cccccc', linewidth=0.8)
ax.grid(True, axis='x', alpha=0.5)
leg = ax.legend(handles=[
    mlines.Line2D([], [], color='#555555', marker='o', markersize=8,
                  markerfacecolor='#555555', markeredgecolor='white', label='Best MAE (trainable head)', linewidth=0),
    mlines.Line2D([], [], color='#555555', marker='D', markersize=7,
                  markerfacecolor='#555555', markeredgecolor='white', label='KNN probe (k=5)',           linewidth=0),
    mlines.Line2D([], [], color='#555555', marker='s', markersize=8,
                  markerfacecolor='white',   markeredgecolor='#555555', label='Linear probe (frozen z)', linewidth=0),
], frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=9, loc='lower right')
fig.tight_layout()
fig.savefig('docs/plots/fig2_mae_dumbbell.png', dpi=160, bbox_inches='tight')
plt.close()
print('Fig 2 done')

# ══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Bubble chart: Alignment × Uniformity, bubble = Eff.Dim (PR)
# combines 3 metrics into one 2D space
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 5.5))

align  = [evals[m]['alignment_uniformity']['alignment']              for m in mods]
unif   = [evals[m]['alignment_uniformity']['uniformity']             for m in mods]
pr     = [evals[m]['effective_dimensionality']['participation_ratio']for m in mods]

# bubble size proportional to PR
sizes = [p * 120 for p in pr]

# ideal zone annotation (low alignment, low/negative uniformity)
ax.add_patch(mpatches.FancyBboxPatch((-0.05, -2.1), 0.22, 0.75,
             boxstyle='round,pad=0.02', facecolor='#EBF5EB',
             edgecolor='#4DAC26', linewidth=0.8, zorder=0, alpha=0.7))
ax.text(0.0, -1.42, 'ideal region', color='#2E7D32',
        fontsize=8, style='italic', va='bottom')

for m, a, u, s in zip(mods, align, unif, sizes):
    c = colors[m]
    ax.scatter([a], [u], s=s, color=c, alpha=0.82, zorder=4,
               edgecolors='white', linewidths=1.0)
    ax.scatter([a], [u], s=s * 2.5, color=c, alpha=0.12, zorder=3)
    ax.annotate(short[mods.index(m)], xy=(a, u),
                xytext=(a + 0.015, u - 0.06),
                fontsize=10, fontweight='bold', color=c)

# bubble size legend
for pr_ref, lbl in [(2, 'PR=2'), (4, 'PR=4'), (6, 'PR=6')]:
    ax.scatter([], [], s=pr_ref * 120, color='#999999', alpha=0.6,
               edgecolors='white', linewidths=0.8, label=lbl)

ax.set_xlabel('Alignment  ↓  (lower = spatially consistent)', fontsize=11)
ax.set_ylabel('Uniformity  ↓  (more negative = evenly spread)', fontsize=11)
ax.set_title('Fig. 3 — Embedding space quality\n'
             'Alignment × Uniformity  (bubble area ∝ Participation Ratio)',
             fontsize=11, fontweight='bold', loc='left')
ax.legend(title='Eff. Dim (PR)', frameon=True, framealpha=0.9,
          edgecolor='#cccccc', fontsize=9, loc='upper right')
ax.grid(True, alpha=0.4)
ax.axvline(0, color='#cccccc', linewidth=0.6)
fig.tight_layout()
fig.savefig('docs/plots/fig3_embedding_quality.png', dpi=160, bbox_inches='tight')
plt.close()
print('Fig 3 done')

# ══════════════════════════════════════════════════════════════════════════
# PLOT 4 — Embedding displacement distributions + temporal r annotation
# Shows mean_dz ± std_dz as Gaussian curves per encoder
# Creative: you see which encoder is "jittery" vs "stable"
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

x_range = np.linspace(-1.5, 4.5, 400)

for m in mods:
    ts  = evals[m]['temporal_smoothness']
    mu  = ts['mean_dz']
    sig = ts['std_dz']
    r   = ts['correlation']
    c   = colors[m]

    pdf = norm.pdf(x_range, mu, sig)
    ax.plot(x_range, pdf, color=c, linewidth=2.2, label=f'{short[mods.index(m)]}  (r={r:.3f})')
    ax.fill_between(x_range, pdf, alpha=0.10, color=c)

    # mean tick
    ax.axvline(mu, color=c, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(mu + 0.04, norm.pdf(mu, mu, sig) * 0.55,
            f'μ={mu:.2f}', color=c, fontsize=8, style='italic')

ax.axvline(0, color='#333333', linewidth=0.8)
ax.set_xlabel('||Δz||₂  — embedding step magnitude', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Fig. 4 — Temporal embedding dynamics\n'
             'Distribution of per-step embedding displacements  (r = Pearson corr. with physical Δy)',
             fontsize=11, fontweight='bold', loc='left')
ax.set_xlim(-0.5, 4.0)
ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=10,
          title='Encoder  (temporal r ↑)', title_fontsize=9)
ax.grid(True, axis='y', alpha=0.4)
fig.tight_layout()
fig.savefig('docs/plots/fig4_temporal_distributions.png', dpi=160, bbox_inches='tight')
plt.close()
print('Fig 4 done')
print('All done — 4 academic plots saved to docs/plots/')
