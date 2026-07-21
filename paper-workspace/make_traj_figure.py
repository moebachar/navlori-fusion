"""Generate a clean trajectory figure: ground-truth path + a half-length
predicted path (slightly deviated but matching), with a red graded glow dot
at the prediction tip (current predicted position).

No axes, grid, text, labels, or border. High resolution PNG (+ vector PDF).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --- ground-truth path: a smooth, natural indoor trajectory -----------------
t = np.linspace(0.0, 1.0, 600)
# a smooth sweeping walk (low harmonics -> no jagged kinks)
gx = 2.7 * t + 0.45 * np.sin(2.0 * np.pi * 0.85 * t)
gy = 1.25 * np.sin(2.0 * np.pi * 0.80 * t + 0.35) + 0.30 * np.cos(2 * np.pi * 1.4 * t)

# --- predicted path: first half of the route, slightly deviated -------------
half = t.size // 2
px_gt, py_gt = gx[:half], gy[:half]

# unit normal to the GT tangent, so the deviation is a clean side-offset
dx = np.gradient(px_gt)
dy = np.gradient(py_gt)
norm = np.hypot(dx, dy) + 1e-9
nx_hat, ny_hat = -dy / norm, dx / norm

# small, smoothly-growing offset that gently weaves to one side then back
s = px_gt.size
u = np.linspace(0.0, 1.0, s)
offset = (0.04 + 0.11 * u) * np.sin(2 * np.pi * 0.9 * u + 0.5)   # small magnitude
px = px_gt + offset * nx_hat
py = py_gt + offset * ny_hat

# --- figure -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))

# ground truth: soft grey, slightly thicker, sits behind
ax.plot(gx, gy, color="#9aa0a6", lw=4.5, solid_capstyle="round", zorder=1)

# prediction: deep blue, on top
ax.plot(px, py, color="#1a73e8", lw=3.2, solid_capstyle="round", zorder=3)

# --- red graded glow dot at the prediction tip ------------------------------
cx, cy = px[-1], py[-1]
# layered translucent circles -> radial gradient / glow
glow_radii = np.linspace(0.42, 0.045, 26)
glow_alphas = np.linspace(0.018, 0.30, 26)
for r, a in zip(glow_radii, glow_alphas):
    ax.add_patch(plt.Circle((cx, cy), r, color="#ea4335", alpha=a, lw=0, zorder=4))
# solid bright core
ax.add_patch(plt.Circle((cx, cy), 0.075, color="#d11607", alpha=1.0, lw=0, zorder=5))
ax.add_patch(plt.Circle((cx, cy), 0.030, color="#ff6f60", alpha=1.0, lw=0, zorder=6))

# --- strip everything -------------------------------------------------------
ax.set_aspect("equal", adjustable="datalim")
ax.set_axis_off()
ax.margins(0.04)
fig.patch.set_alpha(0.0)   # transparent figure background

out_png = "paper-workspace/traj_figure.png"
out_pdf = "paper-workspace/traj_figure.pdf"
fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.0, transparent=True)
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0, transparent=True)
print("wrote", out_png, "and", out_pdf, flush=True)
