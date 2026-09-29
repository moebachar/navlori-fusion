#!/usr/bin/env python
# Register the CAD floor-plan image (Plan_RDC_walls-only_cropped.jpg) to the building/CAD frame (m)
# by fitting the golden runs' lidar walls onto the drawn walls (similarity: scale, rotation, shift, y flipped).
import json, numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
ROOT = "/mnt/x/side_navlori"; STG = ROOT + "/data/staging_golden"; OUT = ROOT + "/data/golden_floorplan"
F = 4                                                                   # work at 1/4 resolution
im = np.array(Image.open(f"{ROOT}/Plan_RDC_walls-only_cropped.jpg").convert("L"))
H0, W0 = im.shape; im = im[:H0 // F * F, :W0 // F * F]
small = im.reshape(H0 // F, F, W0 // F, F).min(axis=(1, 3))            # block-min keeps thin walls
wall = small < 160; D = ndimage.distance_transform_edt(~wall)          # px (1/4 res) to nearest wall
h, w = small.shape
pts = np.concatenate([np.load(f"{STG}/golden_run_{i}/ground_truth/map_points_world.npz")["xy"] for i in range(1, 13)])
k = np.floor(pts / 0.1).astype(np.int64); _, idx = np.unique(k, axis=0, return_index=True); P = pts[idx]
TR = 12.0                                                                # truncation (px) of the distance cost
def to_px(p, v):
    s, th, tx, ty = v; c, sn = np.cos(th), np.sin(th)
    return tx + s * (c * p[:, 0] - sn * p[:, 1]), ty - s * (sn * p[:, 0] + c * p[:, 1])
def cost(v):
    u, r = to_px(P, v); d = ndimage.map_coordinates(D, [r, u], order=1, mode="constant", cval=TR)
    return np.minimum(d, TR).mean()
# initial guess from robust extents (x spans ~the whole building width)
lo, hi = np.percentile(P, [0.5, 99.5], axis=0); cols = np.where(wall.any(0))[0]; rows = np.where(wall.any(1))[0]
s0 = (cols[-1] - cols[0]) / (hi[0] - lo[0])
best = None
for s in s0 * np.linspace(0.85, 1.15, 31):
    for dx in np.linspace(-3, 3, 25):
        tx = cols[0] - s * (lo[0] + dx)
        for dy in np.linspace(-3, 3, 25):
            ty = rows[-1] + s * (lo[1] + dy)
            c = cost([s, 0.0, tx, ty])
            if best is None or c < best[0]: best = (c, [s, 0.0, tx, ty])
print(f"grid best cost {best[0]:.2f} px", flush=True)
TR = 6.0; r1 = minimize(cost, best[1], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-5, maxiter=4000))
TR = 3.0; r2 = minimize(cost, r1.x, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-6, maxiter=4000))
s, th, tx, ty = r2.x
u, r = to_px(P, r2.x); d = ndimage.map_coordinates(D, [r, u], order=1, mode="constant", cval=99) / s   # metres
print(f"scale {s*F:.3f} px/m (full-res), rotation {np.degrees(th):.3f} deg, lidar points within 10 cm of a drawn wall: "
      f"{(d<0.10).mean()*100:.1f}%, within 20 cm: {(d<0.20).mean()*100:.1f}%, median {np.median(d)*100:.1f} cm", flush=True)
# full-res mapping:  px = TX + S*(cos*x - sin*y),  py = TY - S*(sin*x + cos*y)
S, TX, TY = s * F, tx * F, ty * F
json.dump(dict(image="Plan_RDC_walls-only_cropped.jpg", size_px=[W0, H0], px_per_m=S, rotation_rad=th, tx_px=TX, ty_px=TY,
               note="px = tx + s*(cos(r)*x - sin(r)*y); py = ty - s*(sin(r)*x + cos(r)*y); x,y in the CAD frame of tags_ground_truth.json",
               fit=dict(within_10cm_pct=float((d < 0.10).mean() * 100), within_20cm_pct=float((d < 0.20).mean() * 100), median_cm=float(np.median(d) * 100))),
          open(f"{OUT}/plan_transform.json", "w"), indent=1)
# overlay check: lidar walls (red) on the plan, with the CAD tags
fig, ax = plt.subplots(figsize=(22, 7.5)); ax.imshow(small, cmap="gray", vmin=0, vmax=255)
ax.scatter(u, r, s=0.15, c="red", alpha=0.5, rasterized=True)
tags = json.load(open(f"{ROOT}/data/tags_ground_truth.json"))
tp = np.array([[v["x"], v["y"]] for v in tags.values()]); tu, tr_ = to_px(tp, r2.x)
ax.plot(tu, tr_, "s", mfc="gold", mec="k", ms=7)
for k_, a, b in zip(tags, tu, tr_): ax.annotate(k_, (a, b), xytext=(3, 3), textcoords="offset points", fontsize=8)
ax.set_title(f"lidar walls (red) registered on the plan: {(d<0.10).mean()*100:.0f}% within 10 cm, median {np.median(d)*100:.1f} cm; "
             f"rotation {np.degrees(th):.2f} deg; gold = CAD tag coordinates"); ax.axis("off")
plt.tight_layout(); fig.savefig(f"{ROOT}/gt_review/plan_registration_check.png", dpi=90); print("PLAN_REG_DONE")
