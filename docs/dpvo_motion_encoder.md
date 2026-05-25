# DPVO Motion Encoder

A Stage-A vision encoder that estimates **camera motion** between two frames.
This document describes what was *actually built and measured* — not the
original aspiration.

---

## 1. What it is — and what it is not

**What it is.** A learned 2-D displacement regressor. It reuses **one frozen
component of DPVO** — the patch feature trunk `fnet` (a small CNN pretrained
on TartanAir) — and builds a custom correlation tracker + trainable head on
top. Given a pair of camera frames it produces a 128-D motion token; a small
head on that token regresses the world-frame displacement `(dx, dy)` the
camera underwent between the two frames.

**What it is not:**

- **It is not upstream DPVO.** DPVO is a full SLAM system — patch graph,
  recurrent update operator (GRU), differentiable bundle adjustment,
  keyframing. **None of that is used here.** Only DPVO's frozen `fnet`
  feature extractor is reused. Everything else (patch sampling, correlation,
  sub-pixel matching, pooling, the head) is our own code.
- **It is not rigorous visual odometry.** True VO estimates 6-DoF ego-motion
  from multi-view geometry. This encoder regresses a 2-D translation only —
  no rotation output, no geometric model.
- **It is not the DPVO "full pipeline" encoder.** That is a separate encoder
  (`DPVOFullEncoder`) which uses DPVO's *context* features (`imap`, 384-D)
  for absolute place recognition. This one uses the *matching* features
  (`fnet`, 128-D) for relative motion. Different feature branch, different
  task.

In one line: **DPVO-inspired, appearance-assisted 2-D motion regression** —
a frozen DPVO trunk + a correlation tracker + a trained displacement head.

---

## 2. How it works

Input: a frame pair `(B, 2, 3, H, W)`, ordered `[t-1, t]`, ImageNet-normalised
RGB. Output: a 128-D motion token (and, via the delta head, `(dx, dy)` in
metres).

```
frame t-1 ─┐                                  ┌─ frame t
           ▼                                  ▼
  ① preprocess: undo ImageNet norm, apply DPVO's 2x-0.5
           ▼                                  ▼
  ② FROZEN TRUNK  (DPVO fnet, BasicEncoder4, stride 4)
     (B,3,H,W) ─► (B,128,H/4,W/4) feature map         ░ no gradient ░
           ▼                                  ▼
      fmap_prev                          fmap_curr
           ▼                                  │
  ③ sample 64 patch descriptors on an 8×8 grid in frame t-1
           ▼                                  │
  ④ CORRELATION — cosine, not raw dot product:
       L2-normalise descriptors and fmap_curr ⇒ cosine ∈ [-1,1]
       corr (B,64,H,W)
           ▼
  ⑤ LOCAL SEARCH — mask corr outside ±32 feature cells of each
       patch's frame-t-1 location
           ▼
  ⑥ WINDOWED SOFT-ARGMAX:
       global hard-argmax ⇒ integer peak  (handles large motion)
       7×7 soft-argmax around it (temperature 20) ⇒ sub-pixel
       ⇒ matched coords + `sharp` (peak confidence)
           ▼
  ⑦ flow = coords_t − coords_{t-1}   (feature-grid pixels)
       per-patch token = [ fnet_feat(128) | dx | dy | ‖flow‖ | sharp ] = 132-D
           ▼
   ░░░ everything above is frozen / parameter-free → cacheable ░░░
           ▼
  ⑧ TRAINABLE HEAD  (_MotionHead):
       per-patch Linear 132→128 → attentive pool (1-query MHA) → MLP 128→256→128
           ▼
   128-D motion token   ── delta head: Linear 128→2 ──►  (dx, dy) metres
```

Steps ①–⑦ contain no learnable parameters (the trunk is frozen, the rest is
arithmetic), so the per-patch `(N, 64, 132)` tokens are cached once and only
the head (⑧ + delta layer) is trained.

### 2.1 The three fixes that made it work

The encoder existed before but its tracking was degenerate. Three concrete
problems were found and fixed:

| Problem | Symptom | Fix |
|---|---|---|
| Soft-argmax over the whole ~19k-cell correlation map | every patch collapsed to the map centroid → ~zero flow | **windowed** soft-argmax: hard-argmax for the integer peak, soft-argmax only in a 7×7 window |
| Raw dot-product correlation is energy-biased | peaks at bright regions, not true matches; `sharp` ≈ uniform (0.03) | **L2-normalise** → cosine similarity + softmax temperature |
| Global search is ill-posed in low-texture scenes | a uniform-floor patch matches everywhere; identity input gave `(-30,-9)` flow instead of 0 | **local search window** (±32 cells) around each patch's frame-t-1 location |

A synthetic-shift test confirms the fix: known image shifts are recovered
within 0.5 feature cell on the confident (high-`sharp`) patches.

---

## 3. Training

The encoder outputs a 128-D token; to supervise it we attach a `DeltaRegressor`
(the `_MotionHead` + a `Linear(128, 2)` layer) and train it to predict the
world-frame displacement.

- **Samples.** For each path, one sample per camera frame `t ≥ stride`:
  input = `(frame[t-stride], frame[t])`, target = `GT(t) − GT(t-stride)`,
  with GT `(x, y)` linearly interpolated from `ground_truth.csv` onto the
  camera timestamps. Default `stride = 5` (~1 s of motion).
- **Why world-frame `(dx, dy)`** and not ego-motion: a world-frame delta is
  re-derived per frame, so dead-reckoning has *no heading-integration drift*
  — only a translation random-walk. The target needs only GT positions, no
  heading. The cost: the head must infer heading from scene appearance, so
  the *trained head* is dataset-specific (the frozen trunk is not).
- **Procedure.** Cache frozen tokens once; train only the head. Targets are
  standardised (per-axis zero-mean/unit-std); Huber loss (δ=0.1), AdamW
  (lr 1e-3, wd 1e-4), OneCycleLR, 150 epochs, early stopping (patience 30).
  `DeltaRegressor.forward` returns un-standardised metres.

---

## 4. Inference — dead-reckoning

The online use-case: given **one** ground-truth position (at camera frame 0)
and nothing else, reconstruct the whole trajectory.

`deadreckon_path` walks the path's camera frames in **non-overlapping**
`stride` segments `[0, s], [s, 2s], …` so the predicted displacements tile
the path exactly:

```
p[0]   = GT position at frame 0          (the single anchor)
p[k·s] = p[(k-1)·s] + encoder_delta( frame[(k-1)·s], frame[k·s] )
```

No other sensor, no filtering. This is the raw motion-encoder output.

---

## 5. Measured results

All numbers are from `scripts/_smoke_dpvo_motion.py` on the Webots simulation
data (train paths `[1,3–12]`, val `[2,13,14]`, `stride = 5`).

**Per-pair displacement prediction (validation):**

| Metric | Value | Reading |
|---|---|---|
| Δ-MAE | 0.055 m | error on a ~0.30 m mean motion per pair |
| Δ-RMSE | 0.082 m | |
| bias | `[+0.005, −0.008]` m | effectively unbiased |
| scale | 0.997 | no shrinkage / regression-to-mean |
| direction error | 7.9° | motion *direction* well recovered |

**Dead-reckoning from a single GT anchor (validation paths):**

| Path | Path length | Mean error | Final drift | Drift % |
|---|---|---|---|---|
| path_02 | 25.7 m | 0.48 m | 0.74 m | 2.9 % |
| path_13 | 19.7 m | 0.12 m | 0.15 m | 0.8 % |
| path_14 | 25.1 m | 1.06 m | 1.85 m | 7.4 % |

Mean validation trajectory error ≈ **0.55 m**.

**Sanity checks.** Overfitting a single path drives Δ-MAE to 0.015 m and
dead-reckons that path to 0.07 m drift over 23.7 m — i.e. the architecture can
fit the task; the validation gap is generalisation, not a capacity ceiling.

---

## 6. Limitations and scope (honest)

- **Drift accumulates.** Dead-reckoning has unbounded error by construction.
  Observed drift is 1–7 % of path length. The per-segment error is unbiased,
  so most of the drift is the expected random-walk (`≈ √n · Δ-MAE`); the
  excess on some paths is *correlated* error during turns (the appearance-
  inferred heading lags). Removing that residual is the job of Stage C
  (fusion) and Stage D (KalmanNet), not of this encoder.
- **2-D translation only.** No rotation/heading is output. Heading is
  implicit in the head's appearance reasoning.
- **Appearance-assisted heading ⇒ dataset-specific head.** The frozen `fnet`
  trunk is scene-agnostic, but the trained head learns this environment's
  heading cues. It must be retrained per dataset. (This is true of every
  Stage-A encoder head in the project.)
- **Bounded detectable motion.** The ±32-cell search window caps detectable
  inter-frame motion at ~128 image px. Motion beyond that is clipped.
  `search_radius` and `stride` are the knobs.
- **Low-texture patches are ambiguous.** Patches on uniform floor/wall
  produce a low `sharp` value; the attentive pool is expected to discount
  them. ~7 % of patches sit near the search-window cap.
- **Camera-only.** Applies to datasets with camera frames in the
  `async_collection` format — i.e. the Webots simulation data. The external
  datasets (IMUWiFine, IPIN, RoNIN) have no camera and are out of scope.

---

## 7. File map

| File | Role |
|---|---|
| `src/pipeline/encoders/dpvo_motion.py` | `DPVOMotionEncoder` — frozen trunk + correlation tracker + `_MotionHead` |
| `src/pipeline/training/motion.py` | pair building, token caching, `DeltaRegressor`, training, dead-reckoning, profiling, `track_patches` |
| `scripts/_smoke_dpvo_motion.py` | phased dev/verification harness (synthetic shift → overfit → full train → profiling) |
| `runs/_weights/dpvo.pth` | DPVO release weights; only the `module.patchify.fnet.*` keys are loaded |
| `notebooks/encoder_workbench.ipynb` §12 | train + 3 visualisations (curves / dead-reckoned path / patch correspondence) |

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n_patches` | 64 | patches per frame, on an 8×8 grid |
| `search_radius` | 32 | local correlation window half-size (feature cells) |
| `stride` | 5 | frame-pair gap (~1 s of motion) |
| windowed-argmax radius | 3 | 7×7 sub-pixel window |
| softmax temperature | 20 | sharpens the cosine-similarity soft-argmax |

---

## 8. How to run

```bash
# Verification harness (each phase is independent):
python scripts/_smoke_dpvo_motion.py --phase 1   # synthetic-shift sanity
python scripts/_smoke_dpvo_motion.py --phase 3   # overfit one path
python scripts/_smoke_dpvo_motion.py --phase 4   # full train + profiling
```

In `notebooks/encoder_workbench.ipynb`, section 12 trains the head and renders
the training curves, a dead-reckoned trajectory, and the patch-correspondence
visualisation. Run cells 12.0 → 12.3 in order (restart the kernel first if the
encoder module was edited mid-session).
