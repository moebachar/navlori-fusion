# NavLoRI-Fusion — Pipeline Autopsy

A ground-up forensic inspection of the full pipeline (data → x,y), doubting every choice. No building, no patching — only probing, profiling, and analysis. Started 2026-05-20.

Probe scripts live in `scripts/inspect_*.py`. Each section: **what we doubted**, **what we measured**, **what it means**.

---

## Probe 1 — Raw data integrity

**Doubted:** that the data feeding the pipeline is even well-formed (GT units/frame, WiFi density, NaN handling, timestamps, rates).

**Measured** (`scripts/inspect_01_rawdata.py`, per dataset/split):

| dataset | GT extent | GT rate | GT step med/max (cm) | WiFi rate | WiFi NaN/scan | APs vis/scan | IMU rate |
|---|---|---|---|---|---|---|---|
| simulation | 13×18 m | 10 Hz | 3.4 / 5.8 | **1.0 Hz** | **0%** | **117/117** | 31 Hz |
| ipin2024_floor-2 | **98×47 m** | 10 Hz | 8 / 11 | **0.15–0.25 Hz** | **87–90%** | ~20/166 | 25 Hz |
| ronin_a000 | 50×84 m | 10 Hz | 10 / 26 | **0.12–0.16 Hz** | 86–89% | ~20/144 | 33 Hz |
| imuwifine train/val | 85×6 m | **1.7 Hz** | 27 / **5046** | 0.3 Hz | 91% | ~33/343 | 27 Hz |
| imuwifine **test** | 81×**0.4** m | 6.4 Hz | 6 / 136 | 6.2 Hz | 85% | **0 (no IMU)** | — |

**What it means — five findings, two of them severe:**

### 1.1 (SEVERE) WiFi is 5–50× sparser than the pipeline assumes
Real WiFi updates every **4–8 seconds** (0.12–0.25 Hz), not the ~1 Hz the dataset/CLAUDE.md assume. GT is 10 Hz. With `windows[wifi]=1` and "carry forward the most recent scan," **one WiFi scan is reused for 40–80 consecutive GT samples**. At ~0.85 m/s that span covers **3–6 meters**. So the WiFi encoder is asked to map a *single fixed scan* to a cloud of 40–80 different positions several meters wide — the best it can do is predict that cloud's centroid. **This is a multi-meter accuracy floor baked into the data handling, independent of model or transferability.** Needs quantification (Probe 5).

### 1.2 (SEVERE) IMUWiFine data is partly corrupt / structurally inconsistent
- train/val GT has **50-meter single-tick teleports** (`max step 5046 cm`, `4607 cm`) — impossible for a walking human; corrupt samples or broken interpolation.
- train/val GT rate is **1.7 Hz**, not the claimed 10 Hz.
- the **test split is a different beast**: a 1-D line (y ∈ [1.2,1.6], 0.4 m "extent"), 6.4 Hz, **no IMU at all**, WiFi at 6 Hz. Train and test are not the same data format.
- ⇒ **Every IMUWiFine number we've reported (incl. the 3.77 m WiFi-kNN "baseline") is computed on corrupt / mismatched data and is meaningless.** Drop IMUWiFine until reconverted, or treat its numbers as void.

### 1.3 Sim WiFi is physically unlike real WiFi
117/117 APs visible every scan, 0% missing, 1 Hz, smooth. Real WiFi is ~12% visible, 87% missing, 0.15 Hz. The sim→real gap is enormous **at the data level** — a model that works on sim WiFi has learned something that cannot exist in real WiFi. Sim results are not predictive of anything real.

### 1.4 Floor scales differ 8× — "MAE in meters" is not comparable across datasets
sim 13×18 m, IMUWiFine 85×6 m, RoNIN 50×84 m, IPIN 98×47 m. A 25 m error on IPIN's 100 m floor (~25% of extent) and a 0.4 m error on sim's 13 m floor (~3%) are different universes. All cross-dataset MAE comparisons so far have been apples-to-oranges. Need normalized error (Probe 4).

### 1.5 (clean) GT timestamps are monotonic, no NaN/Inf in IMU, yaw in degrees (absmax 180)
The basic plumbing (monotonic time, finite IMU) is sound on sim/IPIN/RoNIN. The rot is in WiFi density and IMUWiFine GT, not in basic file integrity.

**Immediate consequence for the plan:** the WiFi-staleness floor (1.1) is now the prime suspect for the IPIN ~23 m wall. Quantify it next.

---

## Probe 2 — The WiFi-staleness floor (quantified)

**Doubted:** that carry-forward of a sparse WiFi scan imposes a large irreducible error.

**Measured** (`scripts/inspect_02_wifi_staleness.py`): for each GT sample, staleness = `t − t_lastscan`, lag_disp = `‖gt(t) − gt(t_lastscan)‖`, and a "centroid floor" = MAE if every sample is assigned the GT centroid of all samples sharing its scan (the best a WiFi-only model can do under carry-forward).

| dataset/split | staleness med / p90 / max (s) | samples/scan med / max | lag_disp mean (m) | **centroid floor (m)** |
|---|---|---|---|---|
| sim val | 0.48 / 0.9 / 1.2 | 10 / 13 | 0.16 | **0.08** |
| ipin val | 3.08 / **68** / **150** | 40 / **1501** | 7.09 | **4.18** |
| ronin train | 5.88 / 99 / **275** | 43 / **2748** | 8.12 | **5.21** |
| ronin val | 3.59 / 7.8 / 12.9 | 63 / 129 | 3.87 | **1.95** |

**What it means:**

### 2.1 The staleness floor is real but SECONDARY (~4 m of IPIN's 23 m)
IPIN val's best-possible WiFi-only error under carry-forward is **4.18 m**. The model gets 23 m and kNN gets 32 m — **5–8× worse than the floor**. So sparsity/staleness costs ~4 m; it is NOT the dominant problem. My "prime suspect" was wrong: staleness is a contributor, not the wall. (Corrected hypothesis logged.)

### 2.2 The dominant problem is transferability, and this probe isolates it
- centroid-floor 4.18 m = "perfectly recognize each scan, predict its window centroid" (an oracle that needs fingerprints to transfer).
- kNN 32 m = what you actually get.
- **The 4 m → 32 m gap is pure fingerprint-transfer failure** (train scans don't match val scans). Quantify directly in Probe 3.

### 2.3 (SEVERE, actionable) Carry-forward has no staleness cap → corrupts training
`samples/scan max = 1501` (IPIN) and `2748` (RoNIN): a single WiFi scan is fed, unchanged and marked "available", to up to **2748 GT samples spanning 150–275 seconds**. During training the WiFi encoder sees one fixed input paired with thousands of wildly different positions across the whole floor → it's taught to ignore WiFi (can't fit) or regress to a centroid. A 150-second-old scan is fed as if current; nothing masks it. **There is no staleness cutoff anywhere in the pipeline.** During those long WiFi-gap stretches the only live signal is motion, which can't anchor → position is unobservable → these regions inflate the mean error and are exactly where the earlier attribution showed 72 m blow-ups.

### 2.4 Train/val WiFi coverage is asymmetric
IPIN train staleness median 2 s vs val 3 s with val p90 jumping to 68 s; the val paths have materially worse WiFi coverage than train. Any "the model overfits" story is entangled with "val is a harder WiFi regime than train."

---

## Probe 3 — WiFi transferability + split geometry (THE pivotal probe)

**Doubted:** the conclusion (reached twice in prior work) that "WiFi fingerprints don't transfer across trials, so localization is impossible." Tested it directly with PROPER fingerprint matching — RSSI distance over **co-visible APs only** (no -100 fill), val scan → nearest train scan, then the spatial distance between them.

**Measured** (`scripts/inspect_03_transfer.py`):

| dataset | intra-train NN (med/mean m) | train→val NN (med/mean m) | random (m) | **transfer skill** | split overlap |
|---|---|---|---|---|---|
| simulation | 0.53 / 0.92 | 0.36 / 0.49 | 6.8 | **93%** | 100% |
| ipin2024_floor-2 | 3.51 / 8.97 | **4.23** / 15.89 | 38.9 | **59%** | 85% |
| ronin_a000 | 27.4 / 27.5 | 35.9 / 36.5 | 24.3 | **−50%** | 70% |

**What it means — this rewrites the diagnosis:**

### 3.1 (PIVOTAL) WiFi DOES transfer on IPIN — the prior conclusion was an artifact
With proper co-visible-AP matching, a val scan's nearest train scan is **4.23 m away (median)** — 59% better than random. WiFi genuinely encodes transferable position on IPIN. **The "WiFi doesn't transfer / position unobservable" story was wrong.** It came from measuring the pipeline's *destroyed* features, not the WiFi signal itself.

### 3.2 (DOMINANT, FIXABLE) The pipeline's WiFi feature processing destroys the signal
- Proper co-visible fingerprint kNN: **~4 m median** on IPIN val.
- The pipeline's WiFi-kNN baseline (`-100`-fill → PCA-128 → z-norm): **32 m**.
- Same data, 8× worse. The `-100` fill makes Euclidean distance dominated by *which APs are missing* (87% of entries) rather than the RSSI of the ~20 co-visible APs; PCA then captures the missingness pattern, not the fingerprint. **The feature engineering throws away most of the positioning signal before any model sees it.** This is the dominant, *fixable* problem on IPIN — and it equally starves the Anchor2Vec encoder, which eats the same `-100`-filled-PCA input.

### 3.3 Split geometry is fine — not an extrapolation problem
85% (IPIN) / 100% (sim) of val positions sit in 3 m cells train also visited; bounding boxes coincide. Val is not asking the model to extrapolate to unseen regions. The problem is signal processing, not coverage.

### 3.4 RoNIN a000 is a non-viable WiFi dataset — stop drawing conclusions from it
Transfer skill **−50%** (worse than random); intra-train NN error 27 m on an 84 m floor. Only 47 val scans, single subject, ultra-sparse. Its "13 m WiFi-kNN baseline" is essentially the centroid. RoNIN a000 cannot support WiFi localization; any result on it (incl. the earlier 1.87 m leaked number) says nothing about the method.

### 3.5 Revised problem hierarchy on IPIN (the honest benchmark)
1. **WiFi feature destruction (-100 fill + PCA)** — turns a 4 m signal into 32 m. **Biggest, fixable.**
2. **Staleness/carry-forward floor** — ~4 m, plus uncapped 150 s stale tokens corrupting training (Probe 2.3).
3. **Heavy tail** — even proper matching has mean 16 m vs median 4 m: ambiguous scans (few co-visible APs) need handling.
The earlier "it's an observability wall, fusion is fine, give up on WiFi" conclusion is **retracted**. The WiFi pipeline is recoverable.

---

## Probe 4 — Which WiFi encoding step destroys the signal? (the headline)

**Doubted:** the WiFi feature transform (`-100` fill → PCA-128 → z-score). Ran scan-level train→val kNN (k=5, distance-weighted) under each encoding of the *same* scans.

**Measured** (`scripts/inspect_04_wifi_encoding.py`):

| encoding | sim val MAE | ipin val MAE |
|---|---|---|
| A. co-visible-AP fingerprint | 0.50 | 14.04 |
| B. `-100` fill, raw Euclidean | 0.50 | **5.38** |
| C. `-100` fill + per-AP z-score | 0.57 | 8.12 |
| E. `-100` fill + PCA (no z-score) | 0.50 | **5.37** |
| **D. `-100` fill + PCA + z-score (PIPELINE)** | **3.99** | **20.90** |

**What it means:**

### 4.1 (THE FINDING) Post-PCA z-scoring (whitening) destroys the WiFi signal
- PCA *without* z-score (E) = 5.37 m on IPIN, identical to raw (B). A full-rank rotation preserves the metric — PCA is harmless.
- PCA *with* z-score (D, the pipeline) = 20.90 m. **3.9× worse on IPIN, 8× worse on sim, caused entirely by the z-score after PCA.**
- Z-scoring each PCA component to unit variance is **whitening**: it blows the tiny-eigenvalue (noise) directions up to the same scale as the dominant location-signal directions, so every distance the kNN / Anchor2Vec encoder computes is dominated by noise.
- Confirmed in code: `dataset.py:_compute_stats` computes WiFi mean/std *after* `_apply_wifi_pca`, and `_get_window` does `(pca(x) − mean)/std`. One line of well-intentioned normalization is the single biggest accuracy sink in the pipeline.

### 4.2 This retroactively explains almost every prior mystery
- WiFi-kNN baseline at 32 m (it ate whitened features) → would be ~5 m raw.
- Anchor2Vec "couldn't learn good WiFi" → it was fed whitened, noise-dominated input.
- Fusion "leaned on WiFi but landed near the centroid" → the WiFi token was mostly noise.
- "WiFi doesn't transfer / observability wall / give up on WiFi" → an **artifact of whitening**, not a property of the data.
The decomposed-readout failure, the conformal 55 m radius, the IPIN 23 m wall — all are downstream of a starved WiFi feature.

### 4.3 Best simple WiFi encoding found
`-100` fill, **no PCA, no z-score** (or PCA without z-score) → **5.4 m** scan-level on IPIN val. A learned per-AP/BSSID embedding could do better, but even this trivial change is a ~4× WiFi improvement waiting to be claimed.

---

## Probe 5 — Motion-signal quality + spatial scale

**Doubted:** that the *motion* leg (IMU) carries usable displacement signal on real data, and that "MAE in meters" means anything without floor scale.

**Measured** (`scripts/inspect_05_motion_scale.py`): kNN(IMU window → 1 s GT displacement), val displacement MAE vs "predict zero motion"; plus floor extent and the "predict global centroid" do-nothing floor.

| dataset | floor | centroid floor (m) | IMU disp MAE | zero-motion MAE | **motion skill** |
|---|---|---|---|---|---|
| simulation | 14×18 | 5.15 | 0.050 | 0.341 | **85% (usable)** |
| ipin2024_floor-2 | 98×47 | 25.66 | 0.737 | 0.840 | **12% (WEAK)** |
| ronin_a000 | 50×84 | 16.96 | 1.027 | 0.971 | **−6% (NONE)** |

**What it means:**

### 5.1 The motion leg is weak-to-useless on real data with the current IMU handling
Sim IMU predicts its 1 s displacement at 85% skill (clean synthetic). On IPIN it's 12% — barely beats predicting "didn't move." On RoNIN it's **negative** — a kNN on the normalized body-frame IMU window cannot recover displacement at all (RoNIN is *the* hard inertial benchmark; raw IMU→displacement needs a proper heading-aware model, not a CNN on z-scored windows). So even after WiFi is fixed, motion can bridge WiFi-gaps only weakly on IPIN and not at all on RoNIN as currently encoded. The body-frame/world-frame heading rotation is not being handled.

### 5.2 Scale context — what the MAE numbers actually mean
- sim: model 0.41 m on a 14 m floor with 5.15 m centroid floor → **genuinely localizing** (but trivially, dense clean WiFi).
- IPIN: model 23 m vs 25.7 m centroid → **barely better than do-nothing** — *because* WiFi is whitened to noise. A kNN on un-whitened WiFi gets 5.4 m, far below the centroid floor. The headroom is real and large.
- RoNIN: 13 m ≈ 17 m centroid → not localizing; WiFi non-viable + motion none. Dead dataset.

### 5.3 System now fully characterized: two legs, both currently broken on real data
1. **Absolute (WiFi):** informative (5.4 m) but **whitened into noise** (20.9 m) — Probe 4. Fixable now.
2. **Relative (motion):** **weak (IPIN 12%) / none (RoNIN)** with naive z-scored body-frame IMU — needs a real inertial model + heading handling. Secondary, harder.

---

## Probe 6 — What is the trained model actually doing?

**Doubted:** that the 23 m model is just predicting the centroid.

**Measured** (`scripts/inspect_06_model_behavior.py`, the query bake-off model on IPIN val):
- val MAE 23.0 m; GT std (28, 12) m, PRED std (23, 9) m → **only 19% spread shrinkage** (not collapsed).
- pred↔GT correlation x=0.47, y=0.38 → **it tracks GT weakly, not ignoring it.**
- per-path val MAE: path_11 **28.1**, path_12 17.1, path_13 **14.1**, path_14 **32.0** m.

**What it means:**
- **Not centroid collapse.** The model extracts a weak-but-real signal (corr ≈ 0.4) and spreads predictions across the floor. It's doing the best it can with **whitened-noise WiFi features** — which is exactly ~23 m. This confirms Probe 4: the ceiling is set by the destroyed WiFi feature, not by the model giving up.
- **Aggregate MAE hides 2.3× per-path variance** (14→32 m). Reporting a single val_mae masks that some trajectories (WiFi-gap-heavy) are far worse. All future evals should report per-path distributions, not just the mean.

---

# SYNTHESIS — what the f*** is going on, and what to do

## The one-paragraph truth

The pipeline is not failing because fusion is wrong, because position is unobservable, or because WiFi doesn't transfer — **all three were red herrings we chased.** It is failing because of a **single line of feature processing**: WiFi is PCA'd and then **z-scored per component (whitened)**, which amplifies noise directions to the scale of the signal and turns a genuinely informative ~5 m WiFi fingerprint into ~21 m of noise. Everything downstream — the 23 m IPIN wall, the WiFi-kNN baseline losing to the centroid, Anchor2Vec "not learning," fusion "leaning on WiFi but landing near the centroid," the decomposed-readout failure, the 55 m conformal radius — is a **consequence of starving every WiFi-consuming component with whitened input.** On top of that sit three secondary issues: WiFi carry-forward has no staleness cap (one scan reused for up to 2748 samples / 275 s), the motion leg is weakly encoded (12% skill on IPIN, none on RoNIN), and two datasets are unusable (IMUWiFine GT is corrupt; RoNIN a000 WiFi is non-viable).

## Problem hierarchy (measured magnitudes, IPIN honest split)

| # | Problem | Evidence | Cost | Fix difficulty |
|---|---|---|---|---|
| 1 | **WiFi whitening (PCA + per-component z-score)** | Probe 4: 5.4 m → 20.9 m | **~15 m** | trivial (delete a normalization) |
| 2 | **Carry-forward, no staleness cap** | Probe 2: scan reused ≤2748 samples / 275 s; ~4 m floor + corrupts training | ~4 m + training noise | easy (mask stale tokens) |
| 3 | **Weak motion encoding** (body-frame, z-scored, no heading) | Probe 5: 12% skill IPIN, −6% RoNIN | caps the relative leg | hard (real inertial model) |
| 4 | **Ambiguous-scan tail** (few co-visible APs) | Probe 3: median 4 m, mean 16 m | tail inflation | medium |
| 5 | **Corrupt / non-viable datasets** | Probe 1/3: IMUWiFine 50 m GT jumps; RoNIN −50% transfer | invalidates their numbers | data work |

## What to STOP doing (red herrings, now retired)

- ❌ "Position is unobservable / WiFi doesn't transfer" — **false** (59% transfer skill, 4 m median NN). Artifact of whitening.
- ❌ Tuning the fusion readout (query vs decomposed vs cls) — the readout is not the bottleneck; it's fed noise. The decomposed-readout negative result is fully explained by this.
- ❌ Trusting any IMUWiFine number or any RoNIN a000 number.
- ❌ Comparing raw MAE across datasets without normalizing by floor scale.
- ❌ Reporting a single val_mae — it hides 2.3× per-path variance.

## What to DO — ranked by impact / effort (NOT executed; this is the plan)

1. **Kill WiFi whitening (highest impact, lowest effort).** Stop z-scoring WiFi after PCA. Options in order of preference: (a) drop PCA entirely, feed `-100`-filled raw RSSI with at most a single global scale; (b) PCA without per-component z-score; (c) better, a **per-AP/BSSID learned embedding with masked pooling** that never sees a `-100` and handles missingness as masking, not a value. Expected: IPIN WiFi-kNN ~32 m → ~5 m; fused val_mae should follow down hard. *This is the experiment to run first, and the baseline + attribution harness will measure it.*
2. **Add a WiFi staleness cap.** Mark a carried-forward scan `unavailable` once it exceeds e.g. 5–10 s, so the model stops treating a 275 s-old scan as a live fix and stops being trained on one-input-many-targets. Cheap, removes a training-corruption source and the worst tail.
3. **Re-baseline honestly** with the fixed WiFi encoding, per-path distributions, on IPIN floor −2 and floor 0 (the only honest viable splits). Drop IMUWiFine and RoNIN a000 until repaired/replaced.
4. **Then, and only then, revisit motion.** With WiFi anchoring working, a proper heading-aware inertial encoder (rotate body→world via orientation, or a RoNIN-style ResNet on the IMU stream) can bridge WiFi gaps. Until WiFi is fixed this is wasted effort (it can't anchor anything).
5. **Handle the ambiguous-scan tail** (few co-visible APs): down-weight or mask scans below an AP-count threshold; this attacks the median-4 m / mean-16 m gap.

## Bottom line

We have spent the project's effort on the fusion transformer — the one part that was basically fine. The meters were hiding in the **data plumbing**: one whitening line, one missing staleness cap, and a naive motion encoder, measured on two broken datasets. Fix the WiFi feature path first; expect the IPIN wall to fall from ~23 m toward the ~5 m the signal actually supports, *before* touching the model at all.

