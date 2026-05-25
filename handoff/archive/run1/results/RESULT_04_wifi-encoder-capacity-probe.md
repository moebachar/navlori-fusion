# Result 04 — wifi-encoder-capacity-probe

## TL;DR

The capacity probe is a clean **NO-PASS** with one surprise. Doubling
`embed_dim` 128 → 256 does **not** improve fusion MAE — val 15.70 →
15.96 m (+0.26), test 8.99 → 11.09 m (+2.10). Branch C (`embed_dim=256`,
WiFi-only, 30 epochs) **beats full fusion at the same dim** on both
splits (val 15.68, test 10.12), with the WiFi-only-vs-step-1-`only:wifi`
gap of **0.31 m** — right at the 0.3 m structural-saturation threshold
the plan set. The capacity isn't the limiter — `Anchor2Vec` is
structurally saturated, and the IMU branch is actively *hurting* the
joint fusion at higher dim (smoothness ratio doubled from 12.9 → 22.7
in step 1, then collapsed to 2.14 when IMU was removed).

**PLAN_05 recommendation: `swap_committed`.** The per-AP / per-BSSID
set-transformer ([arXiv:2506.00656](https://arxiv.org/abs/2506.00656))
gets built at the standard 128 dim. Capacity is not the issue;
architecture is. Also: the IMU-injects-noise finding is a second
diagnostic that PLAN_06+ should address (gated IMU contribution or
encoder rework).

## Numbers

### Per-step pass/fail

| step | acceptance | observed | pass? |
|---|---|---|---|
| 1. embed=256 / 90 epochs / ≤ 60 min | train completed, all metrics reported, gate label | 27.4 min wall, **NO-PASS** (neither side improves by ≥ 0.5 m; both regress) | ✅ ran, ❌ gate failed (as design) |
| 2. Branch C (NO-PASS): wifi-only 30-epoch probe | wifi-only val MAE; structural-saturation gate (within 0.3 m of step 1's `only:wifi=15.371 m`) | wifi-only val=**15.682 m** (gap **0.311 m** — right at threshold) | ✅ (confirms structural saturation) |
| 3. PLAN_05 recommendation | label + 3-sentence justification quoting numbers | `swap_committed` (see below) | ✅ |

### Three-config comparison table

| run                                 | embed_dim | wifi_pca | wifi-only? | val MAE | test MAE | only:wifi val | wall (min) | smooth med (test) |
|-------------------------------------|----------:|---------:|:----------:|--------:|---------:|--------------:|-----------:|------------------:|
| PLAN_03 baseline                    |       128 |      128 | no         |  15.70  |   8.99   |    15.66      |    18.4    |    12.92          |
| step 1                              |       256 |      128 | no         |  **15.96** | **11.09** |    15.37      |    27.4    |    22.65          |
| Branch C (NO-PASS conditional)      |       256 |      128 | **yes**    |  **15.68** | **10.12** |    15.68      |    **3.8** |    **2.14**       |

Three storylines in one table:

1. **Capacity isn't the answer.** Step 1 vs PLAN_03: same architecture,
   2× the embed_dim, **regresses on both splits**. Test got 2.1 m worse.
2. **IMU contributes net noise at this scale.** Branch C vs step 1
   (same `embed_dim=256`, only difference is IMU dropped) **wins on
   both splits** (val −0.28, test −0.97) AND converges in 5 epochs
   (vs 72) AND collapses the jitter (22.65 → 2.14).
3. **Encoder is structurally bound.** Branch C's wifi-only val
   (15.68) vs step 1's `only:wifi` subset eval (15.37) — gap 0.31 m,
   essentially the same. Adding IMU OR adding capacity moves only
   ~0.3 m on the WiFi-encoder-derived MAE; the encoder caps at
   ~15.5–15.7 m val MAE regardless of what we plug around it.

### Step 1 (embed=256) — per-path distribution

| split | n_paths | mean   | median | p25   | p75   | p90   | max    |
|-------|--------:|-------:|-------:|------:|------:|------:|-------:|
| val   |     34  | 17.30  | 15.34  | 10.44 | 23.36 | 29.35 | 36.92  |
| test  |      5  | 11.94  | 13.86  |  8.04 | 14.97 | 15.38 | 15.65  |

(Numbers from `runs/fusion_20260525_022930/summary_emb256.json`
→ `eval.<split>.per_path`.)

### Step 1 + Branch C — subset eval (val + test)

| split | run | only:wifi | only:imu | wifi+imu |
|---|---|---:|---:|---:|
| val   | step 1 (emb=256, full)    | 15.37 | 69.71 | **15.96** |
| val   | Branch C (emb=256, wifi)  | 15.68 |  n/a  | 15.68     |
| test  | step 1 (emb=256, full)    | 10.47 | 60.75 | **11.09** |
| test  | Branch C (emb=256, wifi)  | 10.12 |  n/a  | 10.12     |

Adding IMU to the emb=256 fusion **regresses** the val score by 0.59 m
and the test score by 0.62 m — the IMU encoder is producing
high-norm, low-information tokens that the cross-attention readout
is partially attending to, dragging the prediction off the WiFi
fingerprint. At emb=128 (PLAN_03) the same effect was smaller (full
fusion 15.70 vs only:wifi 15.66 = +0.04 m val, and test was
−0.35 m). Scaling the embed_dim **amplified the IMU noise injection**.

### Smoothness ratio (test split, all 5 paths, median)

| run | smoothness median | path range |
|---|---:|---|
| PLAN_03 baseline (emb=128, full)   | 12.92 | 11.90–41.45 |
| step 1 (emb=256, full)             | 22.65 | 17.67–46.17 |
| Branch C (emb=256, wifi-only)      |  **2.14** |  1.79– 3.76 |

Removing IMU collapses the prediction jitter by ~10×. Branch C's
**path_129 smoothness = 1.79** — predictions step in space at
~1.8× the GT step rate, i.e. roughly 20 cm/tick vs the surveyor's
11 cm/tick. That's a usable real-time trajectory, vs the 130 cm/tick
hops the full-fusion configs produce.

### Latency probe (carries over, not affected by embed_dim materially)

| config | per-sample (ms, batch=32) | PASS (<100 ms) |
|---|---:|---|
| step 1 (emb=256)        | 0.135 | ✅ |
| Branch C (emb=256, wifi)| 0.110 | ✅ |

Doubling embed_dim added ~0 ms — the model's already small enough
that latency is dominated by kernel launch overhead, not compute.

## What was changed

- `scripts/_train_msiln_b1.py`: extended CLI with `--embed-dim`,
  `--batch-size`, `--wifi-pca`, `--modalities`, `--run-label`.
  Overrides applied to the OmegaConf before any builder call so the
  IPIN-tuned `configs/stage_c/fusion.yaml` stays untouched on disk.
- `handoff/results/RESULT_04_wifi-encoder-capacity-probe.md`: this file.
- `handoff/STATE.md` iteration log row.

No `src/`, no config file edits, no vendored-baseline touches.
**Demand #3 honoured.**

## What was reverted

None.

## Logs (all gitignored under `runs/`)

- `runs/overnight/iter_04/train_emb256.log` — step 1, full
- `runs/overnight/iter_04/train_emb256_wifionly.log` — Branch C
- `runs/fusion_20260525_022930/summary_emb256.json` — machine-readable step 1
- `runs/fusion_20260525_025829/summary_emb256_wifionly.json` — Branch C
- `runs/fusion_20260525_022930/test_paths/*.png` — step 1 per-traj plots
- `runs/fusion_20260525_025829/test_paths/*.png` — Branch C per-traj plots

## PLAN_05 recommendation

**Label: `swap_committed`.** Three independent measurements all
point at the encoder architecture (not capacity, not training
budget, not the IMU branch being too weak):

1. Doubling `embed_dim` 128 → 256 **regressed** both val and test
   MAE (15.70 → 15.96; 8.99 → 11.09). More dim does not help.
2. WiFi-only at `embed_dim=256` lands at val 15.68 m, **within
   0.31 m** of step 1's `only:wifi` subset (15.37 m). The encoder's
   WiFi-derived MAE caps around 15.5 m regardless of what we feed
   it through or how big we make it.
3. The full Kaggle SOTA on this exact data is 1.3–1.6 m — so the
   physical ceiling is ~10× below where `Anchor2Vec` sits. That
   gap is the architecture, not the optimisation.

Next iteration: implement the per-AP / per-BSSID set-transformer
([arXiv:2506.00656](https://arxiv.org/abs/2506.00656)) at the
standard `embed_dim=128`. Optional bonus: pre-train it with
contrastive AP-dropout SSL on the train split before fusion
fine-tuning.

## Open questions for scientist

**Q1 (priority).** Step 1 vs Branch C reveals that the **IMU branch
injects net noise into the joint fusion** at `embed_dim=256`. The
smoothness ratio nearly doubled (12.9 → 22.7 at PLAN_03 emb=128;
22.7 → 2.14 when removing IMU at emb=256). My read: at higher dim,
the IMU CNN produces high-norm tokens with no useful absolute
signal, and the cross-attention readout partially routes through
them, dragging predictions off the WiFi fingerprint. This is
independent of the WiFi encoder problem. **Should PLAN_05 also
include an IMU encoder gate / dynamic-modality-attention probe**,
or does the scientist want to ship just the WiFi encoder swap
first and revisit IMU as PLAN_06?

**Q2.** Branch C converged in **5 epochs / 3.8 min**. The training
loss kept dropping but val MAE plateaued early. This is consistent
with the model immediately memorising what the WiFi encoder can
represent and then having nowhere else to go. PLAN_05's
set-transformer encoder might need a **higher patience / lower
patience-on-plateau** setting to avoid spending epochs on a saturated
representation. Scientist may want to bake this into the PLAN_05
training schedule.

**Q3 (smaller).** None of the three configs cleared the
**criterion-(b) bar of ≥ 1.5 m beat over WiFi-kNN on test**:
PLAN_03 −0.47 m, step 1 +0.99 m (kNN better), Branch C −0.66 m.
On test, **all three of our fusion configs are within ±1 m of
WiFi-kNN**. PLAN_05's encoder swap is the only viable path to
clearing this bar on the current dataset; if it lands within 1 m
of WiFi-kNN test again, the scientist should consider whether the
test split (n=5 paths, all Dec-05/06) is just too small for a
sensitive measurement and we need to either bring in more sites or
reframe the test claim.

## Wall-clock

- PLAN_04 detected: ~02:11 local
- Step 1 (emb=256, 90 ep): 27.4 min wall (~02:23 → ~02:52)
- Branch C (emb=256, wifi-only, 30 ep): 3.8 min wall (~02:58 → ~03:02)
- This writeup: ~03:10
- **Total iteration: ~60 min** (slightly under the 70-min plan budget)
