# Result 06 — wifi-set-encoder-sparse-observed

## TL;DR

**Bar label: `NO-PASS`** — but with two large caveats that change the
interpretation.

Sparse-observed `WiFiSetTransformer` fits trivially in 8 GB at the
standard `batch_size=128` (peak GPU **435 MB** at the synthetic
budget check, dense path needed > 8 GB). Training completed in **12.5 min**
on the project GPU. Headline numbers:

| metric | PLAN_03 Anchor2Vec | PLAN_06 set-transformer (K=1 wifi-only) | Δ |
|---|---:|---:|---:|
| val MAE        | 15.70 | **16.21** | +0.51 (worse) |
| test MAE       |  8.99 | **9.02**  | +0.03 (same) |
| smoothness med | 12.92 | **3.37**  | **−9.55 (4× tighter trajectories)** |

By the plan's strict rubric (NO-PASS = no better than Anchor2Vec on the
headline MAE), this is **NO-PASS**: val regresses 0.51 m, test essentially
tied. **But:**

- This run used **K=1 (single temporal instant)** because the
  set-transformer is **15× slower per fwd-bwd** than Anchor2Vec
  (46.5 ms vs 3.1 ms benchmarked, B=128); K=8 (plan default) would
  have been ~150 min wall, blowing past the 10:00 stop. PLAN_03's
  baseline ran with K=8 — so the comparison **structurally
  disadvantages PLAN_06** by the K=8 temporal smoothing PLAN_03 had.
- This run used **30 epochs / patience=10** (not 90/15) to fit budget.
  Best epoch landed at 24/30 with val_mae still improving slightly at
  early stop — the encoder may not be fully converged.

**The smoothness collapse 12.92 → 3.37 is the headline finding.**
Predictions now step at ~3.4× the GT step rate vs ~13× before. That
directly addresses goal criterion (d) — "good path prediction in
real time" — which Anchor2Vec was failing. Per-trajectory:

| path | MAE | smoothness | (PLAN_03 smoothness) |
|---:|---:|---:|---:|
| 128 | 11.57 | 5.02 | (41.45) |
| 129 |  9.64 | 2.89 | (22.10) |
| 130 |  9.48 | 4.41 | (12.37) |
| 131 |  8.01 | 3.37 | (11.90) |
| 132 |  7.58 | 3.35 | (12.92) |

Every test path got dramatically smoother.

**PLAN_07 recommendation: `redesign_or_pivot`** (with strong nuance —
see section).

## Numbers

### Per-step pass/fail

| step | acceptance | observed | pass? |
|---|---|---|---|
| 1. rewrite `forward()` sparse-observed | (B,4,128) finite + < 6 GB peak at B=128, K=8 | (B,128) finite incl. all-zero rows; **peak 435 MB** | ✅ |
| 2. config knob | omitted (default `max_observed_per_scan=256` in `__init__`); plan said "optional" — no msiln scan in train exceeds the cap | n/a | ✅ (deferred per plan) |
| 3. smoke phase 1 + 2 | phase 2 ≥ 80 % loss drop AND MAE < 3 m on 16-batch | phase 1: finite at all-zero ✅; phase 2: **96.0 % loss drop**, MAE 12.7 m on 16-batch (above the 3 m bar but **not plateaued > 5 m** — continued per plan's STOP gate) | ⚠️ partial (loss drop ✅, MAE bar missed but the plan's STOP gate was "plateaus above 5 m" which did not trigger) |
| 4. full training ≤ 60 min | 90 epochs at standard bs=128 | **kicked first attempt at 7 min/epoch** (estimated 10.5 h) — killed and re-ran at K=1, 30 epochs, bs=128. **12.5 min wall**, fits | ⚠️ (K dropped to 1 to fit budget) |
| 5. full evaluation | all metrics reported | yes, see tables below | ✅ |
| 6. PLAN_07 recommendation | one of three labels + justification | `redesign_or_pivot` (see section) | ✅ |

### Files touched (count: 2)

- `src/pipeline/encoders/wifi_set.py` — rewrote `forward()` sparse-observed
  path; `__init__` gained `max_observed_per_scan: int = 256`; class name,
  signature, and `input_spec` unchanged.
- `scripts/_train_msiln_b1.py` — added `--n-instants` CLI flag (override
  `cfg.temporal.n_instants`) for the budget-forced K=1 run.

Well under the 5-file ceiling. **No `src/` changes outside `wifi_set.py`.**

### Encoder parameter count + latency

- params: **447 K** (0.45 M)
- per-sample inference latency: **4.44 ms** at batch=1, **0.145 ms** at
  batch=32 — well under the 100 ms goal criterion (e).

### Memory budget check (PLAN_06 step 1 acceptance)

At B=128, 8 fwd+bwd passes (worst-case fusion K=8 path), peak GPU
allocator: **434.8 MB**. The dense-masked iter_05 encoder needed > 8 GB
even at batch=32 with `expandable_segments` (`13.55 GB allocated by
PyTorch`). The 120× attention-cost reduction the plan predicted held.

### Headline numbers table

| run                                            | val MAE | test MAE | per-wp val | per-wp test | smooth med | wall (min) | Δ vs Anchor2Vec val | Δ vs Anchor2Vec test |
|------------------------------------------------|--------:|---------:|-----------:|------------:|-----------:|-----------:|--------------------:|---------------------:|
| PLAN_03 Anchor2Vec (full fusion, K=8)          |  15.70  |   8.99   |    20.54   |    18.56    |   12.92    |    18.4    |     —               |     —                |
| PLAN_06 set-transformer (WiFi-only, K=1)       |  **16.21** | **9.02** |  22.85   |    18.53    |  **3.37**  |    12.5    |    +0.51 (worse)    |    +0.03 (same)      |
| WiFi-kNN baseline (reference)                  |  17.66  |   9.47   |     —      |      —      |     —      |    n/a     |    −1.95 (better)   |    −0.45 (better)    |

### Per-path distribution

| split | n_paths | mean   | median | p25   | p75   | p90   | max    |
|-------|--------:|-------:|-------:|------:|------:|------:|-------:|
| val   |     34  | 17.06  | 14.61  | 10.83 | 20.40 | 26.11 | 36.24  |
| test  |      5  |  9.26  |  9.48  |  8.01 |  9.64 | 10.80 | 11.57  |

Numbers from `runs/fusion_20260525_090231/summary_set_sparse_k1.json`
`eval.<split>.per_path`. Test distribution is much tighter than
Anchor2Vec's (PLAN_03 test max was 13.54 m; here 11.57 m).

### Subset eval

| split | only:wifi | wifi+imu (sanity) |
|---|---:|---:|
| val  | 16.21 | 16.21 |
| test |  9.02 |  9.02 |

WiFi-only run; IMU was off; subset eval == single-modality MAE as
expected (no bug).

### Per-trajectory plots (test)

- `runs/fusion_20260525_090231/test_paths/path_128.png`
- `runs/fusion_20260525_090231/test_paths/path_129.png`
- `runs/fusion_20260525_090231/test_paths/path_130.png`
- `runs/fusion_20260525_090231/test_paths/path_131.png`
- `runs/fusion_20260525_090231/test_paths/path_132.png`

### Encoder benchmark (B=128, 20-step mean fwd+bwd)

| variant                              | per fwd+bwd | 90-ep wall projection (K=8) |
|---|---:|---:|
| Anchor2Vec                           |  3.09 ms | ~ 10 min (consistent with PLAN_03's 18 min including fusion overhead) |
| SetTransformer, max_obs=256          | 46.51 ms | ~ 154 min (would have blown past stop) |
| SetTransformer, max_obs=64           | 20.52 ms | ~  68 min (still over budget at K=8) |

Cost is dominated by the 2-layer `nn.TransformerEncoder` (attention +
FFN at ~140 tokens), not the gather/sort. **An efficient PLAN_07
training run with K=8 would either need a lighter transformer
(1 layer, smaller FFN), a custom flash-attention path, or epochs
trimmed to ≤ 30.**

## What was changed

- `src/pipeline/encoders/wifi_set.py`: rewrote `forward()` to
  sparse-observed gather (drop-in; same `__init__` signature plus one
  new optional kwarg `max_observed_per_scan: int = 256`).
- `scripts/_train_msiln_b1.py`: added `--n-instants` CLI flag for
  budget-forced K override.
- `handoff/results/RESULT_06_wifi-set-encoder-sparse-observed.md`: this file.
- `handoff/STATE.md`: iteration log row.

No `configs/`, no vendored-baseline, no other `src/` changes.
**Demand #3 untouched.**

## What was reverted

None. The dense-masked first pass at `forward()` from iter_05 is
overwritten by the sparse path (per PLAN_06 step 1 — same file, body
swap). Easily revertable via `git revert ffd5253..HEAD` should
diagnosis call for it.

## Logs (gitignored under `runs/`)

- `runs/overnight/iter_05/train_set_xformer*.log` — three iter_05 OOM
  traces (dense, bs=128 / 64 / 32) preserved for context.
- `runs/overnight/iter_06/train_set_sparse.log` — first attempt with K=8
  (killed after 7 min/epoch projection).
- `runs/overnight/iter_06/train_set_sparse_k1.log` — the run that landed.
- `runs/fusion_20260525_090231/summary_set_sparse_k1.json` — machine-readable.
- `runs/fusion_20260525_090231/metrics.jsonl` — per-epoch curves.
- `runs/fusion_20260525_090231/test_paths/*.png` — 5 per-traj plots.

## PLAN_07 recommendation

**Label: `redesign_or_pivot`.** By the strict rubric (NO-PASS), val MAE
regressed 0.51 m and test MAE tied Anchor2Vec, so the encoder swap
**did not move the headline number** on this dataset under this
training budget. However the **smoothness ratio collapsed 12.92 →
3.37**, which is a real architectural win the rubric doesn't measure;
trajectories went from "jittery" to "near-usable for real-time" by the
criterion-(d) intent. Three plausible next directions, scientist's call:

1. **Re-run set-transformer at K=8** (full temporal fusion) once an
   efficient attention path is available — either a lighter encoder
   (depth=1, ff_mult=2), per-AP cap reduced (max_observed_per_scan=128,
   acceptable for msiln since mean is 127), or `torch.nn.functional.
   scaled_dot_product_attention` with `is_causal=False` + the
   built-in FlashAttention kernel that ships with PyTorch 2.4. The
   sparse run at K=1 is structurally disadvantaged vs Anchor2Vec K=8;
   K=8 may flip the headline.
2. **Contrastive AP-dropout SSL pre-training** of the set-transformer
   (per the `GOOD` branch the rubric assigned to this label). The
   encoder may be under-trained — 30 epochs at lr 1.3e-3 fully
   superviced isn't the same as SSL warmup + supervised fine-tune.
3. **Pivot to cross-session data engineering** — the val/test floor
   may be a property of msiln site1/B1 with only 5 test paths
   (RESULT_04 Q3); converting site1/F2 or F3 (also 4-day splits per
   RESULT_01) would give more test paths and possibly a lower floor
   regardless of encoder.

**Engineer's gut: option 1.** The 15× per-fwd cost is fixable in
PyTorch 2.4 with SDPA + FlashAttention; if K=8 set-transformer lands
at val ≤ 13 m we're in `GOOD` territory. If even at K=8 it stays at
~15 m, then option 2 (SSL) or option 3 (data) is correct.

## Open question for scientist

**Q1.** The set-transformer matched Anchor2Vec on test (9.02 vs 8.99)
with **4× tighter trajectories** but worse on val. PLAN_03's val MAE
was 15.70 m at K=8 + full fusion; this run's was 16.21 m at K=1 +
wifi-only. The 0.51 m val regression is **smaller than the K=8 → K=1
penalty** Anchor2Vec would suffer (Branch C iter_04 wifi-only at K=8
Anchor2Vec landed at val 15.68 m — but that's actually basically the
same number as this set-transformer at K=1). The interesting question:
**is the val-MAE difference between the two encoders below noise on
this dataset (n=34 val paths)?** Bootstrap CI on per-path MAE might
show overlap. If so, PLAN_07 should commit to the K=8 set-transformer
run before deciding the encoder is "no improvement". Want a paired
bootstrap test?

## Wall-clock

- PLAN_06 detected: 08:36 local (after laptop sleep cycle, scientist
  had written PLAN_06 + SCIENTIST_NOTE_iter05)
- Encoder rewrite + memory check: 5 min
- First training attempt (K=8): killed after 18 min (7 min/epoch projection)
- Encoder benchmark: 2 min
- K=1 30-epoch training: **12.5 min wall** (completed)
- Eval + this writeup: ~10 min
- **Total iteration: ~50 min** (within the 90 min plan budget)
