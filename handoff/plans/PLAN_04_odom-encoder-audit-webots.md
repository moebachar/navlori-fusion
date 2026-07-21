# Plan 04 — Odom encoder audit (`OdomCNN` on Webots sim, internal)

> **Amended-rubric continuation.** Same rules as PLAN_03 (multi-
> condition validation via val/test paths, preprocessing as a
> first-class probe, raw weighted ≥ aligned). One difference from
> PLAN_01–03: Odom has **no public SOTA**, so the "Day-1 SOTA
> reproduction" rule is satisfied by a **trivial-integration
> baseline** (dead-reckoning cumulative integration of odom (v, ω) →
> position) computed on the same data. That is the floor `OdomCNN`
> must beat to justify its existence; CLAUDE.md's pre-run-1
> "Linear Probe MAE ~3.5 m" reference is the second yardstick.

## Hypothesis

`OdomCNN` (1D-CNN, 16-step ≈ 1 s window over 7 odometry features,
channels=(16, 32, 64), embed_dim=128 — small ~0.04 M-param model) is
the lowest-stake encoder in the 4-modality stack: odometry is
locally accurate and globally drifting (no absolute anchor). The audit
question is **not** "is OdomCNN SOTA?" but "does OdomCNN's
*embedding* outperform the trivial dead-reckoning integration as a
fusion-input?" Two outcomes are equally interesting:

- **(α) OdomCNN beats integration on linear-probe MAE.** Means the
  CNN's window features carry information beyond cumulative
  (v, ω) — useful for fusion. Label `keep`.
- **(β) OdomCNN does not beat integration.** Means a learned encoder
  is unjustified for this modality; fusion can consume raw integrated
  odom directly. Label `replace` (with the trivial integration as a
  "non-encoder feature path" in the fusion design).

There's also a `modify` zone: OdomCNN matches integration on raw MAE
but offers cleaner geometry (lower alignment, higher trustworthiness)
that fusion might exploit — judge on the 6-metric panel.

Three orthogonal probes for the bottleneck claim (whichever way the
audit lands):

1. **Architecture probe** — the trivial integration "floor" itself.
2. **Capacity/window probe** — OdomCNN at 2× width OR 2× window
   (32 steps ≈ 2 s).
3. **Preprocessing probe** — raw vs feature-normalised vs
   integrated-displacement input.

After this iteration Phase A is closed; the audit verdict goes
into the Phase B fusion-design discussion.

## Steps

### Step 0 — Recovery + sanity check (5 min)

**Step 0a.** Run-1 added `scripts/inspect_03_transfer.py`,
`scripts/inspect_06_model_behavior.py` and others that touch
odometry. Most aren't needed here, but the **fusion config** and
the **dataset loader** are. Check what's already on this branch:

```powershell
git diff --name-only main..overnight-autonomous-2026-05-24 -- `
  src/pipeline/data `
  configs/stage_a/odom `
  scripts/inspect_*odom*
```

Restore only what's needed for an Odom-only training loop. If the
project already has a `WebotsModalityDataset` or `AsyncCollection
loader on this branch, use it; if not, restore the run-1 version.

`src/pipeline/encoders/odom.py` is already on this branch (confirmed
2026-05-25 15:10) — no encoder file recovery needed.

**Step 0b.** Confirm Webots data shape. The data layout per
`data/async_collection/path_01/`:
- `odometry.csv` — ~15 Hz, 7-column (per CLAUDE.md sensor rate table)
- `ground_truth.csv` — ~10 Hz, (x, y, θ)

Engineer reads the column header of `odometry.csv` and writes it in
the RESULT (it determines what the trivial baseline integrates).
Expected columns include `vx, vy, omega` (linear + angular vel) and
possibly cumulative `x, y, theta` already integrated by the Webots
controller.

**Acceptance:** column header recorded; either a clean (v, ω) pair
or a clean (Δx, Δy, Δθ) pair is found in the columns.

### Step 1 — Trivial integration baseline (the "Day-1 SOTA" analog)

Compute the pure-integration position estimate on the canonical
Webots split (train [1, 3-12], val [2, 13, 14], test [15, 16, 17]).

Two integration variants — pick whichever the columns support, run
the simpler first:

- **Variant I-A (preferred): cumulative-position columns.** If
  `odometry.csv` already contains a cumulative `x, y` (most Webots
  odometry controllers do), the trivial baseline is **the column
  itself**, possibly with a per-path origin shift to align t=0 with
  ground-truth t=0. No integration; just align and compute MAE.
- **Variant I-B: velocity integration.** If only `(vx, vy, omega)`
  is available, integrate forward Euler from `(x_0, y_0, θ_0) =
  ground_truth[0]` and step `(x_t, y_t)` accordingly. Same MAE
  computation.

Compute per-sample MAE (every odom sample's position vs the
ground-truth position interpolated to that timestamp). Report:
- val MAE (per-path mean, then aggregate)
- test MAE (same)
- val-test gap
- per-path distribution (p25 / p50 / p75 / p90 / max)

This is the **`OdomCNN` audit floor**. If OdomCNN doesn't beat
this, it has no reason to exist.

**No training. No GPU. Just numpy + the canonical split.**

**Acceptance:** trivial-integration test MAE reported as the audit
floor. Time-budget: 10 min including writeup.

### Step 2 — `OdomCNN` on Webots canonical split (preprocessing variants)

Train `OdomCNN` (default: window=16, embed_dim=128, channels=(16,32,64))
as a frozen embedding → linear position head. Same protocol as
PLAN_03: AdamW + OneCycleLR + Huber(δ=0.5), 30 epochs, batch 64.

Run two preprocessing variants (correction #2):

- **P-A (default):** raw 7 odom columns, per-feature train mean/std
  normalisation. The pipeline's current convention per `imu.py` /
  `odom.py` defaults.
- **P-B (Δ-features):** for the cumulative columns (x, y, theta),
  replace with their first-difference `(Δx, Δy, Δθ)`; keep
  velocities as-is. This removes the absolute-position drift
  feature and forces the encoder to learn from local motion only.

**Pre-test gate:** 10 % of train paths for 5 epochs; val MAE
moves ≥ 10 %. Kill on failure.

**Memory budget check:** trivial — OdomCNN is ~0.04 M params at
default, ~0.16 M at the 2× width capacity probe. Report peak GPU
MB at full batch (B=64, window=16, 7 channels); must be < 6 GB.

- **Acceptance for Step 2:** train completes; val MAE + test MAE
  reported per preprocessing; per-path distribution + per-traj
  smoothness ratio + per-traj plots for paths 15/16/17 saved
  under `runs/overnight/run2_iter_04/test_paths/`.

### Step 3 — Multi-condition (val/test) gate

Same as PLAN_03 Step 3. Compute the test-val gap ratio
`(test_MAE − val_MAE) / val_MAE` per preprocessing. Acceptance for
`keep`: gap < 20 %.

The Webots paths are the same world but different trajectories →
this is "within-world cross-trajectory" transfer, the same limitation
PLAN_03 documented. C3 (4-modality fusion claim) doesn't need
cross-dataset Odom (Odom is sim-only by project design).

### Step 4 — Capacity OR window probe (one orthogonal probe)

Pick ONE — prefer (b) since CLAUDE.md says the IMU encoder uses a
32-step window and odom uses 16, so a 32-step probe answers
"would more temporal context help?":

- (a) Width: `channels=(32, 64, 128)`, `embed_dim=256` (2× model).
- (b) Window: `window=32`, channels and embed_dim unchanged.

Train same protocol as Step 2 default. Report test MAE delta vs
Step 2 P-A. If the probe closes ≥ 50 % of the gap to (or extends
the win over) the trivial integration baseline, the audit label
moves toward `modify` with the probe direction baked in.

### Step 5 — Six-metric harness (Webots val embeddings)

Run `src/pipeline/evaluation/encoder_eval.py` on the Step 2 P-A
val embeddings, the Step 2 P-B val embeddings, AND the Step 4
probe val embeddings. All six metrics apply (Odom on Webots is
temporally ordered).

If the harness file isn't on this branch, restore from run-1 the
same way as previous Step 0a's. If non-trivial, skip Step 5 and
ship a partial RESULT.

### Step 6 — Audit decision (amended-rubric)

Label `OdomCNN` with `keep` / `modify` / `replace` per:

- **Multi-condition gate (Step 3):** test-val gap < 20 % for `keep`.
- **Floor gate (Step 1 vs Step 2):** `OdomCNN`'s raw test MAE must
  beat the trivial integration's raw test MAE by at least 10 %
  (otherwise — `replace` with the trivial integration as the
  fusion-input path).
- **Preprocessing-aware (correction #2):** label which
  preprocessing (P-A or P-B) the verdict is conditioned on.
- **Raw-weighted (correction #3):** raw test MAE is primary;
  alignment metrics not used (Webots is in-world, no scale
  ambiguity).

One-line justification per label quoting numbers.

**Phase A closes after this RESULT.** Recommend PLAN_05 = C2
closure per STATE's locked plan (data-acquisition + canonical
RoNIN unseen-subjects re-eval).

## Sources

- `src/pipeline/encoders/odom.py` — `OdomCNN` class (on this branch,
  unchanged since the public-release restructure).
- Webots Tiago async-collection dataset: `data/async_collection/path_*/`,
  per CLAUDE.md.
- Amended audit rubric: STATE.md "Amended audit rubric (locked
  2026-05-25 ~12:55 local, applies PLAN_03+)".
- Pre-run-1 Odom baseline (from CLAUDE.md Stage-A table):
  `Linear Probe MAE ~3.5 m`. No published external SOTA on this data.

## What to report back

In `handoff/results/RESULT_04_odom-encoder-audit-webots.md`:

1. **Step 0** — column header of `odometry.csv` + integration
   variant chosen (I-A or I-B).
2. **Step 1** — trivial integration val MAE + test MAE + per-path
   distribution. This IS the floor; everything else compares
   against it.
3. **Step 2 headline** (two preprocessing variants):

   | preprocessing | val MAE | test MAE | test-val gap % | test p25/p50/p75/p90/max | per-traj smoothness median r | params | latency b=1 ms |
   |---|---|---|---|---|---|---|---|
   | trivial integration (floor) | … | … | … | … | … | — | — |
   | OdomCNN P-A (default norm) | … | … | … | … | … | … | … |
   | OdomCNN P-B (Δ-features) | … | … | … | … | … | … | … |

4. **Step 3** — multi-condition gate pass/fail.
5. **Step 4** — capacity/window probe delta.
6. **Step 5** — 6-metric harness rows.
7. **Step 6 audit label** + 3-sentence justification.
8. **Per-trajectory plots** (paths 15/16/17) saved under
   `runs/overnight/run2_iter_04/test_paths/`, listed in RESULT.
9. **Phase A summary table** (4 encoders, their labels, key
   numbers) — at the end of RESULT_04. This is the input to PLAN_05
   and Phase B.
10. **One open question** for scientist.

## Reversibility

- Step 0 (file recovery): permanent if anything restored; throwaway
  if nothing needed.
- Step 1 (trivial integration): throwaway; just numpy.
- Step 2 (OdomCNN training): throwaway; checkpoints under
  `runs/overnight/run2_iter_04/` (gitignored).
- Step 4 (capacity/window probe): permanent if it creates a NEW
  script (`scripts/_eval_webots_odom_*.py`); throwaway if it just
  flips a config kwarg.
- Steps 3, 5, 6: documentation.

Files committed: RESULT_04, restored files (if any), one new
`scripts/_eval_webots_odom.py` (the main audit driver mirroring
`_eval_webots_dpvo.py`'s structure from PLAN_03 — same per-path
distribution + 6-metric + plots pattern).

**Compute budget:** total iteration ≤ 60 min (smaller than
previous iterations — OdomCNN is tiny, integration is free).
- Step 0: 5 min.
- Step 1: 10 min (numpy + plotting).
- Step 2: 15 min (two head trainings, ~7 min each on tiny CNN).
- Step 3: 0 extra min (uses Step 2 outputs).
- Step 4: 10 min (one probe training).
- Step 5: 5 min (eval-only).
- Step 6 + Phase A summary table: 15 min writeup.

If overrun: cut Step 4 first (capacity probe — least informative
for the `keep`-vs-`replace` question, which is decided by Step 1
vs Step 2's floor comparison).

If the trivial integration somehow beats OdomCNN on raw test MAE
(plausible — odometry on Webots may be too clean to need a
learned encoder), DO NOT hide the finding; the audit label is
`replace`, the Phase B plan adopts raw integrated odom as the
direct fusion input, and the run-2 paper's narrative becomes
"3 learned encoders + 1 integrated feature" — which is a fine
PerCom-style finding.

**Demand #3** — no external SOTA to honour; OdomCNN is ours.
