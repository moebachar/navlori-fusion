# Plan 06 — Phase B foundation: restore fusion stack + reproduce run-1 2-modality baseline on Webots

> **Scope change from STATE's earlier draft.** STATE.md originally
> labelled PLAN_06 as "Camera external-SOTA validation (TartanAir /
> EuRoC / KITTI)" per the third-party review of RESULT_03. After
> RESULT_05 closed Step 0 of the Camera retros and reframed Camera's
> verdict as `keep with smoothness debt` AND the FRDR/Globus gate
> killed C2 closure, I'm flipping the priority: **PLAN_06 becomes
> Phase B foundation**. Rationale:
>
> 1. **Time-value.** Run 2 has ~23 hours left. The headline
>    publishable contribution (C3 — 4-modality fusion on Webots) has
>    NOT been measured yet. Camera external-SOTA validation is
>    supporting evidence for the per-leg paper claim, but it's the
>    *same shape* of blockage as C2 (needs heavy external setup:
>    `lietorch`/`altcorr` on Windows, or TartanVO + TartanAir
>    download). It risks another iteration lost to infrastructure.
> 2. **Two deferred Phase C items now, not one.** Both C2 (canonical
>    RoNIN unseen-subjects) and Camera external-SOTA are blocked on
>    external dependencies (Globus auth / CUDA-op installation +
>    public-dataset acquisition) and get bundled as a single
>    **manual / Phase C** task. STATE.md is updated to reflect this.
> 3. **Phase B feeds back into per-leg validation.** If Phase B's
>    bake-off shows a 4-modality fusion that delivers C3 cleanly,
>    the per-leg claim becomes easier to sell ("our encoders
>    composed correctly") even with softer per-leg evidence. The
>    reverse — running Camera external-SOTA without ever having
>    measured a 4-modality fusion — wastes the run.
>
> **PLAN_06's focused experiment** is the smallest viable Phase B
> step: restore the run-1 fusion infrastructure to this branch and
> reproduce the run-1 **WiFi + IMU** baseline (~0.43 m val MAE per
> CLAUDE.md) on Webots. Camera + Odom modality additions are
> queued as PLAN_07 (Camera) and PLAN_08 (Odom 1.5-modality path).
> Bake-off (4 candidate fusion architectures) is queued as PLAN_09
> AFTER the 4-modality default works.

## Hypothesis

The run-1 `FusionTransformer` (`src/pipeline/fusion/transformer.py`)
+ `FusionTrainer` (`src/pipeline/training/fusion_trainer.py`) +
`builder.load_config` already produce a single-instant Webots
fusion baseline of **~0.43 m val MAE** with **WiFi (Anchor2Vec) +
IMU (IMUCNN)** at K=1 instant (CLAUDE.md "Stage A + B/C Complete"
section). That code is on `overnight-autonomous-2026-05-24` but not
on this branch (run-2 was cut from `main`).

This iteration restores the stack and re-runs that baseline.
**Acceptance** is producing a working 2-modality fusion training
loop on Webots that reports val MAE within ±15 % of the documented
0.43 m. If we hit ≤ 0.50 m val MAE, we've cleared C3's lower bound
(criterion (b)) with just 2 modalities — establishing a strong
floor before Camera/Odom add to the stack.

No fusion architecture change in this plan. No Camera. No Odom.
Just: **restore + reproduce.**

## Steps

### Step 0 — Restore fusion infrastructure from run-1 (15–20 min)

```powershell
git checkout overnight-autonomous-2026-05-24 -- `
  src/pipeline/fusion/__init__.py `
  src/pipeline/fusion/base.py `
  src/pipeline/fusion/builder.py `
  src/pipeline/fusion/transformer.py `
  src/pipeline/training/__init__.py `
  src/pipeline/training/fusion_trainer.py `
  src/pipeline/training/motion.py `
  src/pipeline/training/trainer.py `
  src/pipeline/data `
  src/pipeline/evaluation `
  src/pipeline/uncertainty `
  src/pipeline/filters `
  configs/stage_c/fusion.yaml `
  configs/stage_c/cross_attention.yaml `
  configs/training/default.yaml `
  configs/data/simulation.yaml `
  docs/fusion_pipeline.md `
  handoff/fusion-pipeline.md `
  scripts/_smoke_fusion.py
```

`src/pipeline/encoders/__init__.py` is already extended for WiFi +
Camera; Step 0 of this plan re-extends it for IMU/Odom imports if
needed. Verify the encoder registry covers
`{Anchor2Vec, IMUCNN, OdomCNN, DPVOMotionEncoder, WiFiSetTransformer}`
post-restore.

**Smoke imports**:

```powershell
python -c "from src.pipeline.fusion import FusionTransformer; \
           from src.pipeline.training.fusion_trainer import FusionTrainer; \
           from src.pipeline.fusion.builder import load_config, build_datamodule; \
           print('imports ok')"
```

Any missing-module ImportError → restore the chain (encoders/data
sub-packages) until imports succeed. CLAUDE.md "Known gotchas"
warns: `__init__.py` files in run-1 had `from src.X` absolute imports
that fail when the package is loaded as a relative module — use
relative imports (`from .X import ...`).

**Acceptance:** the smoke import + a `load_config('simulation')`
call complete without exception.

### Step 1 — Sanity probe on simulated data (5–10 min)

Run `scripts/_smoke_fusion.py --phase 1` (the shape-NaN check from
run-1, per the 5-phase smoke pattern in CLAUDE.md). Expected: one
forward pass shape-checks, no NaNs.

**Pre-test gate (synthetic):** if `_smoke_fusion.py` fails Phase 1,
do NOT proceed to training; write a partial RESULT with the
specific import / shape / load error and pause for scientist
review.

### Step 2 — Reproduce run-1 2-modality baseline on Webots

The run-1 `configs/stage_c/fusion.yaml` + `configs/data/simulation.yaml`
should already be set up for WiFi + IMU on Webots. Verify config
fields:
- `data: simulation` (Webots TIAGO++ async_collection paths
  [1, 3-12] / [2, 13, 14] / [15, 16, 17] per CLAUDE.md).
- `modalities: [wifi, imu]` (NOT all four — that's PLAN_07/08).
- `temporal: K=1` (single-instant for the first reproduction;
  K-instant probe later).
- `encoders: Anchor2Vec + IMUCNN`.
- `model.embed_dim: 128`.

Train using the canonical builder pattern from CLAUDE.md:

```powershell
.venv\Scripts\python.exe -c "from src.pipeline.fusion.builder import \
  load_config, build_datamodule, build_encoders, build_model, build_trainer; \
  cfg = load_config('simulation'); dm = build_datamodule(cfg); \
  encs = build_encoders(cfg, dm); model = build_model(cfg, encs); \
  trainer = build_trainer(cfg, model, dm); trainer.fit()"
```

(Engineer may wrap in a `scripts/_train_webots_2mod_baseline.py`
runner — small thin wrapper, OK to commit.)

**Memory budget check:** forward+backward at target batch (B=32 or
config default) with 2 modalities + K=1. Peak GPU MB reported; must
be < 6 GB.

**Pre-test gate:** 5-epoch run on 10 % of Webots train paths.
Acceptance: val MAE drops ≥ 10 % across the 5 epochs OR an early
loss curve that's clearly descending.

**Full training:** 90 epochs (or the existing `fusion.yaml` budget),
patience 25, OneCycleLR. Should land near 0.43 m val MAE per
CLAUDE.md.

**Acceptance:** val MAE within ±15 % of 0.43 m (0.37–0.49 m). Test
MAE reported. Per-path distribution (criterion (d)) for test paths
15/16/17 with `per-path mean / p25 / p50 / p75 / p90 / max` and
per-trajectory plots saved under
`runs/overnight/run2_iter_06/test_paths/`.

### Step 3 — Latency probe (the C3-(e) check, free with Step 2)

After training, time per-sample inference at batch=1 on the GPU.
Report ms/sample. Acceptance (criterion (e)): < 100 ms/sample on
the Quadro P4000. CLAUDE.md says run-1 hit 4.2 ms at K=8 — so K=1
should be well under 10 ms.

### Step 4 — Per-modality subset eval (the C3 per-modality check)

Re-evaluate the trained 2-modality model under three subsets,
without retraining (using the modality-dropout / instant-dropout
machinery already in `FusionTrainer.evaluate_subsets` per CLAUDE.md):

| subset | description | expected |
|---|---|---|
| `only:wifi` | mask IMU at eval | close to WiFi-only baseline (≈ 0.46 m per CLAUDE.md run-1 note) |
| `only:imu` | mask WiFi at eval | poor — IMU has no absolute anchor (≫ 1 m) |
| `wifi+imu` | both | the full-fusion number from Step 2 |

This validates that fusion isn't trivially WiFi-only AND surfaces
whether IMU contributes net positive (run-1 found IMU injected
noise at higher embed_dim — see `handoff/archive/run1/results/
RESULT_04_wifi-encoder-capacity-probe.md`).

**Acceptance:** all three numbers reported in a table.

### Step 5 — Decision / PLAN_07 setup

Write a 3-sentence decision in RESULT_06:
- Does the 2-modality baseline reproduce within ±15 %? (= the
  infrastructure is sound)
- Does `only:wifi` − `wifi+imu` show IMU as net-positive or
  net-noise? (= informs whether IMU goes in cleanly or needs the
  late+gate treatment for the bake-off)
- PLAN_07 recommendation (default: add Camera DPVOMotion-P-A as
  3rd modality, K=1, same FusionTransformer; minimal architecture
  change).

## Sources

- Run-1 fusion stack: `overnight-autonomous-2026-05-24`
  branch — `src/pipeline/{fusion,training}/`, `configs/stage_c/`,
  `scripts/_smoke_fusion.py`.
- Documentation:
  - `docs/fusion_pipeline.md` (run-1, restored here).
  - `handoff/fusion-pipeline.md` (run-1, restored here).
- Reference numbers (CLAUDE.md "Stage A + B/C Complete" + run-1
  RESULT_03/04 in archive):
  - Single-instant fusion ≈ 0.43 m val MAE on Webots.
  - WiFi-only ≈ 0.46 m val MAE on Webots.
- Phase A summary table (RESULT_04 / RESULT_05): all encoder
  verdicts.

## What to report back

In `handoff/results/RESULT_06_phase-b-foundation.md`:

1. **Step 0 outcomes** — files restored; any missing-module chains
   resolved; final `__init__.py` exports list.
2. **Step 1 smoke** — `_smoke_fusion.py --phase 1` pass/fail.
3. **Step 2 baseline** —
   | metric | value | run-1 ref | within ±15 %? |
   |---|---|---|---|
   | val MAE | … | 0.43 m | … |
   | test MAE | … | (run-1 didn't publish) | n/a |
   | best epoch | … | … | … |
   | wall (min) | … | … | … |
   + per-path table for test paths 15/16/17 + per-trajectory plots.
4. **Step 3 latency** — ms/sample at b=1; passes < 100 ms?
5. **Step 4 subset eval** — `only:wifi`, `only:imu`, `wifi+imu`
   numbers; IMU contribution interpretation.
6. **Step 5 decision** — infrastructure verdict + PLAN_07
   recommendation.
7. **Open question** for scientist (likely about adding Camera as
   3rd modality vs. running the late+gate bake-off first).

## Reversibility

- Step 0 (file recovery): permanent. Engineer commits the restored
  set with the result.
- Step 2 (training): throwaway model; checkpoint optional under
  `runs/overnight/run2_iter_06/` (gitignored).
- Steps 3–5: documentation.

Files committed: restored Phase B stack, RESULT_06, possibly a
small wrapper `scripts/_train_webots_2mod_baseline.py`.

**Compute budget:** total iteration ≤ 90 min.
- Step 0: 20 min (file recovery + chain resolution + smoke import).
- Step 1: 5 min.
- Step 2: 30 min (90-epoch training, small WiFi+IMU model).
- Step 3: 5 min (latency probe).
- Step 4: 5 min (subset eval, no retraining).
- Step 5: 10 min writeup.

If Step 0 takes > 30 min (unexpected import chains, dependency
errors), write a partial RESULT after 30 min with the blockage
specifics. Scientist takes the import-chain map and writes a
PLAN_06b targeted at the chain.

If Step 2 doesn't reach ±15 % of 0.43 m after 90 epochs, document
the gap honestly. The next iteration probes why (config drift,
encoder restoration missed a step, data path issue) rather than
silently moving on.

## Deferred Phase C tasks (manual / user-required)

Two items now bundled as Phase C manual work, captured for the
scientist + user record:

1. **C2 closure** — fetch canonical RoNIN FRDR archive via Globus
   (requires interactive credentials), run ResNet1D + IMUCNN on
   `list_test_unseen.txt` with Umeyama alignment, file C2 as
   discharged or not. Original instructions in
   `RESULT_02_imu-encoder-audit-ronin.md` addendum.
2. **Camera external-SOTA validation** — pick public VO benchmark
   (TartanAir / EuRoC / KITTI), pick method (DPVO unmodified if
   `lietorch`/`altcorr` install unblocked on Linux/WSL2, else
   TartanVO MIT pure-PyTorch), run our DPVOMotionEncoder trunk
   alongside, file per-leg validation as paper-strength. Original
   queue note in `RESULT_03_camera-encoder-audit-webots.md`
   addendum + RESULT_05's Step 0c.

Both are necessary for paper-strength per-leg validation (C1/C2)
but are NOT necessary for the C3 (4-modality fusion on Webots)
claim. STATE.md is updated to reflect Phase C bundling.

**Demand #3** — no edits to vendored sources at any stage.
