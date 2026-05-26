# Plan 33 — Notebook: real inline training (no shortcuts, no cached fallbacks under `FAST_MODE=False`)

> **User pushback 2026-05-26 ~17:10 local.** Scientist's audit
> of the v2 notebook surfaced that RESULT_32 shipped only 1 of 7
> inline-training cells the directive required:
> - ✓ §2.1 Anchor2Vec: real inline training.
> - ✗ §2.2 IMUCNN: hand-typed `imucnn_canonical_raw = 9.961`.
> - ✗ §2.3 DPVOMotion + head: markdown only.
> - ✗ §2.4 OdomCNN: missing entirely (no cell).
> - ✗ §3 incumbent / CNN1D / LSTM-attn: load cached checkpoints
>   EVEN UNDER `FAST_MODE=False` — fake mode flag for §3.
>
> User's directive: **no shortcuts, take time to get it right,
> publication-grade.** The `FAST_MODE` flag must honestly mean
> what it says in every cell. PLAN_33 closes the 6/7 gap.

## Hypothesis

After this iter, the notebook satisfies the original PLAN_32
contract:

1. **Every "Ours" per-leg cell** (§2.1–§2.4) and **every
   fusion-arch cell** (§3.1–§3.3) honours `FAST_MODE`:
   - `FAST_MODE=True` → load saved checkpoint + eval live.
   - `FAST_MODE=False` → **train inline + eval live; NO cached
     fallback path**.
2. **Drift per "Ours vs archive" cell < 2 %** in both modes.
3. **Both modes smoke-tested end-to-end** via
   `jupyter nbconvert --to notebook --execute --inplace`:
   - `FAST_MODE=True`: ~10-15 min wall-clock.
   - `FAST_MODE=False`: ~45-55 min wall-clock (NOT the 2-3 h
     engineer estimated — actual training times per RESULT_07/17
     are smaller).
4. **No "polish item for next iter" deferral**. The 6 missing
   cells get implemented in this plan. If the engineer hits a
   surprising blocker, they raise it in a partial RESULT and
   stop — no fake-mode flags ship.

## Wall-clock reality check (corrects engineer's 75 min estimate)

Engineer's RESULT_32 estimated 75 min for the fusion-arch
training triplet. The actual archive numbers say:

| training | actual wall-clock | source |
|---|---|---|
| Anchor2Vec UJI (120 ep) | ~3 min CPU / ~10 s GPU | RESULT_32 |
| IMUCNN canonical RoNIN (20 ep on ~38k windows) | **~14 min** | RESULT_07 |
| DPVOMotion + head on TartanAir (linear head on cached features) | **~5 min** | RESULT_08 |
| OdomCNN Webots (30 ep on ~8.5k windows) | **~5 min** | RESULT_04 |
| FusionTransformer (incumbent) 4-mod K=4 B=128 90 ep | **~217 s = 3.6 min** | RESULT_06 |
| CNN1D 4-mod K=4 B=128 90 ep | **~196 s = 3.3 min** | RESULT_17 |
| LSTM-attn 4-mod K=4 B=128 90 ep | **~202 s = 3.4 min** | RESULT_17 |
| **Total `FAST_MODE=False` inline training** | **~35 min** | sum of above |

Plus per-cell eval overhead (~10 min total): full notebook
slow-mode ~45-55 min. The 75-min and 2-3-hour estimates were
wrong. Slow mode is reasonable for a publication-reproducibility
artifact.

## Steps

### Step 0 — Add the 3 missing encoder inline trainers (30 min)

`src/pipeline/training/inline_encoders.py` (extending RESULT_32's
file). Add the 3 missing helpers, each modelled on
`train_anchor2vec`'s pattern (the existing template):

```python
def train_imucnn(seqs_train, seqs_test, root_dir, **cfg):
    """Train IMUCNN on RoNIN canonical train sequences, eval ATE
    on `seqs_test`. Replicates RESULT_07's
    `scripts/_eval_imucnn_ronin_canonical.py` recipe.
    Returns (model, head, per_seq_ate, history). ~14 min on
    Quadro P4000."""
    ...

def train_odomcnn(Xtr, Ytr, Xva, Yva, Xte, Yte, *,
                  features="delta_features",  # P-B winner
                  window=16, embed_dim=128, **cfg):
    """Train OdomCNN on Webots train paths with P-B Δ-features
    preprocessing (RESULT_04 winner). Returns (model, head,
    metrics, history). ~5 min."""
    ...

def train_dpvo_motion_head(image_pairs_train, image_pairs_test,
                            poses_train, poses_test, **cfg):
    """Train the DPVOMotionEncoder's head on TartanAir hospital
    train slice (first 80 % of P000). DPVO trunk stays frozen.
    Returns (head, predicted_traj, gt_traj, history). ~5 min."""
    ...
```

Each helper's signature mirrors `train_anchor2vec`'s pattern:
explicit train + val + test arrays in, trained-model-plus-eval-
results out. Each saves a checkpoint at a deterministic path
under `runs/<canonical_paths>/`.

**Acceptance**: each helper runs cleanly on synthetic-size data;
returns the documented tuple. Engineer factors out from
`scripts/_eval_*.py` (the recipes already exist there).

### Step 1 — Rewrite §2.2 (IMUCNN canonical RoNIN) (10 min)

Replace cell 17's hand-typed values with:

```python
# §2.2 — IMUCNN on canonical RoNIN unseen subjects (Ours).
# Imported from src.pipeline.encoders.IMUCNN; trained inline
# (FAST_MODE=False, ~14 min) or loaded from checkpoint
# (FAST_MODE=True, ~5 s).
from src.pipeline.training import train_imucnn

ckpt = ROOT / "runs" / "encoder_audit_imu" / "imucnn_ronin_canonical.pt"
if FAST_MODE and ckpt.is_file():
    print(f"Loading IMUCNN checkpoint from {ckpt}")
    # Load: build IMUCNN + head, load state_dicts, run eval on canonical unseen.
    model, head, per_seq_ate = load_imucnn_eval_canonical(ckpt, ronin_root, seqs_test)
else:
    print(f"Training IMUCNN on canonical RoNIN train ({len(seqs_train)} seqs, ~14 min on Quadro P4000)...")
    model, head, per_seq_ate, history = train_imucnn(
        seqs_train, seqs_test, ronin_root, epochs=20, batch_size=128, lr=1e-3, seed=SEED,
    )
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "head_state_dict": head.state_dict(),
                "per_seq_ate": per_seq_ate}, ckpt)
    print(f"Saved IMUCNN checkpoint to {ckpt}")

raw_ate_live = float(np.mean(per_seq_ate["raw_ate"]))
umey_ate_live = float(np.mean(per_seq_ate["umey_ate"]))
print(f"IMUCNN canonical unseen: raw ATE {raw_ate_live:.3f} m / Umeyama {umey_ate_live:.3f} m")
print(f"  Archive (RESULT_07): 9.961 / 7.876   |   drift: raw {abs(raw_ate_live - 9.961)/9.961*100:.1f}% / umey {abs(umey_ate_live - 7.876)/7.876*100:.1f}%")
print(f"  SOTA (§1.2 ResNet1D): {ronin_ate:.3f} m   |   raw gap +{(raw_ate_live - ronin_ate)/ronin_ate*100:.0f}%")
```

The hand-typed `imucnn_canonical_raw = 9.961` line is **gone**.
The number is computed live or loaded from a checkpoint. The
`live_numbers["imucnn_canonical_raw"]` dict entry gets the live
value for §5's drift report.

**Acceptance**: §2.2 produces raw ATE within 2 % of 9.961 m and
Umeyama within 2 % of 7.876 m. No `# RESULT_07` hand-typed
fallback survives.

### Step 2 — Add §2.3 DPVOMotion + head live cell (10 min)

§2.3 currently has only markdown. Add a code cell that:
- Loads `DPVOMotionEncoder` from `src.pipeline.encoders`.
- Extracts cached DPVO patch tokens (or computes them if needed —
  the trunk is frozen, ~30 s on the 563-frame hospital sample).
- In `FAST_MODE=True`: loads the linear head checkpoint at
  `runs/overnight/run2_iter_08/dpvo_head.pt` (or similar).
- In `FAST_MODE=False`: trains the linear head inline on the
  first-80 % slice; eval on the last-20 % slice (the RESULT_08
  protocol).
- Reports `last_20_ate_live` + drift vs RESULT_08's 0.293 m.

If the hospital data isn't on disk, the cell prints a setup
note + reads from cached features only — but it does NOT
silently fall back to hand-typed values.

**Acceptance**: §2.3 produces a live ATE in both modes within
±5 % of RESULT_08's 0.293 m (TartanAir trains less stably so
slightly looser tolerance).

### Step 3 — Add §2.4 OdomCNN inline + trivial-integration floor (10 min)

§2.4 has NO code cell at all today. Add:

```python
# §2.4 — OdomCNN on Webots vs trivial-integration floor (Ours).
# Imported from src.pipeline.encoders.OdomCNN; trained inline
# (FAST_MODE=False, ~5 min) or loaded (FAST_MODE=True).
from src.pipeline.training import train_odomcnn

# 1. Trivial integration floor (no training, just compute)
trivial_floor = compute_trivial_integration_floor("webots")
print(f"Trivial integration test MAE: {trivial_floor['test_mae']:.2f} m (per-traj smoothness r={trivial_floor['smoothness']:.3f})")

# 2. OdomCNN-P-B (the winner config from RESULT_04)
ckpt = ROOT / "runs" / "encoder_audit_odom" / "odomcnn_pb_webots.pt"
if FAST_MODE and ckpt.is_file():
    model, head, metrics = load_odomcnn_pb_eval(ckpt, "webots")
else:
    print("Training OdomCNN P-B on Webots train (~5 min)...")
    Xtr, Ytr, Xva, Yva, Xte, Yte = load_webots_odom_pb()
    model, head, metrics, history = train_odomcnn(
        Xtr, Ytr, Xva, Yva, Xte, Yte,
        features="delta_features", window=16, epochs=30, seed=SEED,
    )
    torch.save({...}, ckpt)

print(f"OdomCNN P-B val {metrics['val_mae']:.2f} / test {metrics['test_mae']:.2f} m")
print(f"  Archive (RESULT_04): 4.62 / 4.24   |   drift: ...")
print(f"  Trivial floor test: {trivial_floor['test_mae']:.2f}   |   OdomCNN beats floor by {...}%")
```

**Acceptance**: §2.4 produces val/test MAE within 2 % of
RESULT_04's 4.62/4.24 m; trivial floor reproduces 8.27 m.

### Step 4 — Rewrite §3 fusion bake-off: REAL inline training (15 min)

Replace cell 21's fake `FAST_MODE=False` (which currently falls
back to loaded checkpoints) with the honest implementation:

```python
# §3 — Phase B fusion bake-off (3 archs on full Webots K=4 4-mod B=128).
trainers = {}
ckpt_paths = {
    "incumbent": ROOT / "runs" / "overnight" / "run2_iter_13",
    "cnn1d":     ROOT / "runs" / "overnight" / "run2_iter_17" / "cnn1d",
    "lstm_attn": ROOT / "runs" / "overnight" / "run2_iter_17" / "lstm_attn",
}

if FAST_MODE:
    print("FAST_MODE=True: loading 3 cached fusion checkpoints...")
    for arch, p in ckpt_paths.items():
        trainers[arch] = load_trained(str(p), arch=arch, dataset="simulation")
        params_M = sum(x.numel() for x in trainers[arch].model.parameters()) / 1e6
        print(f"  {arch:10s}: loaded; {params_M:.2f} M params")
else:
    print(f"FAST_MODE=False: training 3 fusion archs inline (~{3.5 + 3.3 + 3.4:.0f} min wall-clock total)")
    for arch in ckpt_paths:
        print(f"  {arch}: training 90 epochs at K=4 B=128 ...")
        t0 = time.time()
        trainer = FusionTrainer.from_config("simulation", arch=arch,
                                              K=4, batch_size=128, lr=1.3e-3,
                                              epochs=90, seed=SEED)
        trainer.fit()
        ckpt_paths[arch].mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(ckpt_paths[arch])
        trainers[arch] = trainer
        print(f"    {arch}: done in {(time.time()-t0)/60:.1f} min")
```

**No `try / except / fallback` to cached checkpoints.** If
`FAST_MODE=False` and training fails, the cell errors loudly so
the engineer sees it.

`FusionTrainer.from_config(...)` + `save_checkpoint(...)` are
the required class-method additions (engineer confirms they exist
from RESULT_28 OR adds them; ~30 lines total).

**Acceptance**: in `FAST_MODE=False`, all 3 fusion archs train
inline; each takes 3-4 minutes; checkpoints saved; subsequent
eval reproduces val/test within 2 % of RESULT_13/17.

### Step 5 — Honest cell labels (5 min)

Every "Ours" cell (§2.1-§2.4 + §3.1-§3.3) prints at the top:

```
print(f"FAST_MODE={'TRUE (loading checkpoint)' if FAST_MODE else 'FALSE (training inline)'}")
```

Visual confirmation that the flag is honored per-cell. No cell
silently overrides the flag.

### Step 6 — Smoke-test BOTH modes end-to-end (30 min)

#### 6a. `FAST_MODE=True` smoke

```powershell
# Sanity-set the flag in the notebook + nbconvert
jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=1800 `
    notebooks/run2_walkthrough.ipynb
```

Expected: ~10-15 min wall-clock; all cells run; 0 errors; all
drifts < 2 %.

#### 6b. `FAST_MODE=False` smoke (NEW — this is the directive)

Engineer EITHER toggles the flag in the notebook source +
re-runs nbconvert, OR uses papermill with parameter override:

```powershell
pip install papermill
papermill notebooks/run2_walkthrough.ipynb /tmp/nb_slow_smoke.ipynb -p FAST_MODE False
```

Expected: ~45-55 min wall-clock; all cells run inline-train
where applicable; 0 errors; all drifts < 2 %.

Engineer records:
- Per-cell training time (so the actual numbers replace the
  estimates in §7's partition table).
- Per-cell drift in both modes.
- Total wall-clock both modes.

**Acceptance gate (HARD)**: BOTH modes produce clean
top-to-bottom executions. If `FAST_MODE=False` fails on any
cell, engineer fixes it BEFORE committing. No "polish item for
next iter" deferral.

### Step 7 — Update §7 partition table with measured times (5 min)

§7's partition table currently has estimated times. Replace
with the measured times from Step 6:

```
| section | FAST_MODE=True | FAST_MODE=False |
|---|---|---|
| §0 datasets       | ... | ... |
| §1 SOTAs          | ... | ... |
| §2.1 Anchor2Vec   | <10s load | ~3 min train |
| §2.2 IMUCNN       | <5s load | ~14 min train |
| §2.3 DPVOMotion   | <5s load | ~5 min train |
| §2.4 OdomCNN      | <5s load | ~5 min train |
| §3 fusion ×3      | <60s load | ~10 min train |
| §4 ablations      | ~5s | ~5s |
| §5 main table     | <1s | <1s |
| §6 honest gaps    | <1s | <1s |
| **TOTAL**         | **~12 min** | **~45-50 min** |
```

(Engineer fills with measured numbers.)

### Step 8 — Commit (5 min)

Single commit: notebook + new helpers in
`src/pipeline/training/inline_encoders.py` + the FusionTrainer
class methods (if added) + RESULT_33 + STATE.md update.

## Sources

- User pushback 2026-05-26 ~17:10 local (this directive).
- RESULT_32 (current notebook, partial inline-training).
- Wall-clock numbers cited from RESULT_06/07/17/04/08.
- `src.pipeline.training.train_anchor2vec` (RESULT_32 template).
- `src.pipeline.training.load_trained`,
  `FusionTrainer.from_config`, `save_checkpoint` (PLAN_28 /
  RESULT_28).

## What to report back

In `handoff/results/RESULT_33_notebook-no-shortcuts-real-inline-training.md`:

1. **Step 0** — 3 new inline trainer helpers in
   `inline_encoders.py`; smoke-imports + shapes match.
2. **Step 1** — §2.2 rewrite; live raw + Umeyama ATE vs archive.
3. **Step 2** — §2.3 new live cell; live ATE vs archive.
4. **Step 3** — §2.4 new live cell; trivial floor + OdomCNN
   numbers vs archive.
5. **Step 4** — §3 honest training path; per-arch wall-clock
   in `FAST_MODE=False` mode.
6. **Step 5** — every "Ours" cell prints its mode honestly.
7. **Step 6a** — `FAST_MODE=True` smoke: total wall-clock +
   drift table.
8. **Step 6b** — `FAST_MODE=False` smoke: total wall-clock +
   drift table.
9. **Step 7** — §7 partition table updated with measured times.
10. **One open question** for the user.

## Reversibility

- Step 0 (inline trainers): permanent under
  `src/pipeline/training/inline_encoders.py`.
- Steps 1-5 (notebook edits): permanent;
  `notebooks/run2_walkthrough.ipynb` gets the 6 rewrites + 2
  new cells.
- Steps 6a-6b (smoke): throwaway; engineer's verification only.

Files committed: `notebooks/run2_walkthrough.ipynb` (revised),
`src/pipeline/training/inline_encoders.py` (extended),
optional `src/pipeline/training/__init__.py` re-exports +
`FusionTrainer.{from_config, save_checkpoint}` if newly added.

**Compute budget**: ≤ 3 hours.
- Step 0: 30 min (3 helpers; mostly factoring from existing
  `scripts/_eval_*.py`).
- Step 1: 10 min.
- Step 2: 10 min.
- Step 3: 10 min.
- Step 4: 15 min.
- Step 5: 5 min.
- Step 6a: 15 min wall (10-15 min notebook execution).
- Step 6b: 60 min wall (45-50 min execution + verification).
- Step 7: 5 min.
- Step 8: 5 min.
- Buffer: 30 min for unexpected blockers (e.g. checkpoint format
  mismatches, missing data files for §2.3 TartanAir cell).

**If `FAST_MODE=False` smoke fails on a specific cell**:
engineer pauses, writes a partial RESULT with the failure
reproducer + the obstacle, schedules a follow-up. Does NOT
ship the notebook with a silent fallback or a "TODO" comment.

**If `FAST_MODE=False` total wall-clock exceeds 90 minutes**
(2× the estimate): engineer documents why + which cell is the
long pole + whether it's a real cost or a config issue. Does
NOT ship a notebook that lies about wall-clock.

## Quality bar (locked, no exceptions)

- No hand-typed result values in any cell.
- No `try / except / fallback` to cached checkpoints in
  `FAST_MODE=False` mode.
- No "TODO" or "polish item" comments — all 7 inline trainings
  ship.
- Both modes smoke-tested end-to-end.
- Drift < 2 % per "Ours vs archive" cell.
- Per-cell wall-clock measured + reported.

The notebook ships when these conditions hold simultaneously
in BOTH `FAST_MODE` branches.
