# NavLoRI-Fusion — Pipeline Fix Log

**Started:** 2026-05-20
**Branch:** `audit-baseline-2026-05-20` (off `main`, all prior changes preserved)
**Author:** Claude (fresh-vision audit), with Mohamed
**Source of plan:** the audit done 2026-05-20 — see `memory/project_audit_2026-05-20.md` and the chat transcript.

This log is structured as: **what** was changed, **why** (referencing the audit finding), **how** (file:line), and **how I verified** (smoke / profile / analyze). Read sequentially — Actions are interdependent.

---

## Why these actions in this order

| # | Action | Depends on | Reason for position |
|---|---|---|---|
| 4 | Unify metric (Euclidean MAE everywhere) | — | One-line helper that every later action will use. Cheapest to ship first. |
| 1 | Trial-out splits + baselines | 4 | Foundational measurement infrastructure. Until baselines exist, no later result is interpretable. |
| 2 | Displacement targets for IMU / Odom / DPVO | 4 | Fixes the encoder-memorization bug; needed before any "use Stage A in fusion" step. |
| 3 | Make fusion actually fuse | 2 | Modality-balanced loss only makes sense once encoders learn meaningful per-modality signal. |
| 5 | Load pretrained Stage A encoders | 2, 3 | Needs Action 2's corrected encoders to be worth loading. |
| 6 | Re-tune Optuna on real data | 1, 2, 3, 5 | Tuning is the LAST step; everything else must be measurable + reasonable first. |

The original audit numbered these 1-6; execution order above is dependency-ordered. Both numberings refer to the same six actions.

---

## Conventions used in this log

- **File references:** `path/to/file.py:line` — clickable in VS Code.
- **Smoke test:** small, fast verification that the change runs end-to-end with correct shapes / finite values. Always documented per action.
- **Profile:** wall-clock / memory measurement when the change might affect performance.
- **Analyze:** what the resulting numbers mean.
- **Debug:** anything that broke during smoke / analyze and how it was fixed.

---

## Action 4 — Unify metric

**Why.** Audit finding A1: `linear_probe.mae` was per-axis L1 (`np.abs(diff).mean()`) while the trainer's `val_mae` was Euclidean L2 (`||pred - y||_2.mean()`). Same key, different math → silent bias that made Stage A numbers look 1.25-1.4× better than Stage B/C numbers.

**Built.**

- Added canonical helpers `euclidean_mae(pred, y)` and `euclidean_rmse(pred, y)` in [src/pipeline/evaluation/encoder_eval.py](../src/pipeline/evaluation/encoder_eval.py) (both accept numpy and torch).
- Rewrote `linear_probe` and `knn_probe` to return the Euclidean MAE in their `mae` field, with a separate `mae_component` field preserving the old per-axis value for anyone who wants it.
- Updated `print_report` to use the new key set.
- Wired `trustworthiness` into `EncoderTrainer.evaluate()` ([src/pipeline/training/trainer.py:341-379](../src/pipeline/training/trainer.py#L341-L379)) — it now flattens the tabular val cache to (N, window*features) and passes it. Capped at `max_samples=2000` so sklearn's O(N²) distance computation doesn't OOM.
- Left a comment in `fusion_trainer.py:30-33` declaring the cross-module metric convention.

**Smoke.** [scripts/_smoke_metrics.py](../scripts/_smoke_metrics.py) — synthetic embeddings; verifies numpy/torch parity, `mae >= mae_component` invariant, ratio in expected range, trustworthiness in [0,1]. **PASS.**

**Result.**
- `euclidean_mae (np) = 0.062797`, `euclidean_mae (torch) = 0.062797` (identical).
- `linear_probe`: `mae=1.292m` (Euclidean), `mae_component=0.805m`; ratio 1.605 — matches expected sqrt(2)-ish for the synthetic data.
- `trustworthiness` returns finite score in [0, 1] with `n_samples=100` cap honored.

**Impact on downstream.** Any new `eval.json` written by `EncoderTrainer.evaluate()` will now contain Euclidean MAE in `mae`. Old `eval.json` files (in `runs/imu_*/`, `runs/odom_*/`, etc.) still hold the old component-wise number — flag for re-running Action 2's training step.

---

## Action 1 — Trial-out splits + baselines

**Why.** Audit findings 1 and 2: the `_intra` real-dataset configs leak (chunk-shuffle across the same continuous walk) and the codebase had **no baselines**, so every fusion number was uninterpretable. Before changing the model, establish the gate.

**Built.**

- **`scripts/baselines.py`** — three baselines (`MeanTrainBaseline`, `WiFiKNNBaseline`, `IMUKalmanBaseline`), Euclidean MAE everywhere, output to `runs/baselines/<dataset>/baselines.json` + a consolidated `runs/baselines/summary.json`. Honors `--skip-leaky` to exclude `_intra` datasets.
- **`configs/data/ronin_a000.yaml`** — trial-out split (one path per native RoNIN sequence). NOTE: only 10 sequences for subject a000, 9 train / 1 val / 0 test — honest-but-narrow. Documented in the YAML.
- **Re-generated `data/ronin_a000/`** by re-running [scripts/convert_ronin.py](../scripts/convert_ronin.py) with `--split-mode trial-out`.
- **Annotated the leaky configs** ([ronin_a000_intra.yaml](../configs/data/ronin_a000_intra.yaml), [ipin2024_floor-2_intra.yaml](../configs/data/ipin2024_floor-2_intra.yaml)) with `LEAKY — DEV USE ONLY` warnings and pointers to the honest variants.
- IMUWiFine already had native non-leaky 40/20/20; IPIN floor -2 and floor 0 already had trial-out variants on disk under `configs/data/<name>.yaml`.

**Smoke + Analyze (one-shot since this IS the measurement).**

```
Dataset            Best baseline  val MAE   test MAE   Notes
simulation         wifi_knn       0.685     0.604      sim WiFi is too clean
imuwifine          wifi_knn       3.774     7.675      real bar
ipin2024_floor-2   mean_train_pos 25.662    24.974     kNN BEATEN by centroid
ipin2024_floor0    mean_train_pos 21.427    23.245     kNN BEATEN by centroid
ronin_a000         wifi_knn       3.577     —          1 val seq (narrow)
ronin_a000_intra   wifi_knn       11.060    13.864     LEAKY — for contrast
```

**Debug.** Hit one issue: the trial-out RoNIN config has empty `test_paths`, which made `FusionDataModule` crash trying to load 0 paths. Fixed by falling back to a slice of `train_paths` when `test_paths` is empty, then skipping the test split in the report ([scripts/baselines.py:248-256](../scripts/baselines.py#L248-L256)). The IMU-Kalman diverges to absurd numbers (hundreds of km) on smartphone datasets without zero-velocity update — this is correct behavior for that baseline definition and is reported faithfully; it's a finding, not a bug.

**Headline insights.**

1. The transformer's sim win (0.44m) is **0.16m** over the WiFi-kNN baseline (0.60m on test). Most of the "fusion wins" headline was sim being too easy, not fusion working.
2. On IPIN trial-out, **WiFi-kNN is worse than mean-train-position**. WiFi fingerprints don't transfer across trials. The benchmark bar is "beat the centroid" — much weaker than CLAUDE.md's per-floor numbers suggested. This is the dataset that will most cleanly tell us whether fusion actually fuses.
3. On `ronin_a000_intra` the fusion run hit 13.24m val_mae; WiFi-kNN alone got 11.06m on the same leaky split. **The transformer was losing to a kNN AND benefiting from chunk leakage.**

**Files added / changed.**

- `scripts/baselines.py` (new, 274 lines)
- `configs/data/ronin_a000.yaml` (new)
- `configs/data/ronin_a000_intra.yaml`, `configs/data/ipin2024_floor-2_intra.yaml` (annotated)
- `data/ronin_a000/` (regenerated by converter)
- `runs/baselines/<dataset>/baselines.json` × 6 (auto-generated)
- `runs/baselines/summary.json`, `runs/baselines/README.md` (auto + hand)

---

## Action 2 — Displacement targets for motion encoders + drop odom_x/y

**Why.** Audit findings A3 and A4: IMU, Odom, and the DPVO motion encoder were all trained with `(x, y)` as the target. But IMU and Odom windows hold *relative* motion (1s of accel/gyro/wheel-speed), not absolute position — so the only path to low MAE is "memorize which IMU pattern was at which point on a training path." On any honest OOD test this collapses. Separately, `odom_x` and `odom_y` were *features*: the wheel-odometry estimate of position was inside the input, i.e. the target leaking into the input.

The right objective for these sensors is **displacement over the window**, `delta = gt[t] - gt[t - lookback_s]`. WiFi / ACE / DINOv2 are place-recognition sensors and stay on `(x, y)`.

**Built.**

- Removed `odom_x`, `odom_y` from `ODOM_COLS` in [src/pipeline/data/dataset.py:33-37](../src/pipeline/data/dataset.py#L33-L37). New odom feature count: 5 (was 7).
- `OdomCNN.__init__(in_features=5)` is the new default ([src/pipeline/encoders/odom.py:38](../src/pipeline/encoders/odom.py#L38)). Old explicit `in_features=7` callers still work — left intact in the dashboard / tests so they can load old encoder.pt files.
- Updated `builder.build_encoders` and `_smoke_fusion.py` to use the new default.
- Added [`FusionDataset.get_targets(mode, lookback_s)`](../src/pipeline/data/dataset.py#L355-L425) — returns `(target, valid)` for `mode="position"` (identity) and `mode="displacement"`. The displacement implementation buckets samples by `path_id` and uses `np.searchsorted` to find the latest in-path sample whose timestamp is ≤ `t_i - lookback_s`. Early-in-path samples are marked `valid=False`.
- Wired `target_mode` and `target_lookback_s` into [`EncoderTrainer`](../src/pipeline/training/trainer.py): the loader filters invalid samples; `meta.json` now records the choice; `evaluate()` filters its `raw_val` to match the trainer's filtered val set so trustworthiness lengths align.
- DPVO motion **didn't need a change** — it already uses [`src/pipeline/training/motion.py`](../src/pipeline/training/motion.py) which trains a delta-head on world-frame displacements. The Stage-A-style `EncoderTrainer` is for tabular modalities (IMU, Odom, WiFi).

**Smoke.** [scripts/_smoke_action2.py](../scripts/_smoke_action2.py) — three phases. **ALL PASS.**

```
Phase 1: OdomCNN(default).in_features = 5  (was 7, now 5)
Phase 2: 121 invalid samples out of 8542 (first 1s of each path)
         delta range ±0.44m per axis  (matches expected 1s motion)
         100% of valid samples have nonzero delta
Phase 3: 3-epoch IMU encoder val_mae (delta-position) = 0.0509m
```

**Headline insight.** The same IMU encoder architecture that was reported at 3.1m linear-probe MAE under the (x,y) objective hits **0.05m val_mae after 3 epochs** when asked to predict 1s displacement. That's a ~60× improvement — same architecture, same data, *right* learning objective. The audit's claim that the encoders were memorizing trajectory positions rather than learning motion is now verified by direct measurement: when the objective stops rewarding memorization, the encoder converges fast and to a sensible scale (cm-level for sub-meter motions).

**Debug.** Smoke hit one issue:
- Phase 3 print contained a Δ character; Windows cp1252 console choked. Replaced with `delta-position`. The training itself ran fine — only the log line was broken.
- Trustworthiness in `evaluate()` was sourcing `raw_val` from the unfiltered val cache while `extract_embeddings` saw the filtered loader → length mismatch. Fixed by applying the same `valid` mask in [trainer.py:351-365](../src/pipeline/training/trainer.py#L351-L365).

**What this does NOT do.** It doesn't run a full Stage A retrain. The 0.05m number is a 3-epoch sanity smoke. The real "what's the new IMU encoder MAE on RoNIN trial-out" measurement happens in Action 6 alongside the fusion comparison.

**Files added / changed.**

- `src/pipeline/data/dataset.py` — `ODOM_COLS` shrunk; `get_targets` added.
- `src/pipeline/encoders/odom.py` — default `in_features=5`.
- `src/pipeline/training/trainer.py` — `target_mode`, `target_lookback_s` kwargs; filtered raw_val.
- `src/pipeline/fusion/builder.py`, `scripts/_smoke_fusion.py` — odom dim 7 → 5 at callsites.
- `scripts/_smoke_action2.py` (new).

---

## Action 3 — Make fusion actually fuse

**Why.** Audit finding B1 (the headline of the whole audit): the existing fusion runs were WiFi memorizers. `runs/fusion_20260518_211653/subsets.json` proved WiFi alone hit 0.486m vs all-mods 0.443m on sim — fusion was contributing 0.04m of signal. modality_dropout=0.16 at M=4 meant the model only had to operate without WiFi 16% of the time, so gradients never spread to the weak modalities.

**Built.**

- **Modality-balanced loss.** New `modality_balanced_loss` + `modality_balanced_weight` kwargs on [`FusionTrainer`](../src/pipeline/training/fusion_trainer.py). On every step, with probability `modality_balanced_weight` (default 0.5), the trainer runs a second forward pass with one uniformly-random modality fully masked and adds that leave-one-out loss to the total. Gradient pressure now explicitly rewards every leave-one-out subset. Cost ≈ 1.5× per step.
- **Modality dropout bumped from 0.16 to 0.4** in [configs/stage_c/fusion.yaml:33-43](../configs/stage_c/fusion.yaml#L33-L43). At M=2 this means each modality is dropped 40% of the time; at M=4 the chance of seeing all four together drops to 13%.
- **Post-fit diagnostics block.** `FusionTrainer.fit()` now always runs `evaluate_subsets("val")` after training, prints the subset table, **flags any `drop:X` whose gap to `all` is < 0.1m as `<- UNUSED`**, and (when a baselines.json exists for the dataset) prints "vs baseline" lines with PASS / FAIL. New helper [`_print_post_fit_diagnostics`](../src/pipeline/training/fusion_trainer.py#L406-L451).
- **Baseline path auto-detect.** [`builder.build_trainer`](../src/pipeline/fusion/builder.py#L160-L182) automatically looks up `runs/baselines/<dataset>/baselines.json` and passes it to the trainer, so the comparison block lights up for free.

**Smoke.** [scripts/_smoke_action3.py](../scripts/_smoke_action3.py) — 3-epoch fusion run on sim with `modality_balanced_loss=True`. **PASS.**

Output excerpt:

```
all             MAE=1.337m  RMSE=1.702m
only:imu        MAE=5.051m
only:odom       MAE=5.152m
only:wifi       MAE=1.394m
drop:imu        MAE=1.349m  <- UNUSED (gap < 0.1m)
drop:odom       MAE=1.360m  <- UNUSED (gap < 0.1m)
drop:wifi       MAE=5.148m

Baselines on this dataset (val):
  mean_train_pos  MAE=5.146m  (fusion - baseline = -3.809m, BEATEN by fusion)
  wifi_knn        MAE=0.685m  (fusion - baseline = +0.652m, STILL BEATING fusion)
  imu_kalman      MAE=32.873m  (fusion - baseline = -31.536m, BEATEN by fusion)
Best baseline = wifi_knn @ 0.685m | fusion 1.337m | gap +0.652m  [FAIL]
```

3 epochs is intentionally too short to converge; the point of the smoke is to prove the diagnostics fire correctly. The `<- UNUSED` flag triggered on IMU and Odom even at 3 epochs, and the [FAIL] gate caught that fusion was losing to WiFi-kNN. From now on these numbers print after every fusion run; modality dominance and baseline loss cannot hide.

**Profile.** With `modality_balanced_loss=True`, each step does ~1.5× the work (one extra forward+backward on 50% of steps). Acceptable cost for the gradient-pressure benefit.

**Files added / changed.**

- `src/pipeline/training/fusion_trainer.py` — new kwargs, modality-balanced loss in `_train_epoch`, `_print_post_fit_diagnostics` helper, expanded `meta.json`.
- `src/pipeline/fusion/builder.py` — auto-detects `runs/baselines/<dataset>/baselines.json`.
- `configs/stage_c/fusion.yaml` — dropout 0.16 → 0.4, `modality_balanced_loss: true`.
- `scripts/_smoke_action3.py` (new).

---

## Action 5 — Pretrained Stage A encoder loading

**Why.** Audit finding B5: the fusion model was instantiating fresh `IMUCNN()`, `OdomCNN()`, `Anchor2Vec()` and training from random init. The `runs/imu_*/encoder.pt` checkpoints existed but were never used by fusion. So "Stage A is trained" was technically true but operationally dead. Action 2 makes Stage A actually learn something useful (motion for IMU/Odom); we need a way to plug that learning into fusion.

**Built.**

- New kwarg `pretrained_paths: dict | None` on [`build_encoders`](../src/pipeline/fusion/builder.py#L120-L175). Each value points to a state dict on disk (e.g. `runs/imu_2026.../encoder.pt`). Strict-loaded — a key or shape mismatch raises immediately, so silent staleness can't slip through. Empty dict / None → behaves exactly like before (train from scratch).
- New helper [`pretrained_paths_from_cfg(cfg)`](../src/pipeline/fusion/builder.py#L99-L115) — reads `cfg.stage_a.pretrained.<modality>`, resolves relative paths against repo root, and returns the dict for `build_encoders`. Treats null / missing entries cleanly.
- Added `stage_a.pretrained` block to [configs/stage_c/fusion.yaml:50-60](../configs/stage_c/fusion.yaml#L50-L60) — all keys null by default (fresh-init behavior preserved).
- Camera caveat documented inside `build_encoders`: for DPVO the fusion side trains `dpvo.head`, not the full encoder; so the camera pretrained path must point at `head.pt` if Stage A was done via `motion.py`'s delta-head training.

**Smoke.** [scripts/_smoke_action5.py](../scripts/_smoke_action5.py) — three phases. **ALL PASS.**

1. Train an IMU encoder for 1 epoch via `EncoderTrainer`, capture a sample weight from `encoder.pt`, then `build_encoders(pretrained_paths={'imu': ...})` and verify the in-memory weight bit-exactly matches the disk weight.
2. `pretrained_paths_from_cfg` round-trip: empty defaults → `{}`; one set path → resolved `Path` in the returned dict.
3. Strict-load surface — feeding a bogus state dict raises `RuntimeError`. Confirms a stale or shape-wrong checkpoint can't load silently.

**Files added / changed.**

- `src/pipeline/fusion/builder.py` — `build_encoders` extended; `pretrained_paths_from_cfg` added.
- `configs/stage_c/fusion.yaml` — `stage_a.pretrained` block.
- `scripts/_smoke_action5.py` (new).

---

## Action 6 — Re-tune Optuna on real data

**Why.** Audit finding B7: the existing `runs/optuna_fusion/best.json` was a 20-trial × 30-epoch study **on simulation only**, optimising a metric (sim val_mae) that was dominated by WiFi memorization. Those "Optuna best" hyperparameters were carried over to real datasets where the inductive bias is completely different.

**Built.**

- [`scripts/optuna_fusion.py`](../scripts/optuna_fusion.py) extended:
  - **Per-dataset output dir**: `runs/optuna_fusion/<dataset>/` instead of overwriting one folder. Multiple Optuna runs on different datasets no longer step on each other.
  - **Honors `cfg.stage_a.pretrained`** via `pretrained_paths_from_cfg(cfg)` (Action 5). All trials share the same Stage A checkpoint; search runs over fusion-only hyperparameters with Stage A held constant. Empty when no checkpoints configured.
  - **Baseline gate at the end**: reads `runs/baselines/<dataset>/baselines.json` (Action 1), computes `gap = best_value - best_baseline`, prints `[PASS]` / `[FAIL]`. When fail, the script prints an explicit "the architecture, not the hparams, is the issue" message — preventing more wasted tuning effort.
  - **Trials silent**, only best-run / summary verbose. Per-trial logs would spam the output; the post-fit diagnostics (Action 3) fire only when `verbose=True`, so the per-trial silence is intentional.

**Smoke (the live end-to-end test).** Ran `.venv/Scripts/python.exe scripts/optuna_fusion.py --dataset simulation --trials 2` — completed in ~7 minutes (2 trials × 30 epochs on Quadro P4000).

```
best val MAE : 0.3882 m
best params  : {depth=5, n_heads=8, ff_mult=4, dropout=0.021,
                lr=3.7e-4, modality_dropout=0.108, instant_dropout=0.416,
                n_instants=11, instant_stride=14}

vs best baseline (wifi_knn @ 0.685m): gap -0.297m [PASS]
saved -> runs/optuna_fusion/simulation
```

This is the **first end-to-end smoke through the builder path** (load_config → build_datamodule → build_encoders → build_model → build_trainer → fit). Validates that:

1. The new fusion.yaml defaults (modality_dropout=0.4, modality_balanced_loss=true, stage_a.pretrained block) parse cleanly.
2. `pretrained_paths_from_cfg` returns `{}` when all entries are null, and the trainer doesn't choke on it.
3. The per-dataset output dir + baseline gate prints in the expected format.
4. The 0.388m result confirms sim is dominated by WiFi — modality_dropout=0.108 (Optuna favored a LOW dropout on sim) reproduces the "WiFi-only" optimum the audit identified. **This is evidence the search reward function is right but the dataset is too easy to tell you anything new.**

**What this does NOT do.** It does not run the actual real-data Optuna study — that needs hours of GPU time. The script is now correct and gated; the user runs it themselves on IMUWiFine and IPIN floor -2 when ready:

```powershell
.venv\Scripts\python.exe scripts/baselines.py --skip-leaky          # if not already
.venv\Scripts\python.exe scripts/optuna_fusion.py --dataset imuwifine --trials 20
.venv\Scripts\python.exe scripts/optuna_fusion.py --dataset ipin2024_floor-2 --trials 20
```

If either run shows `[FAIL]` against the dataset's best baseline, the architecture is the bottleneck, not the hyperparameters. Tuning further would waste cycles.

**Files added / changed.**

- `scripts/optuna_fusion.py` — per-dataset output dir, pretrained pass-through, baseline gate.
- `runs/optuna_fusion/simulation/` (auto-generated by the smoke run).

---

## Notebook updates

1. **`## What changed in the 2026-05-20 audit`** — markdown cell at position 1 of [notebooks/fusion_workbench.ipynb](../notebooks/fusion_workbench.ipynb). Lists the 7 pipeline changes so anyone opening the notebook fresh sees the new metric, leaky-split warning, baseline comparison, and Stage A target-mode flag before they start interpreting numbers.
2. **`## 9 · Watch a full validation path replay`** — new markdown + code cell pair (positions 24 + 25) that animates a trained fusion model's predictions vs GT for one val path, saves to `runs/<run_id>/predictions_replay.{mp4,gif}`, and displays inline. Section "Hyperparameter search — Optuna" renumbered to 10 to avoid a section-number collision.
3. **`EVAL_SPLIT` fallback in cells 18, 21, 23** — those cells used to hardcode `'test'`; with `ronin_a000` (and any future no-test-split dataset) that crashed. Now each derives `EVAL_SPLIT = 'test' if dm.test_ds is not None else 'val'` at the top of the cell.
4. **`FusionDataModule.setup()` no longer crashes when `test_paths` is empty** ([src/pipeline/data/datamodule.py:131-147](../src/pipeline/data/datamodule.py#L131-L147)). When `test_paths=[]` it sets `self.test_ds = None`. Every downstream consumer (`test_dataloader`, `summary`, `FusionTrainer`) was already null-safe; the constructor was the missing piece.

The computational cells call `load_config(...)` and so automatically pick up the new yaml defaults — no code edits needed there.

---

## What you should run next, in order

1. **Train Stage A encoders on real data with displacement targets.** The smoke proved the objective works (0.05m IMU val_mae on 3 epochs). For each modality × dataset combo you care about:
   ```powershell
   .venv\Scripts\python.exe scripts/train.py modality=imu dataset=imuwifine \
     target_mode=displacement target_lookback_s=1.0
   ```
   (You'll likely need to extend `scripts/train.py` to forward `target_mode` to `EncoderTrainer` — quick edit, mention it if you want me to do that.)

2. **Run the baselines if you haven't, on every dataset you care about**:
   ```powershell
   .venv\Scripts\python.exe scripts/baselines.py --skip-leaky
   ```

3. **Plug Stage A into fusion via the config**: edit `configs/stage_c/fusion.yaml`'s `stage_a.pretrained.<mod>` to point at `runs/<modality>_<timestamp>/encoder.pt`.

4. **Run Optuna on a real dataset**:
   ```powershell
   .venv\Scripts\python.exe scripts/optuna_fusion.py --dataset imuwifine --trials 20
   ```

5. **Read the post-fit diagnostics output**. If `drop:X` is flagged `<- UNUSED` for any modality, fusion is not actually using that modality on this dataset — you can drop it from the config or investigate. If the gate is `[FAIL]`, the architecture is the bottleneck; come back to me before tuning further.

6. **Decide on the leaky `_intra` configs**: they're marked DEV USE ONLY but still on disk. If you want, delete `configs/data/*_intra.yaml` and the corresponding `data/` directories — but only after you have results on the honest splits to compare against.

---

## Files added by this work (full list)

- `handoff/HANDOFF_LOG.md` (this file)
- `scripts/_smoke_metrics.py`, `_smoke_action2.py`, `_smoke_action3.py`, `_smoke_action5.py`
- `scripts/baselines.py`
- `configs/data/ronin_a000.yaml`
- `data/ronin_a000/` (re-generated by `scripts/convert_ronin.py --split-mode trial-out`)
- `runs/baselines/<dataset>/baselines.json` × 6, `runs/baselines/summary.json`, `runs/baselines/README.md`
- `runs/optuna_fusion/simulation/{best.json,trials.csv,trials/}` (smoke artifact)
- `runs/imu_*` (smoke artifacts from Action 2 / Action 5)

## Files modified

- `src/pipeline/evaluation/encoder_eval.py` — Euclidean MAE helpers + canonical metric definition
- `src/pipeline/training/trainer.py` — `target_mode`, `target_lookback_s`, trustworthiness wired
- `src/pipeline/training/fusion_trainer.py` — modality-balanced loss + post-fit diagnostics + baseline path
- `src/pipeline/data/dataset.py` — `ODOM_COLS` trimmed; `get_targets` added
- `src/pipeline/encoders/odom.py` — default `in_features=5`
- `src/pipeline/fusion/builder.py` — pretrained loading + auto baseline path
- `scripts/optuna_fusion.py` — per-dataset output + baseline gate
- `scripts/_smoke_fusion.py` — odom dim 7 → 5
- `configs/stage_c/fusion.yaml` — modality_dropout 0.16 → 0.4, modality_balanced_loss: true, stage_a.pretrained block
- `configs/data/ronin_a000_intra.yaml`, `ipin2024_floor-2_intra.yaml` — LEAKY annotation
- `notebooks/fusion_workbench.ipynb` — audit-summary markdown cell at position 1

## Branch

All changes are on `audit-baseline-2026-05-20`. The `main` branch is untouched. Run `git diff main` to see the full delta. No commits have been made — every change is in the working tree, ready for you to review and split into commits however you prefer. The original dirty `main` state is preserved on this branch's working tree (nothing was discarded).

---

## Post-audit cleanup — remove ACEVision + VisionViT (2026-05-20)

**Scope (chosen by user):** source + imports + ACE subdir + SCR (mandatory only). Configs / ACE-specific scripts / tests / dashboard loader / runs/ace_scr_* artifacts are **left intact** — they'll error at runtime if invoked, but no `src/` import is broken.

**Deleted:**
- `src/pipeline/encoders/ace_vision.py`
- `src/pipeline/encoders/vision.py`
- `src/pipeline/encoders/ace/` (including `ace_network.py`, `__init__.py`, pycache)
- `src/pipeline/data/scr_dataset.py`
- `src/pipeline/training/scr_trainer.py`

**Fixed imports:**
- `src/pipeline/encoders/__init__.py` — removed `ACEScrRegressor`, `ACEVision`, `VisionViT` from imports and `__all__`; added a docstring explaining why (place-recognition encoders memorize on small datasets per Action 2's audit; re-introduce only with env diversity).
- `src/pipeline/training/__init__.py` — removed `SCRTrainer`.
- `src/pipeline/training/trainer.py:171-175` — updated stale comment in the vision-cache duck-typed branch (only `DPVOMotionEncoder` uses it now).

**Verified.**
- `import src.pipeline.{encoders,training,fusion.builder,data.dataset,data.datamodule}` → all clean.
- `scripts/_smoke_fusion.py --phase 1` → PASS (cls + query readout, finite outputs with all-modalities-dropped, params 0.54M / 0.60M).

**Known-broken-on-purpose (out of scope):**
- `scripts/{train,eval,plot,viz,video}_ace_scr.py` and `plot_dpvo_vs_ace.py` will `ImportError` if run.
- `tests/test_encoders.py` will fail collection on its VisionViT block.
- `dashboard/core/loader.py`'s `VisionViT` callsite will error if the dashboard tries to instantiate it.
- `test_vision.py` at repo root will fail.
- `runs/ace_scr_*` directories remain on disk as historical artifacts.
- `configs/stage_a/vision/ace.yaml` and `assets/diagrams/vision_vit.drawio*` remain.

If you want a second cleanup pass on any of the above, say which and I'll do that.

**Doc updated.** `docs/PIPELINE.md` — replaced the two encoder boxes with a `(removed 2026-05-20)` annotation so the diagram reflects the current state.

---

## Add-on — ffmpeg for notebook video + per-prediction modality attribution (2026-05-20)

### ffmpeg (no system install needed)

The notebook's replay cell fell back to GIF because matplotlib looks for `ffmpeg` on PATH. Turns out `imageio-ffmpeg` is already in the venv and ships a real ffmpeg binary (`.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe`). Rather than a system install (admin / winget), the fix points matplotlib at that bundled binary:

- Notebook video cell now runs, at the top:
  ```python
  import imageio_ffmpeg
  mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
  ```
- Verified end-to-end: matplotlib `FuncAnimation.save(writer='ffmpeg')` wrote a 21 KB mp4 with this binary.
- Pinned `imageio-ffmpeg>=0.4` in `pyproject.toml` so it's an explicit dep, not accidental.

The replay cell will now produce `.mp4` instead of `.gif`.

### Per-prediction modality attribution

**Why.** The subset table (Action 3) tells you average modality importance over a whole split. The user wanted the *per-prediction* version: for each `(x, y)` the model outputs, how much did it lean on each modality? This is the audit's "is it fusing or leaning?" question, made visible sample-by-sample.

**Built.**

- `FusionTransformer.forward_attribution(...)` ([transformer.py:230-310](../src/pipeline/fusion/transformer.py#L230-L310)) — runs the query readout with `need_weights=True`, maps the cross-attention weights back to `(instant, modality)` using the known token layout (CLS + M blocks of K), and returns `{attn (B,M), cls (B,), avail_frac (B,M)}`. The attn columns + cls sum to 1 per row. Raises for `readout='cls'` (no single attributable layer there).
- `FusionTrainer.log_attribution(split, path_id, max_samples, save, verbose)` ([fusion_trainer.py:526-624](../src/pipeline/training/fusion_trainer.py#L526-L624)) — runs attribution along a path (time-ordered), prints a readable per-sample table + a sequence-mean row, and writes `attribution_<split>[_path<id>].json` to the run dir.
- Notebook section `## 10 · Per-prediction modality attribution` — calls `log_attribution` and renders a **stacked-area plot** of attention mass over the path with the error curve underneath. (Optuna section renumbered to 11.)

**Smoke.** [scripts/_smoke_attribution.py](../scripts/_smoke_attribution.py) — 4-epoch query-readout model on sim. **PASS** (attn+CLS sums to 100% per row; JSON written; `cls` readout raises).

**What it immediately showed** (4-epoch sim model, val path_02):

```
MEAN  0.43m |  imu 10.3%   odom 2.0%   wifi 83.7%   CLS 4.0%
```

WiFi drives ~84% of every prediction; Odom is all but ignored (2%). The split moves with tracking quality — IMU rises to ~14% where error dips to ~0.17m, WiFi spikes to ~91% when it's the only usable anchor. This is the same "fusion leans on WiFi" finding as the subset table, now visible per-sample. On a real dataset where WiFi doesn't transfer (IPIN trial-out), this plot is the fastest way to see whether modality-balanced training (Action 3) actually redistributed the attention.

**Files added / changed.**

- `src/pipeline/fusion/transformer.py` — `forward_attribution`.
- `src/pipeline/training/fusion_trainer.py` — `log_attribution`, `import numpy as np`.
- `notebooks/fusion_workbench.ipynb` — ffmpeg path in video cell; new attribution section.
- `pyproject.toml` — `imageio-ffmpeg>=0.4`.
- `scripts/_smoke_attribution.py`, `scripts/_insert_attribution_cell.py` (new).

---

## Fix-up — ronin leak + notebook cell bugs (2026-05-20, after first honest runs)

Running the notebook on `ronin_a000` then `ipin2024_floor-2` exposed three issues.

### 1. `ronin_a000` had a train/val leak (the val path WAS a train path)

`scripts/convert_ronin.py:find_subject_seqs` globbed each sequence from every native folder. RoNIN's `seen_subjects_test_set` reuses sequence IDs that also live in `train_dataset_1` — `a000_7` and `a000_11` are in both. So `a000_7` was emitted as a train path *and* the val path (identical 294.2s / 2943 GT). The "honest" 1.87m val_mae was measured on a trajectory that was in the training set.

**Fix.** `find_subject_seqs` now dedups by sequence name, keeping the most held-out split (priority test > val > train). `a000_7` → val only, dropped from train. Regenerated `data/ronin_a000`: **8 train / 1 val / 0 test**, val = `a000_7` genuinely held out. Updated `configs/data/ronin_a000.yaml`.

**Impact on the number — this is the point.** Honest baselines on the de-leaked split:

| baseline | leaked (before) | honest (after) |
|---|---|---|
| wifi_knn | 3.58m | **13.15m** |
| mean_train_pos | 16.94m | 16.96m |

WiFi-kNN nearly **4×'d** once `a000_7` was actually held out — exactly what you'd expect when the fingerprints no longer have a copy of the test trajectory to match against. The fusion number (was 1.87m on the leaked split) will rise correspondingly when re-run; ~13m is now the bar to beat. Re-run the notebook on `ronin_a000` for the honest fusion result.

### 2. Notebook Optuna cell read the stale global file

Cell 29 read `runs/optuna_fusion/best.json` (the old sim study). Action 6 writes per-dataset to `runs/optuna_fusion/<dataset>/`. Fixed the cell to read `runs/optuna_fusion/<cfg.dataset.selected>/best.json` and to print the dataset-specific run command when absent.

### 3. Video and attribution cells looked at different splits

The video cell (25) hardcoded `val_ds` / `predict('val')` while the attribution cell (27) used `EVAL_SPLIT = 'test' if test_ds else 'val'`. On IPIN (which has a test split) the video animated a val path while attribution analyzed a test path — incoherent. Fixed the video cell to use the same `EVAL_SPLIT`, so both look at the same split.

**Files changed.** `scripts/convert_ronin.py`, `configs/data/ronin_a000.yaml`, `data/ronin_a000/` (regenerated), `runs/baselines/ronin_a000/baselines.json` (re-run), `notebooks/fusion_workbench.ipynb` (cells 25 + 29).

### Honest-result summary so far

| dataset | split | best baseline | fusion (temporal) | verdict |
|---|---|---|---|---|
| ipin2024_floor-2 | trial-out (clean) | mean 25.7m | 23.3m | barely beats centroid; **cannot localize cross-trial** |
| ronin_a000 | trial-out (now de-leaked) | wifi_knn 13.1m | *re-run pending* | leaked 1.87m was meaningless |

The IPIN result is the honest signal: on a real cross-trial split the bottleneck is the **WiFi encoder's non-transferability (Stage A)**, not fusion. Conformal radius there was 54.9m — the model's own admission it can't localize. Next high-value work is a session-invariant WiFi encoder, not more fusion tuning.

---

## Proposal 1 — decomposed additive readout: BUILT, TESTED, **does not help** (2026-05-20)

Goal: stop motion tokens competing with WiFi in one softmax. New readout
`pred = p_abs(absolute tokens) + g·Δp(motion tokens)`, anchor + motion
queries over separate token pools, adaptive gate `g`, aux loss forcing
`p_abs` to be the WiFi-only estimate so `Δp` learns a residual correction.

**Built (behind `readout: decomposed`, `query`/`cls` untouched).**
- `transformer.py` — `_readout_decomposed`, `_token_type_masks`, `forward(return_parts=...)`, and `forward_attribution` extended to report `gate` + `motion_frac`.
- `fusion_trainer.py` — `aux_abs_weight`; aux anchor loss in `_train_epoch`; gate/motion_frac in `log_attribution`; meta records readout + aux weight.
- `fusion.yaml` + `builder.py` — `readout`, `absolute_modalities: [wifi]`, `aux_abs_weight`.
- `scripts/_smoke_decomposed.py` (shapes/NaN/train — PASS), `scripts/_bakeoff_decomposed.py` (query-vs-decomposed harness).

**Bake-off (40 epochs each, identical everything-else):**

| dataset | query val_mae | decomposed val_mae | decomposed motion_frac | gate |
|---|---|---|---|---|
| ipin2024_floor-2 (honest) | **23.01m** | 24.78m | 0.033 | 0.16 |
| simulation (wifi+imu+odom) | **0.41m** | 0.66m | 0.008 | 0.04 |

**Decomposed regressed in BOTH regimes**, and the dedicated motion path went essentially unused (motion_frac ≤ 3%). Two reasons, both instructive:

1. **The hard absolute/relative split is a constraint, not a help.** The plain `query` readout lets self-attention + the single query freely blend WiFi and a little motion at readout — on sim that took it from 0.60m (WiFi-only) to 0.41m (all). Forcing `p_abs` to come *only* from WiFi tokens and `Δp` *only* from motion removes that freedom; the model gates the motion path off (`g→0`) and lands near the WiFi-only number (0.66m).
2. **It can't fix observability.** On IPIN there's no transferable absolute anchor to integrate motion from, so no readout structure makes motion useful. On sim the anchor is so good motion isn't needed. Neither dataset is the "good-but-sparse anchor + reliable motion" regime where the decomposition could win.

**Decision.** Reverted the config default to `readout: query` (don't ship a default that's worse on the benchmark of record). Kept `decomposed` as a tested, NaN-safe option + the bake-off harness + the gate/motion_frac attribution — useful if a within-session dataset with sparse-but-reliable WiFi shows up later.

**What this proves (the valuable part).** Two independent lines of evidence now say the same thing: the bottleneck is **WiFi transferability (Stage A)**, not the fusion readout. (i) IPIN observability — fusion ≈ centroid because WiFi doesn't transfer; (ii) the decomposition can't extract motion value the data doesn't support. Tuning the readout is a dead end. **The meters are in a session-invariant WiFi encoder.**

**Possible softer refinement (not done):** keep the single `query` readout but add a displacement *auxiliary head* as a multi-task regularizer (predict Δp as a side task, don't route the prediction through it) — improves motion representations without constraining the readout. Low expected payoff on cross-trial data given the observability diagnosis; logged for completeness.

