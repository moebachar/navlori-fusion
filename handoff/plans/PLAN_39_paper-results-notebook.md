# PLAN_39 — Paper-results notebook (scoped to scope.md) + WiFi-Net rename + MAE/ATE-only metrics

> **Phase shift.** PLAN_30→38 built `notebooks/run2_walkthrough.ipynb`
> as the full **run-2 archive** notebook (everything we explored:
> 4 modalities × 4 fusion archs × 7 datasets). That archive stays.
> This plan creates a **second, parallel notebook**
> `notebooks/paper_results.ipynb` containing ONLY what's in
> `scope.md` — the conference paper's actual results section.
> The two notebooks coexist: archive = "full story including
> dead ends"; paper-results = "what the reviewers will see".
>
> **Author's constraints quoted verbatim (2026-06-01):**
> 1. "a scriptless notebook (only importation from the src folder,
>    data, and external repos, any adictional code should be in the
>    notebook) that reproduce the results we will put in the paper
>    (SOTA, encoders, ours)".
> 2. "i want to replace the name anchor2vec to wifi-net everywhere
>    on this project (do for notebooks and everywhere)".
> 3. "for now keep metrics menimal to only 2 MAE and ATE".
>
> **Author's intent:** "honest faire notebook for reproducing".
> Honest = limitations stay visible, numbers are live-computed
> not hand-typed, every paper claim has a notebook cell that
> reproduces it. Fair = SOTA baselines run on the same data
> splits as ours, with identical reporting conventions.
>
> **User-approved judgement calls (2026-06-01):**
> - Notebook coexistence: new `paper_results.ipynb` ships alongside
>   the archive `run2_walkthrough.ipynb`. Both remain.
> - Honest framing as **cells**, not markdown disclaimers — the
>   3 limitations (smoothness, IMU canonical gap, MSILN path-130)
>   render live data in `paper_results.ipynb` §6.

---

## 0. Engineer-vs-scientist contract for this plan

- **Engineer:** does two things in order:
  1. **First:** the project-wide `Anchor2Vec → WiFiNet` rename
     (§0a below). This is a prerequisite — the new notebook
     imports the renamed symbol; doing the notebook first and the
     rename second risks merge conflicts.
  2. **Then:** builds `notebooks/paper_results.ipynb` from scratch
     (don't edit the archive notebook). Uses only the APIs listed in §4.
     Treats `scope.md` as the source of truth for what goes in.
  3. Reports a RESULT_39 covering BOTH: rename audit (files
     touched, files skipped + why) + notebook (per-section
     completion, FAST_MODE=True wall-clock, FAST_MODE=False
     wall-clock estimate, any numbers that diverge from
     `scope.md`'s headline table).
- **Scientist:** reviews the result, iterates on prose / table
  shape / honest framing. Has already updated the scientist-
  side artifacts (`scope.md`, `STATE.md`, `CLAUDE.md`, this
  plan) to use `WiFi-Net` / `WiFiNet` — engineer's rename
  audit should not need to touch those.

---

## 0a. PREREQUISITE — `Anchor2Vec → WiFiNet` rename (project-wide)

> **Scope statement (do not exceed).** Rename `Anchor2Vec` (the
> WiFi encoder class) to `WiFiNet` in **live code, active docs,
> active configs, and active notebooks**. Do NOT rewrite
> historical archives (frozen RESULT_NN_*.md, PLAN_NN_*.md for
> N ≤ 38, `handoff/archive/run1/**`, `runs/overnight/run2_iter_*/`
> JSON sidecars). Those stay as historical records; rewriting
> them revises history.

### 0a.1 Naming convention (binding)

| context | symbol |
|---|---|
| Python class | `WiFiNet` (PascalCase) |
| Python function | `train_wifi_net`, `wifi_net_predict` |
| Python file | `wifi.py` keeps name; if engineer prefers, may rename to `wifi_net.py` (judgement call — document either way) |
| Config YAML file | `configs/stage_a/wifi/anchor2vec.yaml` → `configs/stage_a/wifi/wifi_net.yaml` |
| Config YAML key | `wifi_net:` (under whatever the parent is) |
| Display name (markdown / paper prose) | `WiFi-Net` (with hyphen, matching user's text) |
| Display name (figure labels / table headers) | `WiFi-Net` |
| Checkpoint file name | engineer's judgement — see 0a.4 below |

### 0a.2 LIVE files to rename (engineer-visible blast radius)

The rename grep returned 103 files with 532 occurrences. Of those,
**~60 are frozen archive** (`RESULT_NN_*.md`, past `PLAN_NN_*.md`,
`handoff/archive/run1/**`, `runs/overnight/run2_iter_*/*.json`) and
**~43 are live**. Engineer renames the live set:

**Source code (must rename):**
- `src/pipeline/encoders/wifi.py` — class definition
- `src/pipeline/encoders/wifi_set.py` — likely just a comment reference
- `src/pipeline/encoders/__init__.py` — export
- `src/pipeline/training/__init__.py` — export
- `src/pipeline/training/inline_encoders.py` — `train_anchor2vec`, `anchor2vec_predict`
- `src/pipeline/training/trainer.py`
- `src/pipeline/fusion/builder.py`
- `src/pipeline/fusion/mot_transformer.py`
- `src/pipeline/data/uji.py`
- `src/pipeline/data/webots.py`
- `src/pipeline/evaluation/main_results_table.py`
- `src/pipeline/visualization/_style.py`
- `dashboard/core/loader.py`
- `dashboard/pages/2_Encoders.py`
- `tests/test_encoders.py`

**Configs (must rename including file rename):**
- `configs/stage_a/wifi/anchor2vec.yaml` → `wifi_net.yaml`
- `configs/data/msiln_site1_b1.yaml`
- `configs/config.yaml`

**Scripts (the live ones — must rename references):**
- `scripts/eval_uji.py`
- `scripts/eval_uji_wifi.py`
- `scripts/eval_wlanloc_uji.py`
- `scripts/eval_cnnloc_uji.py`
- `scripts/_train_uji_arch.py`
- `scripts/_smoke_fusion.py`
- `scripts/_smoke_fusion_consolidation.py`
- `scripts/_smoke_evaluation.py`
- `scripts/_eval_uji_setxformer.py`
- `scripts/_eval_uji_6metric.py`
- `scripts/profile_training.py`
- `scripts/gpu_test.py`
- (don't delete any of these — the rename leaves them functional;
  scope.md still defers a future scripts/ audit to journal)

**Notebooks (must rename):**
- `notebooks/run2_walkthrough.ipynb` — the archive notebook (per
  user: "do for notebooks and everywhere")
- `notebooks/encoder_workbench.ipynb` — if it has Anchor2Vec
- `notebooks/_archive/run2_walkthrough_v1_summary.ipynb` —
  **SKIP** (this one IS archive; it's already snapshotted)

**Docs (must rename):**
- `README.md`
- `docs/SOTA_BASELINES.md`
- `docs/fusion_pipeline.md`
- `handoff/SUMMARY.md`
- `handoff/SCIENTIST_BRIEF.md`
- `handoff/SCIENTIST_NOTE_main-results-table.md`
- `handoff/HANDOFF_LOG.md`
- `handoff/fusion-pipeline.md`
- `paper-workspace/scope.md` (if it exists — this is separate from `x:\navlori-fusion\scope.md`)
- `paper-workspace/style-icinco.md`
- `paper-workspace/icinco-2024-relevant.md`

### 0a.3 FROZEN files to LEAVE ALONE (do NOT rename)

Engineer must NOT modify:
- All `handoff/results/RESULT_NN_*.md` (N=01-37). These are
  historical records; renaming them rewrites the iteration log.
- All `handoff/plans/PLAN_NN_*.md` for N=01-38. Same reason.
- `handoff/archive/run1/**`. Pre-run-2 archive.
- `runs/overnight/run2_iter_*/*.json` metrics sidecars. These
  contain saved metrics keyed by the encoder name at that point
  in time; modifying breaks ckpt loading.

**Sanity check before commit:** `git diff --stat` should show
zero changes under `handoff/results/`, `handoff/plans/PLAN_[0-3][0-8]_*`,
`handoff/archive/`, or `runs/`.

### 0a.4 Checkpoint compatibility (judgement call)

PyTorch state_dict keys come from `self.X` *attribute* names,
not the parent class name. So:

- **If engineer renames ONLY the top-level class** (`Anchor2Vec(nn.Module)` →
  `WiFiNet(nn.Module)`) without renaming internal `self.X = ...`
  attributes: old checkpoints **still load** because their
  state_dict keys are unchanged.
- **If engineer renames internal attributes** (e.g.
  `self.anchor2vec_head` → `self.wifi_net_head`): old checkpoints
  **break**. They'd need re-training (~20 min for UJI on
  Quadro P4000).

**Recommendation:** preserve internal attribute names; rename ONLY
the top-level class symbol. Old checkpoints continue to load.
Document the decision in RESULT_39.

**Checkpoint file names on disk:** OPTIONAL rename. Engineer may
either:
- (a) leave `runs/encoder_audit_wifi/anchor2vec_uji.pt` at its
  current path (only the path string in code changes — wait no,
  the path string stays the same too) → zero data move; OR
- (b) `git mv` the .pt file (if tracked) or `mv` (if untracked)
  to `wifi_net_uji.pt` and update path strings everywhere.

Recommendation: do (a) — leave the .pt file alone. Less risky.
Engineer's call; document the choice.

### 0a.5 Acceptance for the rename

- [ ] After rename: `grep -ri "Anchor2Vec\|anchor2vec" src/ notebooks/ configs/ docs/ scripts/ README.md handoff/SUMMARY.md handoff/SCIENTIST_BRIEF.md` returns ZERO hits (case-insensitive).
- [ ] `grep -ri "Anchor2Vec\|anchor2vec" handoff/results/ handoff/plans/PLAN_[0-3][0-8]_* handoff/archive/ runs/` returns the historical hits **unchanged** (their counts should match pre-rename baseline).
- [ ] `tests/test_encoders.py` passes after rename.
- [ ] `python -c "from src.pipeline.encoders import WiFiNet; print(WiFiNet)"` works.
- [ ] `python -c "from src.pipeline.training import train_wifi_net; print(train_wifi_net)"` works.
- [ ] `notebooks/run2_walkthrough.ipynb` (the archive) executes end-to-end with `nbconvert --execute` after the rename (FAST_MODE=True branch).
- [ ] If the engineer chose to keep an alias for backwards compat (e.g. `Anchor2Vec = WiFiNet` in `__init__.py`), that alias is allowed but should print a `DeprecationWarning` and the alias itself is the **only** place `Anchor2Vec` survives in live code. Document in RESULT_39.

### 0a.6 Budget

Mechanical mass find/replace work; ~30-45 min engineer time
including verification of the test + archive notebook.

---

## 1. Hypothesis / Goal

The paper-results notebook ships in 1 iteration if we:

1. **Reuse** the archive notebook's working cells (the WiFi-on-UJI,
   IMU-on-RoNIN, MSILN-transformer cells *already work* — they just
   coexist with cells we now drop).
2. **Drop** everything outside the scope.md slice (Camera, Odom,
   CNN1D / LSTM-attn / MoTTransformer, IPIN, IMUWiFine, TartanAir,
   4-mod Webots).
3. **Add** 4 explicit ablation cells that the paper's §6 needs
   (staleness, K-axis sweep, modality-dropout, latency).
4. **Add** an explicit §7 honest-limitations cell pulling live
   numbers from saved RESULTs.

**No new training is required** — every paper checkpoint exists on
disk per the survey: WiFi/UJI, IMU/RoNIN canonical, MSILN
transformer, UJI transformer, RoNIN transformer, Webots 4-mod
transformer (we'll re-use as 2-mod by reading only WiFi+IMU
encoders at eval time; if that doesn't work cleanly, the
Webots 2-mod RESULT_06 ckpt lives under
`runs/overnight/run2_iter_06/` — engineer to confirm path).

---

## 2. Out of scope (do NOT include in `paper_results.ipynb`)

These belong only to the archive notebook:

- Camera modality (DPVO, ACEVision, VisionViT)
- Odometry modality (OdomCNN)
- TartanAir hospital dataset, TartanVO SOTA
- IMUWiFine dataset (any floor)
- IPIN 2024 dataset (any floor)
- CNN1D / LSTM-attn / MoTTransformer fusion architectures
- 4-modality Webots fusion (the per-paper Webots is 2-mod = WiFi+IMU only)
- Phase B winner discussion (CNN1D 0.339 m winner is out-of-paper-scope)
- 4-architecture bake-off tables (Table B in archive)
- Per-iteration RESULT-source mentions ("RESULT_05 found...")
  — paper artifacts don't reference the archive's iteration log

If a cell touches any of the above, it doesn't belong in this notebook.

---

## 3. Notebook file + naming

- **Path:** `notebooks/paper_results.ipynb`
- **Title cell:** "NavLoRI-Fusion — Paper Results Reproducibility
  Notebook" + 3-sentence abstract pointing at the contribution
  in §1 of `scope.md`.
- **FAST_MODE flag:** identical convention to the archive
  notebook — `FAST_MODE = True` loads saved checkpoints + runs
  eval live (~10 min on Quadro P4000); `FAST_MODE = False`
  retrains every cell inline using the `src.pipeline.training.*`
  APIs (~3 h). The flag is set in §0 setup and every train cell
  guards with `if FAST_MODE: load_trained(...) else: train_*(...)`.

---

## 3a. METRICS POLICY (binding for this notebook + paper)

> **Author's constraint (2026-06-01):** "for now keep metrics
> menimal to only 2 MAE and ATE".

### 3a.1 The two metrics

- **MAE = mean position Euclidean error** in meters.
  Used for: all WiFi-based localization (UJI, MSILN), all
  position-fusion endpoints (Webots, MSILN), all ablations
  (K-axis, modality dropout, staleness).
  Formula: `MAE = mean(‖pred_xy - gt_xy‖₂)` over the eval
  split. One number per (dataset, split, method).
- **ATE = absolute trajectory error** in meters.
  Used for: IMU dead-reckoning (RoNIN canonical).
  Two flavors reported in the limitations section:
  - **Raw ATE** (no alignment) — the primary headline number.
  - **Umeyama-aligned ATE** — reported only in §6 limitations
    as honest context (it narrows the IMU canonical gap from
    +94 % to +53 %; both numbers shown, neither cherry-picked).

### 3a.2 Dropped metrics (NOT in the paper, NOT in the notebook)

The notebook does **not** display:
- ❌ RMSE (any flavor)
- ❌ Linear-probe vs kNN-probe distinction (one MAE per method)
- ❌ Median Euclidean (only mean)
- ❌ Trustworthiness / alignment-uniformity / effective-dim
  (the 6-metric Stage-A encoder harness is out of scope)
- ❌ Per-modality cross-subset MAE tables (the
  `evaluate_all_subsets` output) — visible internally for the
  modality-dropout cell, but reported as a single bar chart per
  subset, not as a table of cross-subset numbers
- ❌ **Pearson r for smoothness** — the smoothness debt limitation
  cell uses a **visual GT-vs-pred trajectory overlay** to make
  the over-smoothing visible, not a Pearson-r number. Visual
  proof, not a quantified metric (deferred to journal alongside
  the smoothness lever experiment).

### 3a.3 Underlying API behavior (NOT changed)

`src.pipeline.training.FusionTrainer.evaluate_subsets(...)`,
`evaluate_staleness(...)`, etc. may still internally compute
RMSE / median / count — the notebook just **doesn't display
them**. No `src/` code changes needed for the metric policy.
This is purely a reporting-surface policy for the paper notebook.

### 3a.4 Latency is NOT a metric — it's a measurement

The latency probe (§5 cell 24) reports `ms/sample at b=1` and
`ms/sample at b=32`. This is a timing measurement, separate
category from error metrics. It stays.

---

---

## 4. Allowed imports (the "scriptless" rule, made concrete)

The notebook may import ONLY from:

- **Standard library**: `pathlib`, `json`, `time`, `dataclasses`,
  `typing`, `os`, `sys`
- **Numeric**: `numpy`, `pandas`, `torch`, `matplotlib`
- **Project source**: `src.pipeline.*` per the survey (post-rename names):
  - `src.pipeline.data` — `load_dataset`, `dataset_stats`,
    `FusionDataModule`, `MODALITY_DIMS`
  - `src.pipeline.encoders` — `WiFiNet`, `IMUCNN`
  - `src.pipeline.baselines` — `load_position_regressor` (wlan_localization),
    `ResNet1D` + `load_test_list` + `compute_ate_rte` (RoNIN)
  - `src.pipeline.training` — `train_wifi_net` + `wifi_net_predict`,
    `train_imucnn`, `train_fusion_arch`, `train_uji_arch`,
    `train_ronin_canonical_arch`, `load_trained`
  - `src.pipeline.fusion` — `build_arch` (only `'transformer'`)
  - `src.pipeline.evaluation` — `MainResultsTable`, eval helpers
    (linear_probe, knn_probe, ATE/RTE computation)
  - `src.pipeline.visualization` — `set_paper_style`,
    `plot_trajectory_comparison`, `plot_staleness_curve`,
    `plot_subset_eval_bar`, `COLOR_PALETTE`
- **External SOTA submodules**: `external_methods/wlan_localization`
  and `external_methods/ronin` (via the
  `src.pipeline.baselines._shims` apply-shims protocol; the
  notebook does NOT itself patch numpy / scipy).

**Forbidden imports / calls:**
- ❌ `subprocess.run(["python", "scripts/_train_*.py", ...])`
- ❌ Any `from scripts.* import ...`
- ❌ Hand-typed paper numbers in markdown ("8.69 m" must be
  rendered from a live variable, never a string literal in prose).
- ❌ Any reference to `RESULT_NN_*.md` files in markdown
  (these are internal archive artifacts).

**Inline code IS allowed in the notebook** for:
- Result formatting (table assembly, mean ± std computation)
- Custom plot polish (figure-format tweaks for PerCom 2-column)
- Small glue between SOTA-eval and our-eval (e.g. align
  prediction tensors, compute deltas)

The rule is: anything that's a paper-results computation lives
in `src/pipeline/*` OR in the notebook. Nothing in `scripts/`.

---

## 5. Cell-by-cell structure

8 sections, ~30-35 cells total. Mirror the paper structure exactly
so each cell maps to a figure or a table in the .tex.

### §0 — Setup + FAST_MODE + paper-style

Cells:
1. **Markdown** — title, 3-sentence abstract, link to `scope.md`,
   FAST_MODE explanation, expected wall-clock per branch.
2. **Code** — config: `FAST_MODE = True`, `SEED = 42`,
   `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`,
   `RUN_ROOT = Path("runs/")`.
3. **Code** — imports (per §4); `set_paper_style()`; warn-silence
   for known matplotlib backend warnings.

### §1 — Datasets

Cells:
4. **Markdown** — "4 datasets: 1 simulation + 3 real-world".
   Table-of-contents: Webots / MSILN / UJI / RoNIN canonical.
5. **Code** — `for name in ['simulation', 'msiln_site1_b1',
   'uji_indoor_loc', 'ronin_canonical']: stats =
   dataset_stats(name); display(stats)`. Renders a 4-row pandas
   DataFrame with: #paths-train, #paths-val, #paths-test,
   #wifi-aps-or-n/a, sensor-rate-imu, sensor-rate-wifi.
6. **Code** — `plot_gt_trajectory(name)` × 4 datasets, one
   subplot each (one figure with 4 panels). Aspect-ratio-fixed
   per the IMUWiFine bug fix (RESULT_35) — even though IMUWiFine
   is out of scope, the fix carries over via `set_aspect`.

### §2 — Per-leg SOTA baselines

Cells:
7. **Markdown** — "Two SOTAs, one per modality. Both are open-
   source unmodified (Demand #3); runtime shims live in
   `src.pipeline.baselines._shims`, not in vendored source."
8. **Code** — wlan_localization on UJI:
   - load preprocessor + position regressor
   - run on UJI val split
   - report mean Euclidean error (m)
   - **Expected:** ~15.17 m val (RESULT_01 anchor)
9. **Code** — wlan_localization on MSILN site1/B1 cross-session:
   - same wrapper, different dataset feed
   - report val + test mean Euclidean
   - **Expected:** val ~21.26 m / test ~28.31 m (RESULT_37
     anchor)
10. **Code** — RoNIN ResNet1D on canonical unseen-subjects:
    - load `external_methods/ronin` ResNet1D pretrained weights
    - run on 32 canonical test sequences
    - report raw ATE (Umeyama-aligned ATE optional)
    - **Expected:** raw ATE 5.14 m (RESULT_07 / RoNIN paper Table 2)

### §3 — Our per-leg encoders

Cells:
11. **Markdown** — "WiFi-Net (WiFi) + IMUCNN (IMU). Same
    public splits; same metric (MAE / raw ATE) as SOTAs above."
12. **Code** — WiFi-Net on UJI:
    - `if FAST_MODE: load runs/encoder_audit_wifi/anchor2vec_uji.pt`
      (file kept at old path per 0a.4 recommendation;
      `WiFiNet.load_state_dict(...)` works because internal
      attribute names preserved)
      `else: train_wifi_net(X_train, Y_train, X_val, Y_val, k=64, epochs=200)`
    - `pred = wifi_net_predict(enc, head, X_val)`
    - report **MAE** (mean Euclidean error, meters)
    - **Expected:** ~8.69 m val (per scope.md headline);
      **−43 %** vs wlan_localization SOTA
13. **Code** — IMUCNN on RoNIN canonical:
    - `if FAST_MODE: load runs/encoder_audit_imu/imucnn_ronin_canonical.pt
       else: train_imucnn(train_dir, val_dir, epochs=80)`
    - report **raw ATE** (meters); Umeyama-aligned ATE deferred
      to §6 limitations only
    - **Expected:** raw ATE 9.96 m (per scope.md headline);
      honest framing: in-domain competitive, cross-subject
      gap visible
14. **Markdown** — "Per-leg results (paper Table A — MAE for
    WiFi, raw ATE for IMU)":
    auto-rendered pandas DataFrame:

    | dataset | modality | metric | SOTA | Ours | Δ% |
    |---|---|---|---|---|---|
    | UJI | WiFi | MAE | 15.17 | **8.69** | **−43%** |
    | RoNIN canonical | IMU | raw ATE | **5.14** | 9.96 | +94% |

    Bold = better. All values pulled from live variables computed
    in cells 8/10/12/13. No hand-typed numbers.
    Umeyama-aligned ATE NOT in this table — appears only in §6.

### §4 — End-to-end fusion (the headline)

Cells:
15. **Markdown** — Method recap: continuous-time set-transformer
    with `time_encoding(Δt)` + modality_embedding + cross-attention
    PositionQuery readout. K=4 instants × 2 modalities (WiFi+IMU)
    = 8 tokens + 1 query. Encoders: WiFi-Net (WiFi) + IMUCNN (IMU).
16. **Code** — Webots 2-mod K=4 (controlled lab):
    - `if FAST_MODE: load_trained("runs/main_table/simulation/transformer", arch="transformer", dataset="simulation")
       else: train_fusion_arch(arch="transformer", dataset="simulation", K=4, modalities=["wifi","imu"], epochs=90, batch_size=128)`
    - report val + test **MAE** (meters)
    - **Expected:** val 0.469 m / test 0.517 m (per scope.md headline)
    - **NOTE for engineer:** if no 2-mod Webots checkpoint exists,
      retrain the 2-mod ckpt as part of this plan (~20 min);
      the 4-mod ckpt at `runs/overnight/run2_iter_*/` is *not*
      a substitute — paper claims 2-mod.
17. **Code** — MSILN site1/B1 cross-session 2-mod K=4 (headline):
    - `if FAST_MODE: load_trained("runs/main_table/msiln_site1_b1/transformer", arch="transformer", dataset="msiln_site1_b1")
       else: train_fusion_arch(arch="transformer", dataset="msiln_site1_b1", K=4, modalities=["wifi","imu"], epochs=90)`
    - report val + test **MAE** (meters)
    - **Expected:** val 15.22 m / test 10.89 m (per scope.md headline)
    - **Headline:** 28.31 (wlanloc test MAE) → 10.89 (ours test
      MAE) = **−62 %**. Render this delta live from cell-9 and
      cell-17 variables.
18. **Code** — GT-vs-pred trajectory overlay on 2-3 representative
    MSILN test paths (incl. path 130 to make the path-130
    composition visible). Two subplot panels per path:
    (a) GT vs our prediction; (b) GT vs wlan_localization
    prediction. Visual proof of the −62 % gap.
19. **Markdown** — End-to-end results table (paper Table 2 — MAE only):

    | dataset | wlanloc MAE | Ours (transformer 2-mod K=4) MAE | Δ% |
    |---|---|---|---|
    | Webots sim test | n/a (no equivalent open SOTA on Webots) | **0.517** | — |
    | MSILN site1/B1 val | 21.26 | **15.22** | **−28%** |
    | MSILN site1/B1 test ⭐ | 28.31 | **10.89** | **−62%** |

### §5 — Ablations (paper §6)

All ablations run on Webots sim because (a) we control the
ground truth precisely, (b) the controlled lab is where ablations
*should* be done per scope.md §5.

Cells:
20. **Markdown** — "Four ablations: K-axis sweep, modality-dropout
    rate sweep, staleness curve, inference latency."
21. **Code** — K-axis sweep K ∈ {1, 2, 4, 8}:
    - `for K in [1, 2, 4, 8]:`
    - `  if FAST_MODE: load existing K-specific ckpt;`
    - `  else: train_fusion_arch(arch="transformer", dataset="simulation", K=K, ...)`
    - report test **MAE** per K (meters); plot bar chart.
    - **Expected:** K=1/2/4 plateau ~0.47-0.49 m MAE, K=8 regresses
      to ~0.65 m MAE (overfit). Sweet spot K=4.
    - **NOTE for engineer:** K=2 + K=8 may not have a saved
      ckpt — RESULT_11/12 explored K=1/4/8. K=2 may need to be
      trained inline (~20 min). Document in result.
22. **Code** — Modality-dropout rate sweep p_drop ∈ {0.0, 0.2, 0.4, 0.6}:
    - train_fusion_arch with `modality_dropout=p_drop`,
      `instant_dropout=0.45` fixed
    - eval `only:wifi`, `only:imu`, full subsets via
      `trainer.evaluate_all_subsets("test")`
    - plot subset-**MAE** bars per p_drop (one bar group per
      subset, one color per p_drop)
    - **Expected:** p_drop=0.0 → catastrophic MAE when wifi missing;
      p_drop=0.4 → graceful degradation. RESULT_05/10/18
      anchors.
23. **Code** — Staleness curve on Webots:
    - load the K=4 winner (cell 16)
    - `trainer.evaluate_staleness(modality="wifi", split="test")`
    - returns (lag_seconds[], test_mae[]) — plot scatter +
      linear fit.
    - report slope (m/s) + intercept; both are MAE-derived.
    - **Expected:** slope ≈ 0.029 m/s, R² ≈ 0.995 across 27 s
      (RESULT_14)
24. **Code** — Inference latency probe (timing, not error):
    - `model = trainer.model.eval().to(DEVICE)`
    - warm up 50 iters; time 500 iters; b=1 and b=32
    - **Expected:** b=1 ~6 ms/sample, b=32 ~0.2 ms/sample on
      Quadro P4000 (RESULT_28)

### §6 — Limitations (paper §7 — VISIBLE, not footnoted)

Honest framing pulled live from saved data. This section is the
"honest fair" part of the user's brief — these are *cells*, not
markdown disclaimers.

Cells:
25. **Markdown** — "Three limitations. The cells below render the
    live data so reviewers can verify nothing is hidden. Per the
    metric policy (§3a), smoothness is shown VISUALLY (no Pearson
    r), and the IMU canonical gap reports both raw and
    Umeyama-aligned ATE (honest framing, neither cherry-picked)."
26. **Code** — Smoothness debt (VISUAL, not quantified):
    - on the Webots fusion winner (cell 16), pick 2-3 test paths
    - for each path, plot GT (x,y) trajectory vs predicted (x,y)
      trajectory as continuous lines on the same axes
    - visual feature to point out: the predicted trajectory is
      visibly over-smoothed compared to GT (over-low motion
      magnitude, especially at turns)
    - markdown caption framing: "Per-trajectory analysis reveals
      an over-smoothing limitation that is architecture-invariant
      in our exploration. A loss-function-level intervention
      (auxiliary velocity loss) is identified as a candidate fix
      for the journal version."
    - **NOT in this notebook:** the median-r number, the IQR, the
      per-arch quantitative sweep. Those stay in the archive
      notebook (run2_walkthrough.ipynb §6 cell 50) for internal
      use; not paper-facing per the MAE/ATE-only metric policy.
27. **Code** — IMU canonical RoNIN gap (honest framing):
    - re-display Anchor row from cell 13 in a "headline-vs-context"
      panel:
      - Raw ATE (paper headline): 9.96 m vs ResNet1D 5.14 m = **+94 %**
      - Umeyama-aligned ATE (context only): 7.88 m vs ResNet1D 5.14 m = **+53 %**
    - print: "Headline raw-ATE gap +94 %. Umeyama-aligned ATE
      (alignment removes rigid-body offset) narrows the gap to
      +53 %, still outside the 20 % SOTA gate. Parameter budget
      context: IMUCNN 0.05 M vs ResNet1D 4.6 M (95× smaller).
      Cross-subject generalization at this parameter budget is
      open future work."
    - Both numbers visible, neither hidden. This is the locked
      honest-framing rule.
28. **Code** — MSILN path-130 composition:
    - per-test-path breakdown table: `path_id | n_samples | our_MAE | wlanloc_MAE | WiFi-kNN_MAE`
    - highlight path 130 row (786 samples ≈ 28 % of test, very
      WiFi-dense → easy for WiFi-kNN baseline)
    - print framing: "Our fusion generalizes uniformly across all
      test paths; WiFi-kNN benefits disproportionately from
      path 130 in particular. Removing path 130 from the average
      shifts WiFi-kNN closer to our fusion while wlan_localization
      remains at +62 % above us. Per-path table reproduced in
      paper supplementary."

### §7 — Summary (paper §1 + §6 numbers in one panel)

Cells:
29. **Code** — Build a headline summary DataFrame pulling EVERY
    number from a live cell variable (no hand-typed). Render
    Markdown table. Bold = best.

    | claim | dataset | metric | Ours | SOTA | margin | cell# |
    |---|---|---|---|---|---|---|
    | WiFi-Net per-leg | UJI val | MAE | **8.69 m** | 15.17 (wlanloc) | **−43%** | §3 c12 |
    | IMU per-leg | RoNIN canonical | raw ATE | 9.96 m | **5.14** (ResNet1D) | +94% | §3 c13 |
    | Webots 2-mod fusion | sim test | MAE | **0.517 m** | n/a | — | §4 c16 |
    | **MSILN cross-session test ⭐** | real | MAE | **10.89 m** | 28.31 (wlanloc) | **−62%** | §4 c17 |
    | K-axis sweet spot | Webots | MAE | K=4 best | — | — | §5 c21 |
    | Staleness slope | Webots | MAE/s | 0.029 m/s | — | — | §5 c23 |
    | Latency b=1 | sim | ms/sample | 6 ms | — | — | §5 c24 |

    Two metrics only (MAE + ATE) per §3a metric policy.
    Umeyama-aligned ATE appears in §6 cell 27 (limitations) only.

30. **Markdown** — "Every number above is computed live in this
    notebook. Run with FAST_MODE=True for ~10 min reproduction
    on a Quadro P4000; FAST_MODE=False for ~3 h end-to-end
    retraining."

### §8 — Reproducibility footnote

Cells:
31. **Markdown** — Wall-clock matrix; checkpoint paths; SOTA
    submodule pinned commits; cite `scope.md` and the
    `external_methods/` submodule URLs.

---

## 6. FAST_MODE convention (binding)

```python
# §0 cell:
FAST_MODE = True  # toggle to False for end-to-end retraining

# Every train cell:
ckpt_dir = RUN_ROOT / "main_table" / dataset_name / "transformer"
if FAST_MODE and ckpt_dir.exists():
    trainer = load_trained(str(ckpt_dir), arch="transformer", dataset=dataset_name)
    history = None  # no training curve in FAST_MODE
else:
    trainer, history, _ = train_fusion_arch(
        arch="transformer", dataset=dataset_name,
        K=4, modalities=["wifi", "imu"],
        epochs=90, batch_size=128, lr=1.3e-3, seed=42,
        save_dir=str(ckpt_dir),
    )

# Common eval path (works in both modes):
val_mae = trainer.evaluate_subsets(split="val")["all"]
test_mae = trainer.evaluate_subsets(split="test")["all"]
print(f"{dataset_name}: val={val_mae:.3f} m, test={test_mae:.3f} m")
```

Same shape for Anchor2Vec and IMUCNN — `load_trained` is the only
load path; the inline trainers from `src.pipeline.training` are
the only train paths.

---

## 7. Acceptance criteria

Engineer must verify all of these before submitting RESULT_39.

### Hard requirements (paper-shipping blockers)

**Rename audit (§0a):**
- [ ] All §0a.2 LIVE files renamed; all §0a.3 FROZEN files
  untouched.
- [ ] `grep -ri "Anchor2Vec\|anchor2vec" src/ notebooks/ configs/ docs/ scripts/ README.md handoff/SUMMARY.md handoff/SCIENTIST_BRIEF.md` returns zero hits.
- [ ] `python -c "from src.pipeline.encoders import WiFiNet"` works.
- [ ] `python -c "from src.pipeline.training import train_wifi_net, wifi_net_predict"` works.
- [ ] `tests/test_encoders.py` passes after rename.
- [ ] Archive notebook `notebooks/run2_walkthrough.ipynb` still
  executes end-to-end (`nbconvert --execute`) with FAST_MODE=True.

**Metric policy (§3a):**
- [ ] Notebook displays only MAE and ATE in result tables.
  No RMSE, no median, no linear-probe vs kNN-probe distinction,
  no Pearson r in any cell output. `grep -i "rmse\|pearson" notebooks/paper_results.ipynb` returns no hits in output strings.
- [ ] Umeyama-aligned ATE appears ONLY in §6 cell 27 (honest-
  framing for IMU canonical gap).
- [ ] Smoothness debt cell (§6 cell 26) renders a VISUAL
  trajectory overlay, NOT a Pearson r number.

**Notebook itself:**
- [ ] **`notebooks/paper_results.ipynb` exists** at the path above
  and is committed (alongside `run2_walkthrough.ipynb` — both
  notebooks coexist).
- [ ] **Notebook executes end-to-end with FAST_MODE=True in ≤ 15 min**
  on the project GPU (Quadro P4000). Use:
  ```powershell
  .venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/paper_results.ipynb
  ```
  Exit code 0 required.
- [ ] **No `subprocess` imports, no `scripts.*` imports, no
  references to `scripts/_*.py` in cell source or markdown.**
  Engineer to run `grep -r "scripts/" notebooks/paper_results.ipynb`
  → empty result. Markdown may mention "see `scope.md`" but not
  any `scripts/` path.
- [ ] **All paper numbers come from live variables.** Engineer to
  verify by changing one variable mid-cell and confirming all
  downstream tables update. No string-literal "8.69" anywhere in
  markdown except in section headers / labels.
- [ ] **All four scope.md datasets** are reachable in the notebook
  (Webots / MSILN / UJI / RoNIN canonical). All cells of §1
  render without errors.
- [ ] **Limitations §6 visible** — the 3 limitations cells render
  the live data, not just markdown text.

### Honest-framing requirements (the "honest fair" part)

- [ ] **No camera, no odom, no IPIN, no IMUWiFine, no TartanAir.**
  Verified by string search on the notebook JSON.
- [ ] **No CNN1D / LSTM-attn / MoTTransformer** anywhere. Only
  `arch="transformer"`.
- [ ] **The MSILN headline 10.89 m vs 28.31 m is shown as
  `−62 %`** in the live-rendered table, computed from variables,
  not a hand-typed string.
- [ ] **The IMU on RoNIN canonical row in Table A reports +94 %
  raw / +53 % Umeyama** — both numbers visible, not just the
  better one. This is the locked honest-framing rule.
- [ ] **Smoothness debt cell prints median r < 0.10** as a real
  value, not text.
- [ ] **Path 130 in MSILN test breakdown is visible** as its own
  highlighted row, not hidden in an aggregate.

### Numbers tolerance

For all paper headline numbers, the FAST_MODE=True (load
checkpoint) reproduction must match the scope.md table within
±0.5 % (just numerical noise from re-eval). If FAST_MODE=False
(retrain) diverges by > 10 % on any headline, document in
RESULT_39 as either (a) seed sensitivity (acceptable, report
the spread) or (b) a real regression (blocker — re-train with
the saved seed/config).

---

## 8. Budget (estimate)

- §0a rename audit (mechanical find/replace + test): ~30-45 min.
- Engineer wall-clock to build the notebook (cells, imports,
  table assembly): ~3-4 h focused work.
- FAST_MODE=True end-to-end run validation: ~10-15 min.
- FAST_MODE=False end-to-end run (NOT a blocker for RESULT_39,
  but engineer should kick off and report ETA): ~3 h.
- Total: 1 engineer session.

If the Webots 2-mod K=4 checkpoint doesn't exist (the survey is
inconclusive — `runs/overnight/run2_iter_06/` may or may not have
a transformer ckpt at 2-mod K=4), engineer must retrain that one
cell inline (~20 min). Document in RESULT_39.

---

## 9. What's deferred to PLAN_40+ (NOT for this iteration)

- **Smoothness lever experiment (B-1 / B-2)** — the 30-min
  candidate from scope.md §10. If user wants to upgrade §6
  limitation framing from "open problem" to "we have a fix",
  this is a small follow-up plan.
- **K=2 + K=8 inline retraining** if those checkpoints don't
  exist — handled as a small follow-up if engineer's FAST_MODE
  scan shows them missing.
- **Camera-only supplementary** for journal — separate notebook,
  separate scope, separate plan.
- **Latex export** — out of scope here; the paper-writing phase
  is a separate iteration that consumes this notebook's outputs.

---

## 10. Notes for the engineer

**Use the archive notebook (`run2_walkthrough.ipynb`) as the
source of working code patterns.** Cells like UJI Anchor2Vec
training (archive cell 15) and MSILN transformer load (archive
cell 30) already work. Copy the cell **structure** into the new
notebook and strip everything outside the scope. Don't reinvent
working logic.

**The honest-framing cells (§6) are the heart of "honest fair".**
Don't hide the +94 % IMU gap by reporting only Umeyama. Don't
hide path 130 in an MSILN aggregate. Don't omit the smoothness
debt cell because the number looks bad. If a reviewer asks
"where's the per-trajectory smoothness analysis?" the answer is
"§6 cell 26 of `paper_results.ipynb`".

**No subprocess.run calls anywhere.** If a cell needs work that's
in a script today, port the script's body into the cell (or, if
the body is already a wrapper around `src.pipeline.*`, just call
the underlying API directly). Per survey §B, all training logic
already lives in `src.pipeline.training.inline_encoders` —
nothing should require porting.

**Commit final state to branch `paper-icinco-2026`** (the current
branch); do NOT git-push. User runs `git push` after review.

---

## 11. RESULT_39 expected structure

Engineer reports back with:

- Per-section completion (✓ / ✗ for each of §0-§8).
- FAST_MODE=True wall-clock (target: ≤ 15 min).
- FAST_MODE=False wall-clock (best effort, may not block).
- Any divergence from scope.md headline numbers > ±0.5 %.
- Any checkpoint that needed retraining (and what the new
  number was).
- Honest-framing self-check: did §6 render all 3 limitations
  with live data?
- Any cells where the scope.md vs archive-notebook mismatch
  forced an interpretation call (engineer's judgement); flag for
  scientist review.

---

*Scope.md anchor: §2 Architecture, §3 Modalities, §4 Datasets,
§5 Paper section structure, §6 Headline numbers, §7 Limitations.*
