# Result 35 — Notebook fixes + comprehensive tables + trajectory overlays

## TL;DR

All 7 user takes shipped against the v4 publication notebook. The notebook is
now bug-free on the three data-honesty items, internally consistent on naming,
and complete on the comparison tables + qualitative trajectory plots.

| # | take | result |
|---|---|---|
| 1 | Webots camera shows TartanAir | **Fixed.** Webots panel now shows one depth frame + the cached DPVO patch features (no RGB persisted, honest about it). |
| 2 | IMUWiFine path = horizontal line | **Fixed.** `plot_gt_trajectory` auto-falls-back to `aspect='auto'` when `max/min(x_range, y_range) > 20`, and stamps the ratio in the title. |
| 3 | UJI WiFi sample fully saturated | **Fixed.** UJI panel is now two plots: a detected-only RSSI bar (~20-50 of 520 APs per scan) + a scan-sparsity histogram. The "saturation" was the 470+ non-detected APs at sentinel 100. |
| 4 | rename `incumbent` → `transformer` | **Done.** Registry alias added in `bakeoff.py` (`CANDIDATES["transformer"] = CANDIDATES["incumbent"]`); on-disk dir renamed `runs/overnight/run2_iter_33/incumbent` → `transformer`; notebook §3 + §5 use `transformer` throughout. |
| 5 | comprehensive summary tables | **Done.** §5 now has 3 Styler tables — Table A (encoders × datasets vs SOTAs, 3-4 rows), Table B (3 fusion archs × Webots val/test), Table C (best method per dataset, cross-cutting). All dynamic, all bold-winners. |
| 6 | poor plot quality | **Done.** `set_paper_style` bumped to font 12 / title 13 / DPI 150; all publication plotters opt into `dpi=150`, larger `figsize`, consistent palette (`transformer` added to `COLOR_PALETTE`). |
| 7 | no trajectory comparison plots | **Done.** NEW `plot_trajectory_comparison(predictions, gt)` helper in `src/pipeline/visualization/publication.py`; NEW §4.5 cell renders one overlay per Webots test path (15, 16, 17) with GT + all 3 fusion predictions. RoNIN + MSILN overlays cut per the plan's overrun provision (Webots covers the qualitative claim). |

## Code changes

- `src/pipeline/visualization/publication.py`:
  - `plot_gt_trajectory`: equal-aspect with auto thin-strip fallback; `dpi=150`.
  - `plot_modality_samples`:
    - **Webots camera**: shows one depth frame + cached DPVO patch features
      (`features` from `data/async_collection/path_*/dpvo_features.pt`).
      TartanAir-image fallback removed.
    - **UJI WiFi**: detected-only RSSI bar + scan-sparsity histogram.
    - All panels render at `dpi=150` with larger fonts.
  - `plot_preprocessing_influence`: `dpi=150` + larger figsize.
  - **NEW `plot_trajectory_comparison(predictions, gt, title, equal_aspect)`**:
    overlays multiple predicted (x, y) trajectories on a GT reference; handles
    thin-strip aspect; start/end markers.
- `src/pipeline/visualization/_style.py`: `PAPER_FONT_SIZE=12`,
  `PAPER_TITLE_SIZE=13`, `PAPER_FIG_DPI=150`; added `"transformer"` to
  `COLOR_PALETTE` (aliases `incumbent`'s grey).
- `src/pipeline/visualization/__init__.py`: re-exports `plot_trajectory_comparison`.
- `src/pipeline/fusion/bakeoff.py`: `CANDIDATES["transformer"] = CANDIDATES["incumbent"]`
  alias; `build_arch("transformer")` works; existing `"incumbent"` key preserved
  for back-compat with prior RESULTs.

## On-disk

- `runs/overnight/run2_iter_33/incumbent/` renamed to `runs/overnight/run2_iter_33/transformer/`
  (so `load_trained(...)` reads the same model under the new arch name).

## Notebook changes

- §0 modality-samples: Webots cam panel now depth + DPVO; UJI cam now
  detected-only bar + sparsity hist (all driven by `plot_modality_samples`).
- §0 GT trajectories: thin-strip auto-detection (IMUWiFine becomes readable).
- §3 (s3-md, s3-load, s3-eval): renamed `incumbent` → `transformer` throughout;
  loop iterates `('transformer', 'cnn1d', 'lstm_attn')`; checkpoint dirs follow
  the new name; `transformer_val_mae` / `transformer_test_mae` vars exposed
  for §5.
- §4.5 (NEW `s45-md` + `s45`): per-path trajectory overlays for Webots paths
  15, 16, 17 — GT + transformer + cnn1d + lstm_attn predictions colour-coded.
- §5 (s5-archive, s5-drift, NEW `s5-summary`): three Styler tables (A
  encoders/SOTA, B fusion × datasets, C cross-cutting best). All values pulled
  live; winner per row is bold.

## Smoke result

`jupyter nbconvert --to notebook --execute --inplace` in `FAST_MODE=True`:

- 0 cell errors.
- 28 embedded figures (up from 25 in v4: +3 trajectory-comparison plots for
  Webots test paths 15, 16, 17; also DPVO patch-feature heatmap).
- 3 Styler bold-winner tables rendered (Tables A, B, C in §5).
- **0 archive-drift prints** in cell output (verified by string search).
- Only one `"incumbent"` substring remaining in stdout — the
  `list_archs()` dump in §0 setup, which shows the registry alias
  (`['incumbent', 'lstm_attn', 'tcn', 'cnn1d', 'mot_transformer', 'transformer']`).
  No code path or display label uses `incumbent` directly.
- 40 cells; output 3.3 MB.
- ~3 min wall-clock (loads iter_33 checkpoints + history.json; renders curves
  + trajectory overlays; no training).

## One open question for the user

Per take #5, Table B is currently Webots-only (the only dataset where all 3
fusion archs were trained in run-2). MSILN / IMUWiFine / IPIN have only
CNN1D / LSTM-attn from prior iters, so they'd appear with `n/a` cells. Two
options:
1. Keep Table B Webots-only (recommended; honest about what we trained).
2. Add MSILN / IMUWiFine / IPIN rows with `n/a` for transformer + footnote
   pointing at the relevant RESULT_NN for the cells that exist.

I went with option 1 since it's tighter, but I can extend if you'd prefer
option 2.

## Files committed

- `notebooks/run2_walkthrough.ipynb` (v5 — publication-grade, bug-free).
- `src/pipeline/visualization/publication.py` (Webots cam fix, UJI sparsity,
  thin-strip aspect, new trajectory-comparison helper, DPI bumps).
- `src/pipeline/visualization/_style.py` (paper-grade defaults).
- `src/pipeline/visualization/__init__.py` (re-exports).
- `src/pipeline/fusion/bakeoff.py` (transformer alias).
- `handoff/plans/PLAN_35_notebook-fixes-and-comprehensive-tables.md`.
- `handoff/results/RESULT_35_notebook-fixes-and-comprehensive-tables.md` (this file).
- `handoff/STATE.md`.
