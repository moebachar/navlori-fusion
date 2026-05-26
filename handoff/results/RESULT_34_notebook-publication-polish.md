# Result 34 — Notebook publication polish (5 user takes)

## TL;DR

`notebooks/run2_walkthrough.ipynb` is now a **publication artifact**, not a
dev-archive reproduction trace. All 5 of the user's takes are implemented:

1. **§0 reshaped** — distributions dropped; each dataset shows a ground-truth
   trajectory, a raw-modality sample, and a preprocessing-influence figure.
2. **Markdown humanized** + drift-vs-archive prints removed from every cell.
3. **RESULT_NN archive references dropped** from cell prints — the notebook's
   live numbers are the published numbers.
4. **Training curves** plotted in every inline-train cell (encoders + 3 fusion
   archs) in both `FAST_MODE` branches.
5. **§5 = two dynamic-bold summary tables** (pandas Styler): Ours vs SOTA, and
   our 3 archs across datasets — all values pulled live, winners bold.

Both smoke modes pass clean (0 cell errors). Ship default `FAST_MODE=True`.

## New / changed code

- **`src/pipeline/visualization/training_curves.py`** (NEW) —
  `plot_training_curves(history, title)`; accepts a dict or a `FusionHistory`
  dataclass; renders train/val-loss + val-MAE panels; returns `None` for
  closed-form fits (no per-epoch history).
- **`src/pipeline/visualization/publication.py`** (NEW) —
  `plot_gt_trajectory(name)`, `plot_modality_samples(name)`,
  `plot_preprocessing_influence(name, modality)`. Defensive: return `None`
  (caller skips) when a dataset's raw data isn't on disk. Verified rendering
  for all 6 paper datasets.
- **`src/pipeline/visualization/__init__.py`** — re-exports the 4 new helpers.
- **`src/pipeline/training/inline_encoders.py`** — `train_fusion_arch` now
  writes `history.json` next to `model.pt` so `FAST_MODE=True` can render the
  fusion training curve without re-training.
- **`notebooks/run2_walkthrough.ipynb`** — §0 reshaped (3 cells); §1.1/§1.2
  archive prints stripped; §2.1-§2.4 + §3 add training-curve plots and print
  only live numbers; §3 unified on `runs/overnight/run2_iter_33/<arch>`
  checkpoints for both modes; §4.3/§4.4 archive prints stripped; §5 replaced
  with two Styler tables; §6/§7 markdown humanized.

## Notebook §3 / §5 live values (FAST_MODE=False populate run)

§3 fusion bake-off (Webots, K=4, 4-mod, B=128, 90 ep, seed=42):

| arch | params (M) | val MAE (m) | test MAE (m) | smoothness r |
|---|---:|---:|---:|---:|
| incumbent | 1.55 | 0.395 | 0.377 | 0.001 |
| cnn1d | 0.51 | 0.272 | 0.346 | -0.003 |
| **lstm_attn** | 0.57 | **0.235** | **0.299** | 0.045 |

**LSTM-attn wins both val and test this run** — consistent with RESULT_33's
finding that RESULT_17's CNN1D-winner verdict rested on a sub-optimal LSTM-attn
seed. Table B bolds `lstm_attn` dynamically.

§5 Table A (Ours vs SOTA, live):

| dataset | modality | SOTA | SOTA (m) | Ours | Ours (m) | winner |
|---|---|---|---:|---|---:|---|
| UJI | WiFi | wlan_localization | 15.17 | Anchor2Vec | 8.58 | **Ours** |
| RoNIN canonical | IMU | ResNet1D | 5.13 | IMUCNN | 12.66 | **SOTA** |
| TartanAir | Camera | TartanVO (offline) | n/a | DPVOMotion | 0.29 | — |

(IMUCNN raw ATE 12.66 m this run — seed-sensitive as documented in RESULT_33;
Umeyama-aligned 7.20 m.)

## Smoke results

- **FAST_MODE=False** (populate run): 0 cell errors; 25 embedded figures
  (18 in §0 + 7 training curves); writes `history.json` for all 3 fusion
  archs; ~27 min wall-clock; output 1.45 MB.
- **FAST_MODE=True** (ship run): 0 cell errors; 25 embedded figures; 2 Styler
  bold-winner tables; **0 archive-drift prints** (Step 2 verified); §3 reloads
  the iter_33 checkpoints and reproduces the populate run exactly (incumbent
  0.395/0.377, cnn1d 0.272/0.346, lstm_attn 0.235/0.299); ~3 min wall-clock;
  output 1.44 MB. This is the shipped state (default flag `FAST_MODE=True`).

## Design decisions

- **§3 unified on `run2_iter_33`** (not the RESULT_13/17 archive checkpoints)
  so both `FAST_MODE` branches reproduce the *same* numbers and the notebook is
  internally consistent (take #3: the notebook is the canonical source).
- **pandas Styler** for the §5 tables (user's choice). Renders as `text/html`
  in the nbconvert output; identical bold-winner CSS is consolidated by Styler
  into one rule across cells (so the HTML may show one `font-weight: bold`
  string covering multiple bolded cells — not a bug).
- **DPVOMotion** has no per-epoch curve (closed-form least-squares head); the
  cell prints a one-line note instead of an empty plot.
- `plot_dataset_overview` (the distribution plotter) stays in the package but
  is no longer called from the notebook (take #1).

## One open question for the user

§3 now reports **LSTM-attn as the best architecture on both val and test**
(0.235 / 0.299 vs CNN1D 0.272 / 0.346), because the notebook trains all three
fresh at seed=42 and LSTM-attn lands in a better basin than RESULT_17's
archived CNN1D-winner run. The §4 ablations and §6 narrative still center on
**CNN1D** as "the winner". Two options:
1. Keep CNN1D as the headline (it's the documented Phase-B winner; LSTM-attn's
   edge is within seed noise) and add a one-line footnote that LSTM-attn ties /
   leads under this seed.
2. Re-center §4/§6 on LSTM-attn as the new winner and re-run the ablations on
   it.

I recommend option 1 for now (CNN1D winner + footnote) since the
cooperative-vs-dead-reckoning structural story (RESULT_18) is the real
contribution and a single-seed re-ordering shouldn't overturn it — but this is
a paper-framing call for you.

## Files committed

- `notebooks/run2_walkthrough.ipynb` (publication v4).
- `src/pipeline/visualization/training_curves.py` (NEW).
- `src/pipeline/visualization/publication.py` (NEW).
- `src/pipeline/visualization/__init__.py` (re-exports).
- `src/pipeline/training/inline_encoders.py` (`train_fusion_arch` history.json).
- `handoff/plans/PLAN_34_notebook-publication-polish.md`.
- `handoff/results/RESULT_34_notebook-publication-polish.md` (this file).
- `handoff/STATE.md`.
