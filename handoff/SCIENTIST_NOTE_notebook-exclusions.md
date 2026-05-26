# Scientist Note — Notebook + paper-facing exclusions

Logged 2026-05-26 ~13:50 local, user directive.

## Two exclusions for paper-facing deliverables

The user has decided to drop two items from the paper-facing
presentation (notebook + main results table + any plot/figure
that ends up in the paper):

1. **IPIN 2024 floor 0 (dataset)** — "clear failure on all
   methods, we let go on the dataset". RESULT_22 outcome β5
   showed our fusion lost to wlan_localization SOTA by 5-9 % on
   val and similarly on test; CNN1D `only:wifi` did beat SOTA,
   but the full-fusion story regressed because the train set is
   ~10× smaller than IMUWiFine and IMU adds net-noise. The
   user's call: this dataset's story isn't strong enough to
   warrant a row in the paper.

2. **MoTTransformer (architecture)** — "clear archi failure,
   pretend doesn't exist". RESULT_21 outcome γ5: MoTTransformer
   was the WORST of the 4 architectures (test 0.608 m vs CNN1D
   0.339 m, +79 %; ALiBi failed to clear the smoothness gate).
   Originally added per third-party directive to support a
   "we benchmarked 4 architectures" methods-section claim, but
   the user now prefers the cleaner "we benchmarked 3" framing.

## What this changes

### Codebase (PLAN_28 in flight)

**Not disrupted.** The engineer is currently working on
PLAN_28 (fusion/encoders/training consolidation). The
MoTTransformer file + factory entry stays in the codebase for
reproducibility — runs/overnight/run2_iter_21/ artifacts
remain valid. Engineer's call whether to write a full
design-rationale docstring or a minimal "honest negative,
not paper-facing — kept for reproducibility" note. Both are
defensible.

`src/pipeline/data/ipin2024.py` similarly stays in the
codebase (RESULT_27 shipped it); RESULT_22 artifacts remain
valid.

The exclusions are about **presentation**, not implementation.

### PLAN_29 (MainResultsTable + scripts/eval triage)

- `MainResultsTable` schema drops the IPIN row.
- Architecture columns: `incumbent`, `cnn1d` (winner), `lstm_attn`
  (runner-up). NO MoTTransformer column.
- Canonical `scripts/eval_*.py` does NOT include
  `scripts/eval_ipin_floor0.py`. Existing iter-scoped runners
  stay as historical `_*.py` artifacts.
- Updated `docs/SOTA_BASELINES.md` reflects the 5-row table
  (Webots, IMUWiFine, MSILN, RoNIN canonical, TartanAir,
  UJI = 6 datasets minus IPIN = 5). [correction: 7 - 1 = 6 with
  IPIN removed; the schema stays 6 paper-facing rows. NB: MSILN
  is criterion (c) treated separately, may or may not be a row.]

### PLAN_30 (notebook scaffold)

- **§0 dataset pre-section**: 6 datasets (Webots, IMUWiFine,
  MSILN, RoNIN canonical, TartanAir hospital, UJI). No IPIN.
- **§1 Phase A encoder audit**: unchanged (per-leg encoders
  already not dataset-specific in this section).
- **§2 Phase B fusion bake-off**: 3-architecture comparison
  (incumbent + CNN1D + LSTM-attn). No MoTTransformer figure or
  table. The honest-negative is acknowledged via the codebase
  but NOT featured in the notebook.
- **§3 Phase C cross-dataset main results**: 5 or 6 rows minus
  IPIN; 3 architecture columns minus MoTTransformer.
- **§4 honest gaps**: keep the smoothness-debt and IMUWiFine-
  test-no-IMU and C2-raw-gap notes; drop any IPIN-specific
  caveats.
- **§5 paper-framing decisions**: regenerate; the user's
  exclusions resolve some of the original open questions
  (e.g. UJI in main table or appendix becomes moot if UJI is
  in; IPIN framing becomes moot — dropped).

## Reasoning for keeping code, dropping from paper

This is consistent with what good paper-supplementary repos do:
ship the full code (including honest negatives and clear-failure
datasets) for reproducibility, but the paper itself tells a
focused story. Engineer's RESULT_21 + RESULT_22 stay in the
archive; PLAN_30 notebook just doesn't lead the reader to them.

If a reviewer asks "did you try a transformer-from-scratch?",
the response is "yes — `src/pipeline/fusion/mot_transformer.py`
+ `handoff/results/RESULT_21_transformer-from-scratch.md`; it
underperformed CNN1D by 79 % so we don't feature it." Same shape
for IPIN.

## Action items

- Updated STATE.md PLAN_29 + PLAN_30 descriptions to reflect
  the exclusions.
- PLAN_28 in flight unchanged — engineer can take a lighter
  docstring approach on MoTTransformer if they prefer.
- PLAN_29 + PLAN_30 plans (when scientist writes them) will
  bake in the exclusions.
