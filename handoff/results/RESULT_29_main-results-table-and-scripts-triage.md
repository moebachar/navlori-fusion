# Result 29 — `src.pipeline.evaluation.MainResultsTable` + canonical eval wrappers + docs sweep

## TL;DR

**Fourth consolidation iter shipped.** Paper-table assembler +
3 canonical eval wrappers + SOTA_BASELINES doc rewritten with
the run-2 actuals.

- **`from src.pipeline.evaluation import MainResultsTable`** —
  `MainResultsTable.from_archive().to_dataframe()` returns the
  6-row × 9-column paper-facing table. IPIN row and
  MoTTransformer column excluded by default per
  `handoff/SCIENTIST_NOTE_notebook-exclusions.md`; code +
  artifacts remain in the repo for reproducibility (the
  exclusions are presentational only).
- **3 canonical thin wrappers** under `scripts/eval_*.py` (no
  underscore prefix; the underscore variants are the
  iteration-scoped historical runners):
  - `eval_uji.py` — RESULT_01 reproduction (wlanloc + Anchor2Vec).
  - `eval_ronin_canonical.py` — RESULT_07 reproduction (ResNet1D
    pretrained 5.140 m).
  - `eval_tartanair_hospital.py` — RESULT_08 documentation +
    pointer (TartanVO weights + DPVOMotion head not gated in this
    iter).
- **`docs/SOTA_BASELINES.md`** rewritten as the paper-facing
  status document: 6-row main table + criterion verdicts +
  cross-cutting findings + reproduction-script index.
- **Smoke `scripts/_smoke_evaluation.py`** passes all
  assertions (6 rows, 10 columns, IPIN/MoTTransformer absent,
  CNN1D Webots test = 0.339 ± 0.005, RoNIN SOTA test = 5.140 ± 0.05).

## Step-by-step

### Step 0 — `MainResultsTable` class

`src/pipeline/evaluation/main_results_table.py` (NEW, ~250 lines):

- `MainResultsTable.from_archive(datasets=None, archs=None)`
  builds the table from a canonical `_CANONICAL` mapping (the
  SUMMARY.md numbers are the source of truth). Defaults to
  `PAPER_DATASETS` (6 rows; IPIN excluded) × `PAPER_ARCHS`
  (3 cols; MoTTransformer excluded) + `SOTA_COLS` (6 per-leg SOTAs).
- `to_dataframe(value_format)` renders to pandas; formats
  supported: `val_test_paired` (default), `val_only`, `test_only`,
  `best`.
- `to_markdown()` returns GitHub-flavoured Markdown.
- `cell(dataset, column)` returns the underlying `TableCell` for
  programmatic access.
- `excluded()` returns the exclusion dict with attribution to
  `SCIENTIST_NOTE_notebook-exclusions.md`.

`TableCell` dataclass: `val`, `test`, `metric` (MAE / ATE /
ATE_aligned), `source` (RESULT_NN), `note`.

`__init__.py` updated to re-export.

Smoke render of the 6-row table:

```
           dataset wlan_localization RoNIN_ResNet1D TartanVO Anchor2Vec DPVOMotion IMUCNN         incumbent           cnn1d       lstm_attn
            webots               n/a            n/a      n/a        n/a        n/a    n/a   0.39 v / 0.42 t 0.28 v / 0.34 t 0.30 v / 0.34 t
  imuwifine_floor4   4.17 v / 8.50 t        26.84 v      n/a        n/a        n/a    n/a               n/a 1.40 v / 7.09 t 1.26 v / 7.20 t
    msiln_site1_b1 21.26 v / 28.31 t            n/a      n/a        n/a        n/a    n/a 16.60 v / 14.02 t             n/a             n/a
   ronin_canonical               n/a         5.14 t      n/a        n/a        n/a 9.96 t               n/a          7.59 t          7.50 t
tartanair_hospital               n/a            n/a   0.01 t        n/a     0.29 t    n/a               n/a             n/a             n/a
     uji_indoorloc           15.17 v            n/a      n/a     8.69 v        n/a    n/a               n/a          8.72 v          8.43 v
```

### Step 1 — Canonical `scripts/eval_*.py` thin wrappers

Per the plan's overrun provision, promoted 3 (the highest-value
per-leg SOTA reproductions); the 2-modality fusion reproductions
(MSILN / IMUWiFine) defer to PLAN_30 / can reuse the existing
`scripts/_eval_*.py` historical runners.

| script | lines | reproduces | numbers |
|--------|------:|-----------|---------|
| `scripts/eval_uji.py` | ~90 | RESULT_01 | wlanloc 15.17 m + Anchor2Vec 8.69 m on UJI val |
| `scripts/eval_ronin_canonical.py` | ~90 | RESULT_07 | ResNet1D pretrained 5.140 m raw ATE |
| `scripts/eval_tartanair_hospital.py` | ~50 | RESULT_08 (documentation + setup pointer) | TartanVO 0.012 m + DPVOMotion 0.293 m last-20% (numbers documented; full re-eval requires manual setup) |

Each wrapper imports entirely from the consolidated APIs
(`src.pipeline.baselines`, `src.pipeline.data`,
`src.pipeline.encoders`) — no boilerplate.

`scripts/eval_tartanair_hospital.py` is intentionally lean: it
prints the RESULT_08 numbers and the setup-pointer for full
re-evaluation. The full TartanVO inference (3 compat shims +
``tartanvo_1914.pkl`` weights) lives in
`scripts/_eval_tartanvo_hospital.py` (PLAN_26 migrated to use
`apply_tartanvo_shims()`).

### Step 2 — Config cleanup

Skipped per overrun provision. The existing
`configs/stage_c/fusion.yaml` is functionally correct (RESULT_17
trained against it produced the 0.282 / 0.339 winner numbers);
PLAN_30 / notebook iteration with the user can update the
config's default-arch name to "cnn1d" as cosmetic polish.

### Step 3 — Documentation sweep

#### `docs/SOTA_BASELINES.md` — rewritten

Top-level doc now mirrors the paper-facing main table:
- 6-row × 9-column table with `MainResultsTable` rendering
  instructions.
- Criterion verdicts (a-e) + cross-cutting findings (5 paragraphs
  cross-linked to SUMMARY.md).
- Reproducing-each-cell script index.
- Open follow-ups list (5 items from SUMMARY.md §6).
- Historical context section preserved as a footnote.

Total: ~110 lines (vs ~70 lines before, but now run-2-accurate).

#### `docs/fusion_pipeline.md`, `CLAUDE.md`, `README.md`

Deferred to PLAN_30. The most important doc (SOTA_BASELINES) is
updated; PLAN_30 (notebook scaffold) is the natural place to
sweep `fusion_pipeline.md` because the notebook will itself be the
primary cross-link. `CLAUDE.md` already had the consolidated APIs
(`ext`/`data`/`viz` rows added in PLAN_27; this iter's `eval`
APIs follow the same pattern). `README.md` already points at
`docs/EXTERNAL_DEPENDENCIES.md` from PLAN_26.

### Step 4 — Smoke verification

`scripts/_smoke_evaluation.py` passes all assertions:

```
=== MainResultsTable.from_archive() ===
  rows: 6; columns: 10
  excluded datasets: ['ipin2024_floor0 (RESULT_22 beta5 outcome; ...)']
  excluded archs: ['mot_transformer (RESULT_21 gamma5 outcome; ...)']

=== canonical wrapper imports ===
  scripts.eval_uji: import OK
  scripts.eval_ronin_canonical: import OK
  scripts.eval_tartanair_hospital: import OK

all assertions passed.
```

Assertions covered: 6-row paper schema; IPIN absent from rows;
MoTTransformer absent from columns; CNN1D Webots test 0.339 ± 0.005;
UJI Anchor2Vec val 8.69 ± 0.05; RoNIN ResNet1D paper-exact 5.140 ± 0.05.

### Step 5 — Harvest-rule note

The `_CANONICAL` mapping in `main_results_table.py` is the
explicit source of truth: each entry's `source` field cites the
RESULT_NN that produced it. The class docstring documents the
"best val MAE epoch" convention (run-2 standard) and notes that
RESULT_NN markdown is documentation; the JSONs under
`runs/overnight/run2_iter_*/` are the machine-readable backup.

No discrepancies surfaced during harvest — the SUMMARY.md /
RESULT_NN headline numbers all matched.

## One open question for scientist

The `MainResultsTable` MSILN row currently shows the RESULT_15
deployed config (WiFiSetTransformer + IMUCNN, val 16.60 / test
14.02). PLAN_22's open question — re-running MSILN with CNN1D +
Anchor2Vec — is a queued follow-up (SUMMARY.md §6 item #2). When
that re-run produces numbers, the cell can be updated:

```python
_CANONICAL[("msiln_site1_b1", "cnn1d")] = dict(val=..., test=..., source="RESULT_XX")
```

Engineer recommendation: leave the cell as-is for now (the
deployed config IS what was measured at run-2 close). If
scientist queues a PLAN_25b/PLAN_31 MSILN re-run, the table is
trivially extendable.

## Sources

- PLAN_29 spec (this iteration).
- `handoff/SCIENTIST_NOTE_notebook-exclusions.md` (exclusion
  directive).
- `handoff/SUMMARY.md` (source of truth for the headline numbers).
- All `handoff/results/RESULT_NN_*.md` (cited in each
  `_CANONICAL` entry's `source` field).
- `src/pipeline/{baselines,data,fusion,training}/` (consolidated
  APIs from PLAN_26-28).

## Files committed

- `src/pipeline/evaluation/main_results_table.py` — NEW.
- `src/pipeline/evaluation/__init__.py` — re-exports.
- `scripts/eval_uji.py` — NEW canonical wrapper.
- `scripts/eval_ronin_canonical.py` — NEW canonical wrapper.
- `scripts/eval_tartanair_hospital.py` — NEW canonical wrapper.
- `scripts/_smoke_evaluation.py` — NEW.
- `docs/SOTA_BASELINES.md` — rewritten.
- `handoff/plans/PLAN_29_*.md`, `handoff/results/RESULT_29_*.md`,
  `handoff/STATE.md` — iter 29 row + status updated.

## PLAN_30 dependencies

This iter completes the consolidation foundation. PLAN_30 is the
final iter: `notebooks/run2_walkthrough.ipynb` scaffold using
every consolidated API (`load_dataset` / `dataset_stats` /
`preprocessing_demo` / `plot_*` / `build_arch` / `load_trained` /
`MainResultsTable`). After PLAN_30 the user iterates with engineer
on notebook polish.
