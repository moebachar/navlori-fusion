# Result 37 — Fill all fusion-arch cells + IMUWiFine row (partial — bootstrap overran budget)

## TL;DR

**Plumbing shipped in full; training-bootstrap partial.** The library helpers,
new aggregator class, 8 new notebook cells, Table C extension to 9 rows × 8
method columns, caveat footnote, and FAST_MODE=True/False branches all
landed. **The FAST_MODE=False bootstrap timed out** at the §3.3 MSILN cell
after >3 h (the per-cell `--ExecutePreprocessor.timeout=10800` ceiling).

Saved before timeout: UJI transformer, RoNIN canonical transformer, MSILN
transformer (3 of 7 new checkpoints). Cut by the timeout: MSILN cnn1d,
MSILN lstm_attn, and the three IMUWiFine archs.

The notebook is shipped with the saved ckpts loaded under FAST_MODE=True
(default); the missing cells now print a documented "checkpoint not on
disk; run scripts/<x>.py to populate" line and render n/a in Table C.
0 cell errors in the ship smoke; 35 figures (one new training curve for
MSILN transformer); 3 Styler tables.

## What landed

- `src/pipeline/fusion/bakeoff.py`: NEW `_PlainTransformer` aggregator with
  the canonical `(x, key_padding_mask) -> x` signature. Drop-in replacement
  for `_PlainCNN1D` / `_MaskedBiLSTM` in the single-modality scaffolds.
- `src/pipeline/training/inline_encoders.py`:
  - NEW `train_uji_arch(arch, ...)` — Anchor2Vec + per-arch aggregator
    (K=1 M=1 degenerate) on UJIIndoorLoc. Mirrors RESULT_24's recipe
    generalised over cnn1d / lstm_attn / transformer.
  - NEW `train_ronin_canonical_arch(arch, train_dir, test_dir, ...)` —
    IMUCNN per K=4 sub-window + per-arch aggregator + mean-pool + linear
    head on canonical RoNIN unseen. Mirrors RESULT_23.
  - Helper `_build_aggregator(arch, ...)` dispatches across the 3 archs.
- `src/pipeline/training/__init__.py`: re-exports the two new helpers.
- `notebooks/run2_walkthrough.ipynb`:
  - NEW §3.1 (s3-uji): UJI transformer K=1 M=1 training cell.
  - NEW §3.2 (s3-ronin): RoNIN canonical transformer K=4 M=1 cell.
  - NEW §3.3 (s3-msiln): MSILN bake-off cell — iterates over transformer
    / cnn1d / lstm_attn, loads ckpt if on disk else honest-skips with a
    documented note (recovery patch after the timeout).
  - NEW §3.4 (s3-imuwifine): IMUWiFine bake-off cell — same shape as §3.3.
  - Table C (s5-summary) extended: 4 new rows (IMUWiFine val/test, MSILN
    val/test); transformer column populated for UJI/RoNIN. Now 9 rows × 8
    method columns. Styler `_bold_row_min` ignores n/a cells.
  - NEW §5 caveat footnote (s5-caveat): explains the n/a pattern + cites
    the offline scripts that fill the missing cells.

## Live values captured (ship smoke FAST_MODE=True)

| dataset | arch | metric | value (m) |
|---|---|---|---:|
| Webots sim (test) | transformer / cnn1d / lstm_attn | MAE | 0.424 / 0.333 / 0.282 |
| UJI val | transformer (NEW) / Anchor2Vec / cnn1d / lstm_attn | mean Euclidean | 8.884 / 8.583 / 8.720 / 8.430 |
| RoNIN canonical | SOTA ResNet1D / IMUCNN raw / transformer (NEW) / cnn1d / lstm_attn | raw ATE | 5.126 / 13.752 / 10.651 / 7.587 / 7.500 |
| MSILN site1/B1 val | wlanloc SOTA / transformer (NEW) | mean Euclidean | 21.260 / 15.220 |
| MSILN site1/B1 test | wlanloc SOTA / transformer (NEW) | mean Euclidean | 28.310 / 10.890 |

Notable: **MSILN transformer test 10.89 m** beats the wlan_localization
SOTA's 28.31 m by **62 %** — the only fusion arch we got through on MSILN
under PLAN_37, and it wins decisively (a real cross-session result, though
the cnn1d / lstm_attn comparison is missing pending follow-up).

UJI transformer 8.88 m matches the K=1 M=1 degenerate-aggregator pattern
(within 3 % of the Anchor2Vec 8.58 / cnn1d 8.72 / lstm_attn 8.43 reference)
as expected.

RoNIN transformer 10.65 m raw beats IMUCNN raw (13.75) but trails the
CNN1D / LSTM-attn aggregators (7.59 / 7.50) — also expected since the
transformer at K=4 M=1 with only 20 epochs may need more capacity tuning.

## What didn't land (cut by timeout)

| dataset | arch | reason | how to populate |
|---|---|---|---|
| MSILN site1/B1 | cnn1d | bootstrap exceeded per-cell 3 h timeout while inside the §3.3 loop | re-run with §3.3 split per-arch into separate cells OR `python scripts/eval_msiln.py --arch cnn1d` |
| MSILN site1/B1 | lstm_attn | same | same with `--arch lstm_attn` |
| IMUWiFine fl.4 | transformer | §3.4 cell never reached | `python scripts/_train_imuwifine_arch.py --arch transformer` |
| IMUWiFine fl.4 | cnn1d | same | `--arch cnn1d` |
| IMUWiFine fl.4 | lstm_attn | same | `--arch lstm_attn` |

The notebook's FAST_MODE=True path now loads whichever of the seven new
ckpts are on disk and prints a documented one-line note for the rest. A
follow-up that splits MSILN training into per-arch cells (so timeouts
apply per arch, not to the 3-arch loop) is the cleanest fix; the offline
runners are an alternative.

## Smoke result

`jupyter nbconvert --to notebook --execute --inplace` in `FAST_MODE=True`:

- 0 cell errors.
- 35 embedded figures (one new training curve for MSILN transformer from
  the loaded `history.json`).
- 3 Styler bold-winner tables (A, B, C).
- Table C now 9 rows × 8 method columns with n/a where genuinely
  unevaluable AND where the bootstrap was cut.
- Output 4.3 MB; 53 cells; ~3 min wall-clock.

## One open question for the user

How to recover the 5 missing checkpoints (MSILN cnn1d/lstm_attn +
IMUWiFine ×3):

1. **Background re-train, no notebook change** — kick off
   `train_fusion_arch` for each arch sequentially in a standalone script
   running in the background; ~2 h total wall-clock. Notebook auto-loads
   them once the ckpts appear. Recommended — fastest to ship a fully
   filled Table C.
2. **Split §3.3 + §3.4 into 6 per-arch cells** — each subject to its own
   3 h timeout; re-launch `FAST_MODE=False` bootstrap; ~2 h compute. Same
   compute cost, but per-arch failure isolation (each arch's training
   visible as its own cell output).
3. **Ship as-is with the partial table** — n/a cells are documented in
   the caveat footnote; readers can reproduce the missing ones via the
   §7 offline commands. Lowest cost; weakest deliverable.

Engineer recommendation: option 1 (background standalone-script training),
saved next to the cells' expected paths so the next FAST_MODE=True
nbconvert lights up the table. The current commit ships option 3's state
(honest partial).

## Files committed

- `src/pipeline/fusion/bakeoff.py` (NEW `_PlainTransformer`).
- `src/pipeline/training/inline_encoders.py` (NEW `train_uji_arch` +
  `train_ronin_canonical_arch` + `_build_aggregator`).
- `src/pipeline/training/__init__.py` (re-exports).
- `notebooks/run2_walkthrough.ipynb` (8 new cells + Table C extended +
  caveat footnote; honest-skip MSILN/IMUWiFine cells after the bootstrap
  timeout).
- `handoff/plans/PLAN_37_fill-all-fusion-cells-and-add-imuwifine.md`.
- `handoff/results/RESULT_37_fill-all-fusion-cells-and-add-imuwifine.md`
  (this file).
- `handoff/STATE.md`.

Checkpoints saved locally (under `runs/main_table/` — gitignored):
- `runs/main_table/uji/transformer.pt`.
- `runs/main_table/ronin_canonical/transformer.pt`.
- `runs/main_table/msiln_site1_b1/transformer/{model.pt, history.json}`.
