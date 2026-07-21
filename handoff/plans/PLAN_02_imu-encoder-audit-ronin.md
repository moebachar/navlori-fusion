# Plan 02 — IMU encoder audit on RoNIN (vs RoNIN ResNet1D)

## Hypothesis

`IMUCNN` is a small 1D-CNN (~500 k params) competing with RoNIN's
ResNet18 (~4.6 M params). Run-1 `docs/SOTA_BASELINES.md` (now restored)
reports **IMUCNN 14.41 m / 8.41 m raw / aligned ATE** vs **RoNIN ResNet1D
5.93 m raw** on the official `list_test_unseen.txt` (32 sequences). The
gap is ~2.4× (raw) / 1.6× (aligned) — outside the **20 %** audit window
of the SCIENTIST_BRIEF's criterion (a), so the most likely audit label
is **`replace` or `modify`** depending on whether one orthogonal probe
(capacity, Step 3 below) closes the gap.

Cycle rule "no anchoring on a single ablation as the bottleneck"
requires **3 orthogonal probes** before we can call X the bottleneck.
For IMU, those are:

1. **Architecture probe** — RoNIN ResNet1D (Step 1).
2. **Capacity probe** — IMUCNN with 2× hidden width / depth (Step 3).
3. **Preprocessing probe** — already done pre-run-1 (3.5× drop from
   hand-rolled → RoNIN's world-frame loader, documented in restored
   `docs/SOTA_BASELINES.md`).

After this iteration we'll have all three; the audit decision is
defensible.

Expected outcomes (per criterion (a)):
- `keep` — IMUCNN within 20 % of RoNIN ResNet1D (≤ 7.12 m raw if SOTA
  reproduces at 5.93 m) → don't replace.
- `modify` — IMUCNN 2× width closes ≥ 50 % of the gap → propose a
  beefed-up IMUCNN for Phase B fusion.
- `replace` — IMUCNN gap persists at any tested capacity → adopt
  RoNIN ResNet1D unmodified as the fusion IMU encoder (Demand #3 makes
  this clean: open-source code, used pure).

This is the second encoder in the Phase A audit (WiFi → **IMU** →
Camera → Odom). No fusion training in this plan.

## Steps

### Step 0 — Recovery + data-feasibility probe (5–10 min, gates everything)

**Step 0a.** Restore run-1 IMU audit files. They exist on
`overnight-autonomous-2026-05-24` but not on this branch:

```powershell
git checkout overnight-autonomous-2026-05-24 -- `
  scripts/eval_ronin_ate_fixed.py `
  scripts/eval_ronin_ipin.py `
  scripts/inspect_08_worldframe_imu.py `
  scripts/convert_ronin.py
```

Sanity: `python -c "from src.pipeline.encoders import IMUCNN"` already
works on this branch (Step 0 of PLAN_01 set up encoders/__init__.py
correctly for WiFi; `imu.py` was untouched by that change).

**Step 0b.** **Feasibility probe — does the full RoNIN dataset exist
on this machine?** The eval script needs `<root>/<sequence_id>/data.hdf5`
for every sequence in `list_train.txt` ∪ `list_test_unseen.txt` (~138
+ ~31 sequences). On this branch only `data/ronin_a000/` and
`data/ronin_a000_intra/` exist (subject a000 only). Probe in this
order, stop at the first hit:

1. `find $env:UserProfile -maxdepth 6 -name "FRDR_dataset_538*" -type d`
2. `find x:\ -maxdepth 5 -name "a006_2" -type d` (one of the unseen
   test sequences; if found, parent is the RoNIN root)
3. `dvc status data/` (in case there's an undocumented .dvc entry)

Branch on outcome:

- **Branch X — full data found.** Set `RONIN_ROOT=<found-path>` and
  proceed to canonical unseen-subjects benchmark in Steps 1–4.
- **Branch Y — only a000 available.** Skip Branch X and run the
  **a000 intra-session proxy audit** (Steps 1Y–4Y below). The proxy
  uses the LAST 30 paths of `data/ronin_a000` as a held-out test
  set (NOT the canonical "unseen subjects" — explicitly flagged in
  RESULT). This unblocks the audit decision now; Phase C will revisit
  with the full dataset for the C2 paper claim.
- **Branch Z — download.** If the dataset is reachable and engineer
  estimates < 30 min wall-clock to fetch + unpack, fetch via the
  FRDR URL in `Temp/ronin/README.md`. Otherwise default to Branch Y.

**Acceptance for Step 0:** branch chosen + path written to result.
If Branch Y, the audit's C2-relevance limitation is explicit in the
TL;DR.

### Steps for Branch X (canonical unseen-subjects benchmark)

#### Step 1X — Day-1 SOTA reproduction: RoNIN ResNet1D on unseen subjects

Use the vendored repo at `C:\Users\FabLab\AppData\Local\Temp\ronin\`,
specifically `source/ronin_resnet.py`. Demand #3: run unmodified
(except for runtime shims in OUR wrapper, never in their source).

Two options, prefer (a):

- **(a) Pretrained checkpoint, eval-only.** RoNIN README says "The
  models are trained on the entire dataset" and links pretrained
  weights at https://doi.org/10.20383/102.0543. If a checkpoint is
  obtainable in < 5 min and ≤ 500 MB, use it:
  ```powershell
  python C:\Users\FabLab\AppData\Local\Temp\ronin\source\ronin_resnet.py `
    --mode test `
    --test_list C:\Users\FabLab\AppData\Local\Temp\ronin\lists\list_test_unseen.txt `
    --root_dir $env:RONIN_ROOT `
    --out_dir runs\overnight\run2_iter_02\ronin_resnet_eval `
    --model_path <downloaded-checkpoint.pt>
  ```
- **(b) Train from scratch.** Same script, `--mode train` (their
  defaults). Expected ~30 min on Quadro P4000 for 100 epochs.

**Pre-test gate** (only if training from scratch): run 5 epochs on
the first 20 sequences from `list_train.txt`; expect train loss to
drop monotonically. If not, the data path or loader is wrong; stop
and document.

**Acceptance:** RoNIN ResNet1D unseen-subjects raw ATE within
±10 % of **5.93 m** (run-1 ref) or ±20 % of **5.14 m** (paper). Save
the report-style metrics RoNIN emits (`metrics.csv` / `*.json`).

#### Step 2X — Reproduce IMUCNN on the same split

Run the restored `scripts/eval_ronin_ate_fixed.py` (20 epochs default).
**Pre-test gate:** first 5 epochs — train Huber loss drops ≥ 30 %;
abort and document if not.

**Memory budget check:** the script's training uses batch 128 on
window 200, 6 channels — peak GPU should be well under 6 GB.
Engineer reports peak.

**Acceptance:** IMUCNN unseen-subjects raw ATE within ±10 % of
**14.41 m** (run-1 ref). Aligned ATE within ±10 % of **8.41 m**.
Both with per-sequence min/median/max.

#### Step 3X — Capacity probe (orthogonal probe #2)

Copy `scripts/eval_ronin_ate_fixed.py` → `scripts/_eval_ronin_imucnn_2x.py`
(underscore prefix marks iteration-scoped). Change only the encoder
construction line to:

```python
enc = IMUCNN(in_features=6, embed_dim=256).to(dev)  # 2× width
```

and the head to `nn.Linear(256, 2)`. **Memory budget check** still
applies (B=128, 6×200 window, embed_dim=256) — report peak; must
be < 6 GB. Train 20 epochs same protocol.

**Acceptance:** report raw + aligned ATE. Compare against Step 2X
result. If 2× width closes ≥ 50 % of the gap to RoNIN ResNet1D, the
audit label is `modify` (architecture is fine, just under-capacity).
Otherwise the gap is structural → label `replace`.

#### Step 4X — Six-metric harness

Run `src/pipeline/evaluation/encoder_eval.py` (or restore from run-1
if not yet present on this branch) on the held-out IMUCNN embeddings
from Step 2X and the 2× version from Step 3X.

For IMU on RoNIN, the 6 metrics behave differently than for WiFi:
- **Linear probe**: position from embedding — N/A (IMU is dead-
  reckoning velocity, not absolute position; skip or document).
- **kNN probe**: meaningful if labels are velocity vectors.
- **Alignment / uniformity / eff. dim / trustworthiness**: all run on
  embeddings; the geometry interpretation matters less for motion
  encoders than for fingerprints.
- **Temporal smoothness**: **applicable** (unlike WiFi/UJI). RoNIN
  is sequential; report ‖Δz‖ correlation with ‖Δgt_pos‖ along each
  unseen sequence.

If the 6-metric harness file isn't on this branch, restore from run-1
the same way as Step 0a. If restoration is non-trivial, write a
partial RESULT with only the regression metrics — do not let the
harness block the audit decision.

**Acceptance:** one row per (IMUCNN base, IMUCNN 2×, RoNIN ResNet1D);
only the metrics that are well-defined for motion encoders.

### Steps for Branch Y (a000 intra-session proxy)

If Branch X data isn't available, run a proxy benchmark on the
data we have:

#### Step 1Y — a000 proxy split

Build train/test lists from `data/ronin_a000/` (215 paths):
- Train: paths 0–184 (first 86 %).
- Test: paths 185–214 (last 14 %).

Write `runs/overnight/run2_iter_02/a000_lists/{list_train.txt,
list_test_unseen.txt}` and use them as `--test_list` / `--train_list`
to `eval_ronin_ate_fixed.py` (the script accepts paths relative to
`--root_dir`).

**Limitation explicit in RESULT:** this is **NOT** the canonical
"unseen subjects" — it's "unseen sessions of one subject." Inter-
subject generalisation is NOT tested. This is acceptable for the
audit decision (it answers "is IMUCNN structurally underpowered?")
but **does NOT discharge C2** — C2 requires the canonical benchmark
in a later iteration with the full dataset.

#### Steps 2Y, 3Y, 4Y — same as 2X, 3X, 4X but on the a000 proxy

Numbers should NOT be directly compared to run-1's 14.41 m / 5.93 m
(those are unseen-subjects). They are this-iteration's reference
only. Report raw + aligned ATE per sequence + mean.

### Step 5 — Audit decision (both branches)

Label IMUCNN with `keep` / `modify` / `replace` per the hypothesis
rubric. One-line justification quoting the numbers from Steps 1–4.

If Branch Y was used: include explicit caveat that the cross-subject
claim (C2) remains unresolved and is queued for Phase C.

**Acceptance:** explicit label + 3-sentence justification + PLAN_03
recommendation (default = Camera audit). If audit labels IMUCNN
`replace`, also note explicitly that PLAN_03 onwards uses RoNIN
ResNet1D unmodified as the fusion IMU encoder.

## Sources

- RoNIN paper: Herath, Yan, Furukawa, ICRA 2020.
  https://arxiv.org/abs/1905.12853 — paper ATE 5.14 m unseen-subjects.
- RoNIN repo (vendored MIT): `C:\Users\FabLab\AppData\Local\Temp\ronin\`,
  upstream https://github.com/Sachini/ronin.
- RoNIN dataset (FRDR): https://doi.org/10.20383/102.0543 (50 %
  released publicly, README says).
- IMUCNN: `src/pipeline/encoders/imu.py` (on this branch, untouched
  by PLAN_01).
- Run-1 restored audit doc: `docs/SOTA_BASELINES.md` — IMU section
  (14.41 m / 5.93 m references).
- Existing eval script (to be restored): `scripts/eval_ronin_ate_fixed.py`.

## What to report back

In `handoff/results/RESULT_02_imu-encoder-audit-ronin.md`:

1. **Step 0 outcome:** which branch (X/Y/Z) was taken, what data was
   found, what was downloaded (if anything).
2. **Pre-test gate outcomes** for Steps 1X/2X/3X (or Y variants).
3. **Memory budget checks** for the new architecture (Step 3X, 2×
   width).
4. **Headline numbers table:**

   | encoder / method | RoNIN unseen raw ATE (m) | aligned ATE (m) | params | latency (ms / window) | source / notes |
   |---|---|---|---|---|---|
   | RoNIN ResNet1D (SOTA) | …  (target 5.93 ±10 %) | … | ~4.6 M | … | pretrained / from-scratch (specify) |
   | IMUCNN (ours, base) | … (target 14.41 ±10 %) | … (target 8.41 ±10 %) | ~0.5 M | … | scripts/eval_ronin_ate_fixed.py |
   | IMUCNN 2× width (ours) | … | … | ~2 M | … | NEW _eval_ronin_imucnn_2x.py |

   (Branch Y: same table but on the a000 proxy split, with an
   explicit "this is NOT unseen-subjects" caveat next to it.)
5. **6-metric harness table** (only well-defined metrics for motion
   encoders): kNN, alignment, uniformity, eff. dim, trustworthiness,
   temporal smoothness. One row per encoder variant.
6. **Audit label:** IMUCNN = `keep|modify|replace`. One-line
   justification.
7. **PLAN_03 recommendation:** continue to Camera audit, OR insert
   a follow-up iteration if Step 3X surprises (e.g. 2× width closes
   most of the gap → propose a `modify` track for Phase B).
8. **C2 status note:** is C2 (per-leg IMU SOTA) discharged or queued
   for Phase C? Explicit.
9. **One open question** for scientist.

## Reversibility

- Step 0a (file recovery): permanent. Engineer commits with the result.
- Step 0b (data probe): throwaway documentation.
- Step 1X (SOTA reproduction): throwaway — vendored repo runs against
  vendored code; no edits.
- Step 2X (IMUCNN eval): throwaway training run; checkpoint optional.
- Step 3X (`scripts/_eval_ronin_imucnn_2x.py`): permanent. Engineer
  commits.
- Steps 4–5: documentation only.
- Branch Y artifacts under `runs/overnight/run2_iter_02/a000_lists/`
  are throwaway and gitignored.

All run artefacts under `runs/overnight/run2_iter_02/`, gitignored.
Files committed: restored files from Step 0a, RESULT_02,
`scripts/_eval_ronin_imucnn_2x.py`.

**Demand #3 untouched.** RoNIN vendored repo is not edited; the
`np.int = int` compat shim sits in OUR wrapper, never in vendored
source. Same convention as PLAN_01.

**Compute budget:** total iteration ≤ 90 min.
- Step 0: 10 min (recovery + data probe).
- Step 1X: 5 min (pretrained eval) or 30 min (from-scratch).
- Step 2X: 15–20 min (20 epochs IMUCNN, small model).
- Step 3X: 20–25 min (20 epochs IMUCNN-2×).
- Step 4X: 5 min (eval-only).
- Step 5: 5 min writeup.

If overrun: drop Step 4 first (6-metric harness — least essential
for the audit decision). Drop Step 3 only if absolutely needed
(losing the orthogonal capacity probe makes the audit label less
defensible).

If Step 1X can't run (data missing, RoNIN repo can't be invoked),
the iteration becomes "audit blocked on data acquisition" — write
partial RESULT, scientist plans data acquisition as its own
iteration (worst case, but rare).
