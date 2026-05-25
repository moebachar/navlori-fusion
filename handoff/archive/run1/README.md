# run1 — overnight 2026-05-24 → 2026-05-25 (archived)

This is the first overnight scientist+engineer run. Archived because
the user (Mohamed) flagged systemic issues with the work and asked
for a fresh restart with a different framing.

## Why archived

- Focused too narrowly on **WiFi + IMU** when the project's
  publishable story is **4-modality fusion** (WiFi + IMU + Odom + Camera).
- Never ran an **open-source SOTA baseline** on the new dataset
  (Microsoft ILN 2.0) — only trivial kNN / centroid / IMU-Kalman
  floors. So the "fusion beats baseline" claim was against trivial
  references, not published methods.
- Scaled compute before evidence (every iteration ran full 90-epoch
  training on full data, no small-subset pre-tests).
- Treated single-ablation results as definitive bottleneck diagnoses
  (capacity probe ran once → "encoder is structurally bound" too
  quickly).
- Engineer /loop died with the laptop sleep cycle and never recovered
  → 197 min of silence + later a partial recovery at 08:36 that
  ended with NO-PASS on iter_06.

## What's preserved (still in `src/` and `scripts/` — not archived)

- `src/pipeline/encoders/wifi_set.py` — `WiFiSetTransformer` class
  (sparse-observed forward after iter_06).
- `scripts/convert_msiln.py` — Microsoft ILN 2.0 converter.
- `configs/data/msiln_site1_b1.yaml` — dataset config.
- `data/msiln_site1_b1/` (untracked) — 133 converted traces with the
  Nov-train / Dec-test cross-session split.
- All Stage A encoders (`anchor2vec.py`, `imu.py`, `odom.py`,
  `dpvo_motion.py`, etc.) — untouched.

These are useful for run 2 — to be re-evaluated against SOTA in the
encoder audit (run 2 PLAN_01-04).

## Final numbers (for context)

Microsoft ILN 2.0 site1/B1, cross-session (train Nov-24, val Nov-25,
test Dec-05/06):

| method | val MAE | test MAE | latency |
|---|---:|---:|---:|
| Centroid floor | 65.1 m | 53.1 m | — |
| WiFi-kNN (k=5) | 17.7 m | 9.5 m | — |
| IMU Kalman | 115.0 m | 259.8 m | — |
| FusionTransformer (Anchor2Vec, K=8) | 15.7 m | 9.0 m | 4.2 ms |
| FusionTransformer (Set, K=1, sparse) | NO-PASS strict; smoothness 12.9→3.4 | — | 15× slower |

Goal was MAE ≤ 3.0 m + ≥ 1.5 m beat over best open-source single-
modality baseline. Run 1 did not reach the goal.

## What to read if picking up the history

1. `SUMMARY.md` here — one-pager of run 1's diagnostic stack.
2. `SCIENTIST_NOTE_iter05.md` — OOM root-cause analysis.
3. `results/RESULT_01..04.md` — committed iteration outputs.
4. `results/RESULT_06_wifi-set-encoder-sparse-observed.md` — the
   set-transformer attempt after engineer recovered.

Run 2 is launching from a fresh `STATE.md`, `SCIENTIST_BRIEF.md`, and
`plans/PLAN_01_*.md` at the parent `handoff/` directory.
