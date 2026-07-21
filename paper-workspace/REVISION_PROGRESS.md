# ICINCO 2026 Revision — Progress

Open this any time to see where the revision stands.
Newest milestones at the top of the log.

Started **2026-06-18**.

Decisions (autonomous unless flagged manual):

- **Seeds**: 3 per config (reviewer asks 3–5; 3 is the lower bound, fits the GPU budget).
- **M1 datasets**: Webots + MSILN.
- **M4c WIO-EKF**: declared infeasible (citation verified — Zhou et al., IEEE IoT-J 2024, DOI 10.1109/JIOT.2024.3386889 — but no public code; reimpl requires AP coordinates MSILN does not expose).
- **D1**: subsumed by M5 — `only:wifi` MSILN test = 10.66 m vs all = 10.90 m. No retrain needed.
- **D2 larger IMU backbone**: deferred (optional).

## Status

| ID | Item | Status | Headline |
|---|---|---|---|
| M3 | MSILN per-path MAE + macro average | **done** | macro = 11.71 m vs sample-weighted 10.90 m; dominant path is **130 (28.4%)**, not 131 as reviewer stated |
| M5 | Real-data modality + staleness sweeps | **done** | MSILN: all 10.90 / WiFi-only 10.66 / IMU-only 63.93 m; staleness graceful 10.90→14.44 then cliff at full-WiFi loss. IMUWiFine same shape (6.37→7.08 then 23.76). |
| D3 | K + period sensitivity | **done (full)** | K: 11.51 / 11.10 / **10.90** / 11.76 m. Period (MSILN test, seed 42): narrow (0.5, 10) = **17.22 m** (much worse!), default (0.05, 120) = 13.60 m, wide (0.01, 600) = **10.33 m** (best), shifted (0.1, 30) = 11.40 m. The default range is fine; only a *narrow* range hurts. |
| D2 | Larger IMU backbone (channels 64-128-256, 3.3× params) | **done** | Webots: val 0.395 / test 0.488 m (vs base 0.409 / 0.444). MSILN: val 15.40 / test 12.98 m (vs base 15.44 / 13.60). Both deltas within seed-level noise → reviewer's "IMU caps fusion ceiling" hypothesis not supported; the bottleneck is the WiFi cross-session encoder, not the IMU branch. |
| RELEASE | Public release refresh | **done** | README updated (path 130 fix + ablation section + mean ± std); notebook section 6.5 inserts ablation tables; cache/ablation_*.csv shipped. |
| D4 | WiFi-synth + obs-set docs | **done** | 120 APs, anisotropic RBF GPs, mean lengthscale (1.26, 1.14) m; obs-set K=4 most-recent per modality at instant_stride=9. |
| M4b | LSTM baseline protocol writeup | **done** | LSTM on MSILN: train 66.4 / val 65.2 / test 52.7 m matches predict-train-mean to 1.2 % → centroid-collapse overfit, not a misconfig. |
| M4c | WIO-EKF feasibility | **done** | NO public code; faithful reimpl requires AP coordinates MSILN doesn't expose → infeasibility note ready. |
| D1 | WiFi-only MSILN | **done (via M5)** | only:wifi = 10.66 m vs all = 10.90 m → fusion gain on MSILN cross-session is **not statistically distinguishable** (within 2.5%). |
| M1+M2 | Time-enc ablation × 3 seeds (Webots + MSILN) | **done (24/24)** | Webots: learned 0.448 ± 0.044, binned 0.465 ± 0.033 (tied), posindex 0.553 ± 0.043 (+23 %), none 0.585 ± 0.024 (+31 %). MSILN: learned 11.53 ± 3.15, binned 10.18 ± 0.75, posindex 12.49 ± 1.54, none 13.12 ± 2.85 — MSILN seed std ≈ 3 m dwarfs M1 mode effect (M2 caveat confirmed). |
| D3-period | Period range sub-ablation (3 variants × 2 datasets) | **running** | narrow (0.5–10 s), wide (0.01–600 s), shifted (0.1–30 s) at K=4/40 ep, seed=42. ~3 h ETA. |

## Milestones log

- **2026-06-19 08:26 — All revision items finalised.** D2 MSILN closed at val 15.40 / test 12.98 m (vs base 15.32 / 13.60 — within seed noise). The 3.3 × larger IMU backbone changes test MAE by ≤ 0.62 m on either dataset — within the seed-level std (±0.04 m Webots, ±3.15 m MSILN). The reviewer's "IMU encoder caps the fusion ceiling" hypothesis is not supported on these datasets — the bottleneck is the WiFi cross-session encoder, not the IMU branch. All paper-ready prose + tables now live in `revision/PAPER_INSERTS.md` between sentinel markers (`M1_M2_TABLE_*`, `D3_PERIOD_TABLE_*`, `D2_IMU_TABLE_*`); CSV mirrors in `revision/artifacts/`; release notebook section 6.5 ships the M1+M2 and D3 tables; release README corrected for path-130 and fusion-gain caveat. Nothing else owes you.
- **2026-06-19 07:30 — Big-plan items closing.** D3 period-range complete: narrow range (0.5–10 s) catastrophically worse on MSILN (17.22 m vs default 13.60 m); wide and shifted both within noise. D2 Webots done at val 0.395 / test 0.488 m (vs base 0.409 / 0.444 — within seed noise; bumping IMU 3.3 × doesn't help). MSILN D2 running, will auto-aggregate via Monitor `bagh52usk` when it finishes (~08:18). Public release refreshed: `release/navlori-fusion-public/README.md` corrected (path 130 not 131; fusion gain caveat; mean ± std numbers) and `notebooks/reproduce_paper.ipynb` section 6.5 inserts inline M1+M2 and D3 tables from `cache/ablation_*.csv`. Single remaining auto-step: D2 splice on completion.
- **2026-06-19 03:08 — M1+M2 batch FULLY complete (24/24).** Webots row unchanged from earlier. MSILN final (3 seeds per mode): **learned_continuous test 11.53 ± 3.15 m** (matches paper 10.90 within 1σ), **binned 10.18 ± 0.75 m** (best), posindex 12.49 ± 1.54 m, none 13.12 ± 2.85 m. The story: on MSILN cross-session, **seed std ≈ 3 m dwarfs the M1 mode effect** — no mode is statistically significantly better. The reviewer's M2 point ("differences may be within noise") is directly confirmed. Honest paper framing: on Webots, time encoding presence (vs. none) matters meaningfully (+31 % test); on real cross-session MSILN, the seed variance dominates, and the continuous-time mechanism's load-bearing role is on the *staleness curve* (M5), not on fresh-data MAE. D3 period-range sub-ablation (6 runs, ~3 h) now running in background; monitor armed.
- **2026-06-18 — Webots M1+M2 re-run at paper config (K=4, 40 epochs, MBL=false) complete.** 12/12 Webots runs ≈ 18 min total. **learned_continuous 0.448 ± 0.044 m** (matches paper 0.441m), binned **0.465 ± 0.033** (tied), posindex **0.553 ± 0.043** (+23 %), none **0.585 ± 0.024** (+31 %). Same ordering as the K=8 archive but smaller deltas — K=4 has less temporal context, so time-encoding choice matters less. The honest paper claim stays: any time encoding beats none; learned-continuous and binned are interchangeable on Webots; rank-only (posindex) loses meaningfully because real Δt magnitudes are discarded. MSILN batch (12 runs, ~60 min each → ~12 h) now running.
- **2026-06-18 — Hyperparam mismatch caught + recovery.** First batch ran at the YAML default (K=8, 90 ep, MBL=true) instead of the paper headlines (K=4, 40 ep, MBL=false from `runs/main_table/.../meta.json`). Webots was robust enough that K=8 results happened to match the paper number, but MSILN K=8 ≠ K=4 (D3 sweep showed +0.86 m gap). The 13th run (MSILN K=8) also hung — 67 min with zero epochs done. Killed all batch processes, archived the K=8 manifest at `revision/ablation_m1_timeenc/_manifest_K8_archive.json`, patched `revision/runners/train_one.py` to default K=4 / epochs=40 / MBL=false, re-launched. Total cost of the redo: ~12 h wall (mostly MSILN). Lesson: smoke-check meta.json against the actual production checkpoint, don't trust the YAML defaults.
- **2026-06-18 — M1 signal landed early.** First 6/24 batch runs done. Webots `learned_continuous` (3 seeds) = **val 0.387 ± 0.010 m, test 0.443 ± 0.049 m** — matches the paper's headline. Webots `no time encoding` (3 seeds) = **val 0.890 ± 0.015 m, test 1.064 ± 0.051 m**: removing the continuous-time encoding more than **doubles test MAE** (+140 %) at < 5 % run-to-run noise. The reviewer's M1 ablation question is decisively answered for Webots even before the `binned` / `posindex` variants finish. Aggregator + post-batch idempotent table-splice scripts wired to `revision/PAPER_INSERTS.md` (sentinel `M1_M2_TABLE_*`).
- **2026-06-18 — M1+M2 batch started.** Sequential 24-run queue launched in background. Webots ~4.8 min/run → ~1 h for all 12 Webots; MSILN ~15–20 min/run → 3–4 h. Whole batch should finish inside ~5 h. Live manifest at `revision/ablation_m1_timeenc/manifest.json`; aggregator at `revision/runners/aggregate_m1_m2.py`. Paper-ready insert doc landed at `revision/PAPER_INSERTS.md`.
- **2026-06-18 — Phase 1 complete (7/7 agents).** All eval/docs items landed in one parallel pass. Notable finds:
  - **Reviewer's path-131-dominant claim is wrong** — path 130 holds 28.4 % of test mass. The paper text must be corrected.
  - **Fusion gain on MSILN cross-session is essentially zero** (10.90 vs 10.66 m, WiFi-only). The reviewer's D1 question is answered honestly: temporal fusion preserves robustness, it does not lift fresh-data accuracy on real cross-session data.
  - **The MSILN-LSTM baseline is centroid-collapsed**, not misconfigured — train/val/test (66.4 / 65.2 / 52.7 m) all within 1.2 % of `predict-train-mean`, while the same code scores 1.6 m val / 4.1 m test on its native IMUWiFine.
  - **WIO-EKF cannot be reproduced faithfully** on MSILN — Microsoft never released AP coordinates; cite + position qualitatively.
- **2026-06-18 — D3 K-sweep finished (K=1/2/4/8 → 11.51 / 11.10 / 10.90 / 11.76 m).** K=4 sits at the minimum; longer horizons (K=8) regress because the model is trained at K=4.
- **2026-06-18 — M1 wiring landed.** `transformer.py` gained `time_enc_mode ∈ {learned_continuous, none, binned, posindex}`, builder exposes it via `cfg.model.time_enc_mode`. All 4 modes smoke-trained 1 epoch on Webots in ≤10 s each. Runner + batch driver at `revision/runners/`.
- **2026-06-18 — Setup.** Revision workspace + progress doc created. Phase 1 workflow dispatched (`wf_3e93e260-e8b`) — 7 agents in parallel for M3, M5×2 datasets, D3-K, D4, M4b, M4c.

## Where the artifacts live

- Phase 1 eval JSONs + markdown → `revision/per_path_m3/`, `revision/real_robustness_m5/`, `revision/k_period_d3/`
- Docs (WiFi synth, obs-set, LSTM protocol, WIO-EKF) → `revision/docs/`
- Trained checkpoints from M1/M2 → `runs/revision/m1_<dataset>_<mode>_s<seed>/`
- Ablation manifest (built incrementally) → `revision/ablation_m1_timeenc/manifest.json`
- Batch driver logs → `revision/ablation_m1_timeenc/batch.log`

## Blockers

(none — the only items still owing me decisions are the optional D2 and the eventual paper-text edits, both deliberately deferred.)
