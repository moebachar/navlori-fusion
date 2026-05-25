# Plan 06 — WiFiSetTransformer sparse-observed rewrite (OOM fix; supersedes PLAN_05)

## Hypothesis

PLAN_05's dense-masked design (build 1419 per-AP tokens per scan,
mask the unobserved 92 % out of attention) hit GPU OOM on all three
attempted batch sizes (128, 64, 32 — see
`handoff/SCIENTIST_NOTE_iter05.md` for the full diagnostic). The
mask only stops *softmax* from attending to those tokens; the tokens
themselves are still constructed, projected to `embed_dim`, and
fed through the feedforward block, costing the full `O(N²)`
activation memory at N = 1419.

The fix is **sparse-observed**: each scan feeds only the actually
observed APs as tokens (~127 mean per RESULT_01; budget 256 max).
That drops the attention cost ~120× and fits trivially in 8 GB at
the standard `batch_size = 128`.

The encoder file from iter_05 is **rewritten** (not patched) — the
`forward()` body differs materially. The wiring committed in iter_05
(`__init__.py` export, `builder.py` dispatch, config field, CLI
flags in `_train_msiln_b1.py`) **stays as-is**; no further plumbing
changes needed.

This is **iter_05 redone with the correct encoder cost model.** Same
hypothesis and bar levels apply.

### Expected outcome (WiFi-only training, msiln_site1_b1)

- `EXCELLENT` — val MAE ≤ 10 m AND test MAE ≤ 6 m → encoder works;
  PLAN_07 = re-introduce IMU with a learned modality gate.
- `GOOD` — val MAE ≤ 13 m AND test MAE ≤ 8 m → encoder helps;
  PLAN_07 = contrastive AP-dropout SSL pre-training to squeeze more.
- `MARGINAL` — val MAE ≤ 15 m AND test MAE ≤ 9.5 m → modest gain;
  scientist re-evaluates direction.
- `NO-PASS` — encoder no better than Anchor2Vec (val ≈ 15.7,
  test ≈ 9.0 from PLAN_03) → architecture isn't the fix; pivot to
  contrastive SSL or cross-session-only data engineering.

## Steps

1. **Rewrite `src/pipeline/encoders/wifi_set.py`** with the
   sparse-observed forward. Keep the class name `WiFiSetTransformer`
   and the `__init__` signature **unchanged** so the builder dispatch
   committed in iter_05 still works. Only the `forward()` body
   changes. Add ONE new init kwarg `max_observed_per_scan: int = 256`
   (defensive budget cap).

   Forward-pass logic (reference pseudocode — engineer may polish):

   ```python
   def forward(self, x):
       if x.ndim == 3:
           x = x.squeeze(1)               # (B, n_aps)
       B, N = x.shape

       observed = x > self.epsilon         # (B, N) bool
       n_obs    = observed.sum(dim=1)      # (B,)

       # Sort each row so observed APs come FIRST, ordered by RSSI strength.
       # observed → 10..11 range; unobserved → 0..1 range; ties broken by RSSI.
       sort_keys = observed.float() * 10.0 + x
       _, sort_idx = sort_keys.sort(dim=1, descending=True, stable=True)

       max_obs = int(n_obs.max().clamp(max=self.max_observed_per_scan).item())
       max_obs = max(max_obs, 1)           # never empty
       sort_idx = sort_idx[:, :max_obs]    # (B, max_obs)

       obs_bssid = sort_idx                # column index == BSSID id
       obs_rssi  = x.gather(1, sort_idx)   # (B, max_obs)

       bssid_emb = self.bssid_embed(obs_bssid)               # (B, max_obs, bssid_dim)
       tokens    = self.token_proj(torch.cat(
           [bssid_emb, obs_rssi.unsqueeze(-1)], dim=-1))     # (B, max_obs, D)

       cls = self.cls_token.expand(B, -1, -1)                # (B, 1, D)
       tokens = torch.cat([cls, tokens], dim=1)              # (B, max_obs+1, D)

       pad_mask  = obs_rssi <= self.epsilon                  # (B, max_obs)
       cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
       key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)

       out = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
       return self.out_norm(out[:, 0])
   ```

   Why this avoids OOM:
   - At `max_obs ≤ 256`: attention scores per layer per sample =
     `257² × 4 B = 264 KB` (vs `1420² × 4 B = 8 MB` for dense). At
     `batch=128, depth=2`, total activation footprint < 200 MB —
     fits comfortably in 8 GB alongside the rest of the fusion
     graph.
   - Sorting by `observed * 10 + x` keeps the strongest-signal
     observed APs first; if a scan has > 256 observed APs the
     weakest are dropped first (rare — RESULT_01 mean=127, max
     unmeasured but bounded by 1419).

   - **Acceptance:** module imports; forward pass on a synthetic
     `(4, 1, 1419)` input with random 100-observed-per-row returns
     `(4, 128)` with no NaN. Run a **memory budget check** with
     `B=128, K=8` (the worst real case in the fusion pipeline):
     synthetic batch → forward → backward → no OOM and peak GPU
     memory reported < 6 GB.

2. **Optional protective config knob.** Add `wifi_set_max_observed:
   256` to `configs/data/msiln_site1_b1.yaml`; the builder passes it
   to `WiFiSetTransformer(max_observed_per_scan=...)`. If this knob
   spreads to other files (builder.py), keep the touch count ≤ 2.
   - **Acceptance:** `load_config('msiln_site1_b1')` still loads;
     `build_encoders` instantiates the encoder with the configured
     value.

3. **Smoke gate (phase 1 + phase 2 — same as PLAN_05).** Run shape +
   NaN sanity on one mini-batch and the 16-sample overfit. Use
   `--modalities wifi --wifi-encoder set_transformer --wifi-pca 0`.
   - **Acceptance:** phase 1 passes; phase 2 drops training loss
     ≥ 80 % over 500 steps and reaches MAE < 3 m on the 16-batch.
     If phase 2 plateaus > 5 m, STOP and report — the architecture
     can't fit 16 scans.

4. **Full WiFi-only training run** at `batch_size = 128` (the
   pre-PLAN_05 default that previously OOM'd). 90 epochs,
   `patience = 15`, OneCycleLR, AdamW + Huber as before.
   Background-run with `flush=True`.
   - **Acceptance:** training completes ≤ 60 min wall (sparse path
     should be 2–3× faster than dense was even ignoring OOM); bar
     label assigned per the rubric in the Hypothesis section.

5. **Full evaluation** (identical shape to RESULT_03 / RESULT_05
   spec):
   - per-sample MAE on val + test (mean, median, RMSE).
   - per-path distribution (median, p25, p75, p90, max).
   - per-waypoint MAE (Kaggle-style).
   - subset eval (only:wifi == wifi+imu since IMU is off — sanity).
   - inference latency (1 sample + batch 32; should still be < 100 ms).
   - per-trajectory plot for all 5 test paths + smoothness ratio.
   - **Acceptance:** all numbers present in RESULT_06; explicit
     `Δ vs Anchor2Vec (PLAN_03)` and `Δ vs WiFi-kNN baseline`
     columns.

6. **PLAN_07 recommendation.** One of `add_imu_with_gate` /
   `contrastive_ssl_pretraining` / `redesign_or_pivot` per the bar
   rubric, with a 3-sentence justification quoting the measured
   numbers.

## Sources

- `handoff/SCIENTIST_NOTE_iter05.md` — full OOM diagnostic from iter_05.
- `runs/overnight/iter_05/train_set_xformer*.log` — the three OOM traces.
- PLAN_05 — the original plan; the encoder build it commissioned is
  preserved minus the `forward()` body.
- Encoder template: [Lazaro et al. 2025, arXiv:2506.00656](https://arxiv.org/abs/2506.00656).
- Set-attention reference for sparse tokens: `torch.gather` + stable
  `argsort` is the standard PyTorch pattern; no external lib needed.

## What to report back

In `handoff/results/RESULT_06_wifi-set-encoder-sparse-observed.md`:

1. Per-step pass/fail with the measured number.
2. **Memory budget check** from step 1: peak GPU MB at B=128, K=8.
3. Encoder parameter count + per-sample inference latency.
4. **Bar label** (EXCELLENT / GOOD / MARGINAL / NO-PASS) — one line,
   prominent.
5. Headline numbers table:

   | run                                       | val MAE | test MAE | per-wp val | per-wp test | smooth med | wall (min) | Δ vs Anchor2Vec |
   |---|---|---|---|---|---|---|---|
   | PLAN_03 Anchor2Vec baseline               | 15.70   | 8.99     | 20.54      | 18.56       | 12.92      | 18.4       | —               |
   | PLAN_06 set-transformer (WiFi-only)       | …       | …        | …          | …           | …          | …          | …               |

6. Per-path distribution table for both splits.
7. 5 trajectory plot file paths.
8. PLAN_07 label + 3-sentence justification.
9. One open question for scientist.

## Reversibility

- **Step 1** (rewrite `wifi_set.py:forward()`): permanent. The
  iter_05 commits already include the file with the dense-masked
  body; this iteration overwrites it. Easy to `git revert` either
  way.
- **Step 2** (optional config knob): permanent, default-safe.
- **Steps 3–5** (smoke + training + eval): throwaway under
  `runs/fusion_msiln_b1_set_sparse_<ts>/`, gitignored.
- The iter_05 partial work (encoder file + builder + config + CLI
  changes) is preserved — only the `forward()` body changes.

**Demand #3 untouched** (no vendored code involved).

**File-touch ceiling:** 1–2 files (`wifi_set.py` rewrite, optionally
`configs/data/msiln_site1_b1.yaml` + a one-line builder kwarg). Stays
well under the 5-file limit.

**Compute budget:** training ≤ 60 min; total iteration ≤ 90 min.
If the memory-budget check in step 1 still shows > 6 GB at B=128,
the engineer drops to `max_observed_per_scan=128` and reports the
new peak; do NOT spend another iteration on memory triage.

**Engineer note from iter_05 (no formal RESULT_05) explicitly
acknowledged.** The 1-line `runs/fusion_20260525_043131/meta.json`
artefact can be deleted or left as-is — it is just empty preamble.
