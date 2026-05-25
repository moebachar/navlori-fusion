# Plan 05 — Build per-AP / per-BSSID WiFi set-transformer encoder (WiFi-only validation)

## Hypothesis

RESULT_04 closed off three alternative hypotheses for the
`Anchor2Vec` ceiling on `msiln_site1_b1`:

- **Capacity** is not the issue (embed_dim 128 → 256 regressed both
  splits).
- **PCA dim** is not the issue (Branch C had standard pca=128 and
  matched only:wifi within 0.31 m).
- **IMU contribution** is *negative* — IMU branch poisons fusion at
  higher dim (smoothness 12.9 → 22.7).

What's left is the encoder's **architectural inductive bias**.
`Anchor2Vec`'s soft k-means over 64 anchors on a 1419-dim PCA basis
collapses the per-AP identity. The literature alternative
([Lazaro et al. 2025, arXiv:2506.00656](https://arxiv.org/abs/2506.00656);
[SelfLoc, MDPI Electronics 2025](https://www.mdpi.com/2079-9292/14/13/2675))
gives each BSSID its own learnable embedding and uses masked
attention so the encoder reasons over an unordered *set* of
(BSSID, RSSI) tokens with missingness as a first-class mask.

**Expected outcome (WiFi-only training, msiln_site1_b1):**

- `EXCELLENT` — val ≤ 10 m AND test ≤ 6 m → encoder is great;
  PLAN_06 = re-introduce IMU with gating / weighted attention.
- `GOOD` — val ≤ 13 m AND test ≤ 8 m → encoder helps; PLAN_06
  = contrastive AP-dropout SSL pre-training to squeeze further.
- `MARGINAL` — val ≤ 15 m AND test ≤ 9.5 m → modest gain;
  scientist re-evaluates direction.
- `NO-PASS` — no improvement → architecture is wrong or this
  dataset is much harder than the literature claims; scientist
  must redirect.

Training **WiFi-only** (not full fusion) is deliberate: Branch C
showed IMU currently hurts; the cleanest test of a new WiFi encoder
is in isolation. Re-introducing IMU becomes PLAN_06 once we know
the WiFi piece is good.

## Steps

1. **Implement `WiFiSetTransformer`** — new file
   `src/pipeline/encoders/wifi_set.py`. Subclass `BaseEncoder`,
   matching the `Anchor2Vec` contract exactly so the builder
   dispatch needs minimal plumbing:
   - Input: `(batch, 1, n_aps)` or `(batch, n_aps)` — same as
     `Anchor2Vec.forward` (see `src/pipeline/encoders/wifi.py` for
     the contract).
   - Input normalization assumed: `(rssi + 100) / 100` so missing
     APs are exactly `0.0` and observed APs are in `(0, 1]`.
   - Architecture (template — engineer may tune):
     - BSSID embedding table `nn.Embedding(n_aps, bssid_dim)` with
       `bssid_dim = 32`.
     - RSSI scalar projection `nn.Linear(1, bssid_dim)`.
     - Per-AP token = concat([bssid_emb(j), rssi_proj(x[:,j])]) →
       `nn.Linear(2*bssid_dim, embed_dim)`.
     - 2-layer `nn.TransformerEncoder` (`d_model=embed_dim,
       nhead=4, dim_feedforward=4*embed_dim, batch_first=True`).
     - **Padding mask** = `(x ≤ epsilon)` (missing APs masked out;
       choose `epsilon=0.005` so a 1-bit RSSI quantization is still
       observed). CRITICAL — never mask all tokens of a row
       (autopsy: softmax-of-all-minus-inf NaN). Prepend an unmaskable
       `CLS` parameter to guarantee at least one valid token.
     - Pool via the CLS token output → `(batch, embed_dim)`.
   - Implement `input_spec` property returning
     `{"modality": "wifi", "shape": (1, n_aps), "dtype": "float32"}`.
   - **Acceptance:** module imports; instantiating with
     `n_aps=1419, embed_dim=128` builds < 5 M params; forward pass
     on a random `(4, 1, 1419)` input returns `(4, 128)` with no
     NaN even when all RSSI = 0 (fully-missing row, CLS-only).

2. **Wire the builder to dispatch on `encoder_type`.** Add a
   `wifi_encoder_type: {anchor2vec, set_transformer}` field
   (default `anchor2vec` so IPIN/sim configs are untouched). In
   `configs/data/msiln_site1_b1.yaml`, set
   `wifi_encoder_type: set_transformer` and `wifi_pca: 0` (or
   whatever value disables PCA — engineer chooses) so the raw
   1419-d vector reaches the encoder. The `--wifi-pca` CLI flag
   added in iter 4 already handles the latter.
   - **Acceptance:** `from src.pipeline.fusion.builder import
     load_config, build_encoders; load_config('msiln_site1_b1')`
     succeeds; `build_encoders(...)['wifi']` returns a
     `WiFiSetTransformer` instance when the flag is set; the
     IPIN/sim configs (`load_config('ipin2024_floor-2')` etc.)
     still build `Anchor2Vec` (no regression).

3. **Smoke gate.** Use the existing `scripts/_smoke_fusion.py`
   pattern (or the `_train_msiln_b1.py` wrapper from iter 4 with
   `--smoke`-style early exit). Run two phases on msiln_site1_b1
   with `--modalities wifi --wifi-encoder set_transformer`:
   - Phase 1: shape + NaN sanity on one mini-batch (incl. a row
     of all-missing-AP synthetic input).
   - Phase 2: overfit a 16-sample batch.
   - **Acceptance:** phase 1 passes (no NaN, output shape correct);
     phase 2 drops training loss by ≥ 80 % over 500 steps AND
     reaches MAE < 3 m on the 16-batch (capacity check — the new
     encoder must easily memorise 16 scans). If phase 2 plateaus
     above 5 m, STOP — the encoder is broken; do not proceed to
     full train.

4. **Full WiFi-only training run on msiln_site1_b1.** Use the
   wrapper from iter 4 (`scripts/_train_msiln_b1.py`) with
   `--modalities wifi --wifi-encoder set_transformer
   --wifi-pca 0 --patience 15`. 90 epochs, AdamW + OneCycleLR +
   Huber as before. Background-run with `flush=True`.
   - **Acceptance:** training completes ≤ 60 min; best-val
     checkpoint saved; **bar label** assigned per the
     `EXCELLENT/GOOD/MARGINAL/NO-PASS` rubric in the hypothesis
     section.

5. **Full evaluation (same shape as PLAN_03 RESULT_03).**
   - per-sample MAE on val + test (mean, median, RMSE);
   - per-path distribution (median, p25, p75, p90, max);
   - per-waypoint MAE (Kaggle-style);
   - subset eval (only:wifi vs wifi+imu — `wifi+imu` should equal
     `only:wifi` since IMU is off; sanity);
   - inference latency (1 sample + batch 32);
   - per-trajectory plot for all 5 test paths with smoothness
     ratio (criterion d).
   - **Acceptance:** all numbers present in RESULT_05; explicit
     `Δ vs Anchor2Vec (PLAN_03)` and `Δ vs WiFi-kNN baseline`
     columns.

6. **PLAN_06 recommendation.** Based on the bar label, one of:
   - `add_imu_with_gate` (EXCELLENT) — next iter wires IMU back
     in with a learned modality gate (kills the noise injection).
   - `contrastive_ssl_pretraining` (GOOD) — next iter pre-trains
     the encoder with AP-dropout + RSSI jitter SSL on the
     unlabeled train scans.
   - `redesign_or_pivot` (MARGINAL / NO-PASS) — scientist must
     reconsider; possibly the BSSID-embedding alone isn't enough
     and we need MAML-style adaptation, or the dataset cross-session
     gap is harsher than the literature scan implied.
   - **Acceptance:** label + 3-sentence justification quoting
     measured numbers.

## Sources

- Encoder template: [Lazaro et al. 2025, arXiv:2506.00656](https://arxiv.org/abs/2506.00656)
  — Permutation-Invariant Transformer for Set-Based Indoor
  Localization (2.23 m mean on a 6-building campus dataset; no
  public code, so we implement to spec).
- Contrastive-SSL bonus (for PLAN_06 if GOOD):
  [SelfLoc, MDPI Electronics 2025](https://www.mdpi.com/2079-9292/14/13/2675).
- Contract reference: [src/pipeline/encoders/wifi.py](src/pipeline/encoders/wifi.py)
  (Anchor2Vec — match its `forward()` signature, `input_spec`
  property, and `BaseEncoder` subclass exactly).
- RESULT_04 — confirms structural saturation of Anchor2Vec on this
  data; justifies the swap.

## What to report back

In `handoff/results/RESULT_05_wifi-set-transformer-encoder.md`:

1. Per-step pass/fail with the measured number.
2. Files touched (count + paths; must be ≤ 5, otherwise pause and
   report "scope-too-large").
3. New-encoder param count + per-sample inference latency.
4. **Bar label** (EXCELLENT / GOOD / MARGINAL / NO-PASS) — one
   line, prominent.
5. Headline numbers table:

   | run | val MAE | test MAE | per-waypoint val | per-waypoint test | smooth med (test) | wall (min) | Δ vs PLAN_03 Anchor2Vec |
   |---|---|---|---|---|---|---|---|
   | PLAN_03 baseline | 15.70 | 8.99 | 20.54 | 18.56 | 12.92 | 18.4 | — |
   | PLAN_05 set-transformer (wifi-only) | … | … | … | … | … | … | … |

6. Per-path distribution table for both splits.
7. Subset eval (only:wifi should == wifi+imu since IMU is off; if
   not, that's a bug — flag).
8. 5 trajectory plot file paths.
9. PLAN_06 label + 3-sentence justification.
10. One open question for scientist.

## Reversibility

- **Step 1** (`src/pipeline/encoders/wifi_set.py`): **permanent** —
  new file, committed. Existing encoders untouched.
- **Step 2** (builder dispatch + config field): **permanent** but
  default-off (`wifi_encoder_type: anchor2vec` is the default;
  only the new msiln config opts in). Committed.
- **Steps 3–5** (smoke + training + eval): throwaway under
  `runs/fusion_msiln_b1_set_<ts>/`, gitignored.
- If the encoder regresses everything, revert is `git rm
  src/pipeline/encoders/wifi_set.py` + revert the small builder
  patch — IPIN/sim configs and the existing `Anchor2Vec` path are
  unchanged.

**Demand #3 untouched** (no vendored code involved).

**File-touch ceiling.** Estimate: 3–4 files (`encoders/wifi_set.py`
new, `encoders/__init__.py` export, `fusion/builder.py` dispatch,
`configs/data/msiln_site1_b1.yaml` field). If integration requires
also touching `data/dataset.py` to bypass PCA, that's file #5;
acceptable. If it spreads beyond 5 (e.g. trainer or fusion model
internals), the engineer should pause per PROTOCOL.md, write a
partial RESULT_05 with the encoder built + smoke-tested only,
and request a narrower PLAN_06.

**Engineer notes from RESULT_04 acknowledged:**
- Q1 (IMU noise injection): explicitly deferred to PLAN_06; PLAN_05
  trains WiFi-only to cleanly test the new encoder.
- Q2 (patience): patience=15 baked into step 4 per Branch C's
  5-epoch convergence observation.
- Q3 (5-path test split too small): noted; if PLAN_05+PLAN_06
  stabilise method, PLAN_08+ will convert a second floor
  (e.g. site1/F2 or F3 — both have 4-day cross-session splits per
  RESULT_01) for additional test paths and per-floor robustness.

**Compute budget:** training ≤ 60 min; total iteration ≤ 90 min.
If the engineer estimates > 5 files at design time, send a partial
result and split — do not extend the budget.
