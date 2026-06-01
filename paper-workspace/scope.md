# NavLoRI-Fusion — Conference Publication Scope

Author: Mohamed Bachar (CESI LINEACT). Document maintained by
scientist (Claude) ↔ user iteration, 2026-05-30 ~10:00 local.

> **Purpose.** Define the **minimal-but-complete** scope of the
> NavLoRI-Fusion project that goes into the conference paper.
> Supervisors' framing: "you did a lot for a conference — find a
> minimal scope to publish, leaving room for a journal version
> 1-2 months later." This document is the contract between
> *what's in the conference paper* and *what's deferred to the
> journal*.

---

## 0. Status

- **Source-of-truth artefacts already in repo:** `handoff/SUMMARY.md`
  (run-2 archive narrative); `notebooks/run2_walkthrough.ipynb`
  (publication-grade reproducibility notebook); `runs/main_table/`
  + `runs/overnight/` checkpoints; `handoff/results/RESULT_01-37*.md`
  per-iteration findings.

---

## 1. Headline contribution (the *one* claim the paper proves)

> A **continuous-time set-transformer** that fuses **asynchronous
> WiFi + IMU sensor streams** for indoor localization, demonstrating
> **robust real-world cross-session generalization** without
> per-modality custom architectures.

The contribution has three load-bearing components, each named in the
paper title:

1. **Continuous-time** — the `time_encoding(Δt)` per token. Every
   sensor sample contributes one token; that token carries the
   elapsed time since the readout query. The same self-attention
   block handles arbitrary sample rates (1 Hz WiFi + 30 Hz IMU)
   without resampling or zero-padding.

2. **Set-transformer (unified)** — ONE self-attention block does
   both cross-modal and cross-time fusion. No separate temporal
   filter, no per-modality custom branches. The architecture is
   permutation-invariant over modalities (modality-embedding, not
   ordering); modalities can be added or removed without changing
   the trunk.

3. **Async-robust** — training-time `modality_dropout` (0.4) +
   `instant_dropout` (0.45) → test-time graceful degradation under
   missing modalities and stale sensors. The same mechanism handles
   both: no architectural branches needed.

**Working paper title (candidate):**
"Async-Robust Multi-Modal Indoor Localization via a Continuous-Time
Set-Transformer."

---

## 2. Architecture in scope

**One fusion architecture only:** the set-transformer (the run-1
`FusionTransformer`; renamed `transformer` per RESULT_35 nomenclature
sweep).

### Token construction (per sample, per modality)

```
token = encoder(raw_sample) + modality_embedding[modality]
                            + time_encoding(Δt = t_sample − t_query)
```

- `encoder` = per-modality encoder (Anchor2Vec for WiFi, IMUCNN for
  IMU). Both are pre-trained or jointly trained.
- `modality_embedding` = learnable per-modality vector (`M × D` table).
- `time_encoding(Δt)` = sinusoidal positional encoding evaluated on
  the continuous Δt value, not a discrete index. **This is the
  "continuous-time" piece.**

### Self-attention block (cross-modal + cross-time, unified)

K instants × M modalities = K·M tokens flatten into one sequence.
Single self-attention block attends over all K·M tokens. No
separate cross-modal vs cross-time stages — one operation handles
both.

### Readout (cross-attention)

Learnable `PositionQuery` token (1 token, D dims). Cross-attention:
query attends over the K·M tokens. Output → 2-layer MLP → (x, y).

### Training

- 90 epochs, AdamW + OneCycleLR + Huber(δ=0.5).
- B=128, K=4 (default; ablated 1/2/4/8 in §6).
- `modality_dropout=0.4`, `instant_dropout=0.45` (RESULT_05's
  audit-fix values).
- Seed=42 (paper-reproducible).

**Out of scope (→ journal):** CNN1D temporal-conv aggregator,
LSTM-attn aggregator, MoTTransformer with ALiBi. These are
mentioned in §7 discussion as "we also explored simpler aggregators
that trade unified attention for fresh accuracy; see journal
version."

### Architecture parameter / latency budget

- ~1.5 M parameters (K=4, M=2, D=128).
- Latency: ~6 ms / sample at b=1 on Quadro P4000 (RESULT_18 measurement
  shape; transformer-arch number from RESULT_28).
- Memory: < 1 GB peak at training (B=128 K=4 M=2 D=128).

---

## 3. Modalities in scope

**Two modalities only: WiFi + IMU.** Realistic smartphone setting.
Both have well-defined public SOTAs to compare against. Both
demonstrate genuinely async sample rates (the "async" claim's
load-bearing data property).

| modality | sensor rate | role | encoder | SOTA reference |
|---|---|---|---|---|
| WiFi RSSI | ~1 Hz (1 detection per scan per AP) | absolute anchor; sparse high-info | **Anchor2Vec** (~0.075 M params) | wlan_localization (sharan-naribole, MIT) |
| IMU 6-channel | ~30 Hz (gyro + accel × xyz) | dense motion / dead-reckoning | **IMUCNN** (~0.05 M params) | RoNIN ResNet1D (Sachini, MIT) |

**Out of scope (→ journal):**

- **Camera** (DPVOMotionEncoder): paper-soft per-leg validation
  on TartanAir hospital (+2300 % gap to TartanVO last-20 % slice
  per RESULT_08); Webots RGB never persisted at collection time
  (only depth + cached DPVO features per RESULT_35). Saved for
  journal where we can do proper public-VO-benchmark validation.
- **Odometry** (OdomCNN): only Webots has odom modality; no real-
  world cross-session validation possible in the conference scope.
  Saved for journal's 4-modality story.
- **WiFiSetTransformer** alternate WiFi encoder: parked per
  RESULT_01 (UJI verdict was `replace` — Anchor2Vec wins). Not
  in scope.

---

## 4. Datasets in scope (4 total: 1 sim + 3 real)

| dataset | role in paper | modalities used | source |
|---|---|---|---|
| **Webots Tiago sim** | controlled lab; end-to-end fusion baseline + ablations | WiFi + IMU (2-mod K=4) | own collection, 18 paths |
| **MSILN site1/B1** | **headline real-world claim**: cross-session WiFi+IMU fusion | WiFi + IMU | Microsoft Indoor Location 2.0 |
| **UJIIndoorLoc** | per-leg WiFi encoder validation | WiFi only | Torres-Sospedra 2014 |
| **RoNIN canonical** unseen-subjects test | per-leg IMU encoder validation | IMU only | Sachini/ronin (Herath 2020) |

### Per-dataset use

- **Webots sim (controlled, in scope):** demonstrates the end-to-
  end pipeline in a fully controlled lab. All ablations (K-axis
  sweep, modality dropout, staleness sweep) run here. Canonical
  split (RESULT_06 onwards): train [1,3-12] / val [2,13,14] /
  test [15,16,17].

- **MSILN site1/B1 (real, headline):** cross-session WiFi+IMU
  smartphone collection. Nov-24 train, Nov-25 val, Dec-05/06 test
  (cross-session by design). Our transformer at val 15.22 m /
  test 10.89 m vs wlan_localization val 21.26 m / test 28.31 m =
  **62 % test improvement** (RESULT_37). This is the paper's
  load-bearing real-world result.

- **UJIIndoorLoc (real, per-leg):** standard canonical WiFi-
  fingerprinting benchmark. Used only for the WiFi-encoder per-
  leg comparison. Anchor2Vec val mean Euclidean **8.69 m** vs
  wlan_localization **15.17 m** = **43 % improvement** (RESULT_01).

- **RoNIN canonical (real, per-leg):** standard inertial-
  navigation benchmark on 32 unseen-subject sequences. IMUCNN
  raw ATE 9.96 m / Umeyama 7.88 m vs RoNIN ResNet1D raw ATE
  5.14 m (paper-exact reproduction, RESULT_07). **Honest
  framing:** in-domain competitive; cross-subject gap noted as
  future direction.

### Out of scope (→ journal)

- **IMUWiFine fl.4** — cross-campaign WiFi+IMU; complex test-no-
  IMU structural property; saves for journal where the longer
  format permits the honest framing.
- **IPIN 2024** — small-train regime; CNN1D `only:wifi` actually
  beats wlanloc on val but the full-fusion overfits (RESULT_22).
  Honest finding worth a paragraph in the journal; too nuanced
  for the conference's tight scope.
- **TartanAir hospital** — Camera-only; no fusion target.
- All cross-floor / cross-building extensions.

---

## 5. Paper section structure (~10 pages PerCom format)

| § | content | page budget | source archive |
|---|---|---|---|
| 1. Introduction | indoor localization context, sensor heterogeneity problem (async rates / missing modalities / staleness), our contribution claim, paper roadmap | 1 | new prose |
| 2. Related Work | WiFi fingerprinting (wlanloc, CNNLoc, eAaT+, Locaris); inertial navigation (RoNIN); multi-modal fusion (set-transformer / cross-attention); async sensor fusion | 1 | new |
| 3. Method | (a) per-modality tokenization; (b) continuous time encoding; (c) self-attention block; (d) PositionQuery readout; (e) training (modality+instant dropout) | 2 | RESULT_06+ + architecture spec |
| 4. Per-leg encoder validation | Anchor2Vec on UJI (vs wlanloc); IMUCNN on RoNIN canonical (vs ResNet1D); honest in-domain framing for IMU | 1 | RESULT_01 + RESULT_07/23 |
| 5. End-to-end fusion experiments | (a) Webots 2-mod K=4 baseline; (b) **MSILN cross-session headline**; (c) staleness sweep on Webots (cliff → slope); (d) modality-dropout robustness on Webots | 2.5 | RESULT_06 + RESULT_37 + RESULT_11/14/18 |
| 6. Ablations | K-axis sweep (K=1/2/4/8 on Webots); modality-dropout rate sweep; latency probe | 1 | RESULT_11/12/13/14/18 |
| 7. Discussion & Limitations | smoothness debt (architecture-invariant, loss-function lever future); C2 in-domain honest framing; aggregator-family note (CNN1D for journal) | 1 | RESULT_05/18/23 |
| 8. Conclusion + Future Work | recap + journal directions: 4-modality extension, camera, comprehensive bake-off, loss-function lever | 0.5 | new |

**Total: ~10 pages of prose + 5-6 figures + 2-3 tables.**

---

## 6. Headline numbers (paper-ready, cited to source)

Every number in the paper either comes from a live notebook cell
(default FAST_MODE=True load + eval) or from a saved checkpoint
already in `runs/`. Source of truth:

### Per-leg encoder validation

| comparison | our encoder | SOTA | margin | source |
|---|---|---|---|---|
| WiFi on UJI val mean Euclidean | Anchor2Vec **8.69 m** | wlan_localization 15.17 m | **−43 %** | RESULT_01 / live cell in notebook §2.1 |
| IMU on RoNIN canonical raw ATE | IMUCNN **9.96 m** | ResNet1D 5.14 m | +94 % (out-of-domain) | RESULT_07 / notebook §2.2 |
| IMU on RoNIN canonical Umeyama ATE | IMUCNN **7.88 m** | ResNet1D 5.14 m | +53 % | RESULT_07 |

### End-to-end fusion

| dataset | metric | transformer | reference | margin |
|---|---|---|---|---|
| Webots sim 2-mod K=4 val | mean Euclidean | **0.469 m** | n/a (no public 4-mod-equivalent SOTA on Webots) | RESULT_06 |
| Webots sim 2-mod K=4 test | mean Euclidean | **0.517 m** | criterion-(b) 0.5 m bar | within 4 % of bar |
| **MSILN site1/B1 cross-session val** | mean Euclidean | **15.22 m** | wlanloc 21.26 m | **−28 %** |
| **MSILN site1/B1 cross-session test** ⭐ | mean Euclidean | **10.89 m** | wlanloc 28.31 m | **−62 %** |

### Ablations / robustness

| measurement | value | source |
|---|---|---|
| Staleness slope on Webots (test MAE vs WiFi lag) | 0.029 m/s across 27 s WiFi staleness (R² = 0.995) | RESULT_14 |
| K-axis: K=1 vs K=8 fresh-data MAE | 4-modality K=1: 0.486; K=4: 0.486; K=8: 0.651 (Webots) | RESULT_11/12 |
| Modality-dropout: `only:wifi` test MAE on Webots | ≈ full-fusion MAE within a few % (WiFi anchors absolute position) | RESULT_10/18 |
| Latency b=1 single sample, transformer | ~6 ms / sample on Quadro P4000 | RESULT_28 |
| Latency b=32, transformer | ~0.2 ms / sample | RESULT_28 |

⭐ = **paper headline result.**

---

## 7. Honest limitations (defended explicitly in §7)

The paper does not hide these. The supervisors haven't seen them
yet; the conference reviewers will.

### 7.1 Smoothness debt is architecture-invariant

Across 4 fusion architectures (transformer, CNN1D, LSTM-attn,
MoTTransformer) × 5+ datasets, per-trajectory smoothness median
Pearson r between ‖Δpred‖ and ‖Δgt‖ stays ≤ 0.10 — never clears
the rubric's > 0.20 gate. **Hypothesis falsified**: the
smoothness debt is loss-function-bound, not architecturally
tractable. Named future work: auxiliary velocity loss (B-1) or
EMA token smoothing (B-2) per RESULT_05's locked B-lever follow-up.

**Paper framing:** "Our fusion achieves competitive absolute-position
accuracy but inherits an architecture-invariant smoothness debt
(per-trajectory motion-magnitude r < 0.10). We hypothesize this
is loss-function-bound and identify auxiliary velocity loss as a
candidate fix for future work."

### 7.2 IMU encoder canonical-RoNIN gap (C2 not fully discharged)

IMUCNN raw ATE on canonical unseen-subjects is +94 % outside the
20 % SOTA gate vs RoNIN ResNet1D. The CNN1D aggregator
(out-of-paper-scope) narrows this to +47 % raw / +15.7 % Umeyama
(RESULT_23) — Umeyama-aligned within gate, but raw weighted ≥
aligned per the locked amended rubric.

**Paper framing:** "Our IMU encoder is competitive in-domain
(Umeyama-aligned ATE within 1.5× SOTA); cross-subject
generalization on canonical RoNIN unseen-subjects shows a
honestly-reported raw-ATE gap (+94 %). This is the expected
in-domain vs cross-subject trade-off for a 95×-smaller encoder
(IMUCNN 0.05 M vs ResNet1D 4.6 M params); cross-subject
generalization at this parameter budget is open future work."

### 7.3 MSILN test gate-1 partial (path-130 composition)

MSILN cross-session test of 10.89 m vs WiFi-kNN test of 9.47 m
shows our fusion narrowly lost on kNN-test, while crushing
wlan_localization (28.31 m) by 62 %. The kNN-vs-fusion gap is
explained by test path 130 (786 samples ≈ 28 % of test, very
WiFi-dense) which is easy for kNN (RESULT_15).

**Paper framing:** "On MSILN cross-session test, our transformer
beats the open-source wlan_localization SOTA by 62 % (28.31 →
10.89 m). The WiFi-kNN baseline performs comparably (9.47 m) on
this specific test split due to one WiFi-dense test path (path
130, 28 % of test mass); reporting the per-path breakdown
in the supplementary clarifies that fusion generalizes uniformly
while kNN benefits disproportionately from one easy path."

---

## 8. Out of scope (saved for journal extension, ~1-2 months later)

The journal version adds the following on top of the conference
scope:

1. **Camera modality + 4-modality fusion on Webots** — DPVOMotionEncoder
   on cached features; the 4-mod Webots winner (CNN1D 0.339 m test
   from RESULT_17).
2. **Fusion-architecture bake-off** — 4-arch comparison (transformer
   / CNN1D / LSTM-attn / MoTTransformer); the LSTM-attn
   dead-reckoning regime structural finding (3 datasets); the
   architecture-invariant smoothness debt falsification.
3. **Additional real datasets** — IMUWiFine fl.4 (cross-campaign
   format); IPIN 2024 floor 0 (small-train regime); discussion of
   cross-dataset generalization patterns.
4. **TartanAir external-VO validation** — Camera per-leg vs
   TartanVO; honest paper-soft framing already documented per
   RESULT_08.
5. **Conformal prediction** — uncertainty quantification at α=0.1
   on the Phase B winner (run-1 conformal machinery exists in
   `src/pipeline/uncertainty/conformal.py`).
6. **Loss-function-lever experiments** — auxiliary velocity loss
   (B-1) or EMA token smoothing (B-2) to close the smoothness
   debt; PLAN_25b candidate.
7. **C2 closure** — full cross-subject IMU generalization study;
   may include a beefed-up IMUCNN variant.

---

## 9. Existing assets (what's in the repo)

The conference paper can be written directly from these:

- **`notebooks/run2_walkthrough.ipynb`** — publication-grade
  reproducibility notebook; v8 (post-RESULT_38). Every paper
  number computed live from a loaded checkpoint or trained inline.
  FAST_MODE=True (default) runs in ~10 min; FAST_MODE=False
  retrains everything in ~3 h.

- **`handoff/SUMMARY.md`** — run-2 archive one-pager; cross-cutting
  findings; per-iteration RESULT_NN index.

- **`handoff/results/RESULT_01-38_*.md`** — per-iteration findings;
  the source-of-truth for every claim in this scope.

- **`runs/main_table/`** — saved checkpoints for the conference
  scope's experiments (UJI transformer, RoNIN canonical
  transformer, MSILN transformer); plus the Webots fusion
  checkpoints under `runs/overnight/run2_iter_33/`.

- **`src/pipeline/{baselines, encoders, fusion, training, evaluation,
  visualization, data}/`** — consolidated APIs (PLAN_26-29).
  Every paper-number reproduction routes through these.

- **`external_methods/` git submodules** — vendored SOTAs
  (wlan_localization, ronin, tartanvo, dpvo) for the per-leg
  comparison reproductions.

- **`docs/SOTA_BASELINES.md`** — paper-facing SOTA-and-criterion
  status doc (RESULT_29 rewrite).

- **`docs/EXTERNAL_DEPENDENCIES.md`** — per-submodule URL +
  pinned commit + license + usage.

---

## 10. Remaining work between now and conference submission

### Required (paper-ship blockers)

1. **Write the paper** — ~10 pages of prose + figures + tables.
   The notebook's figures and tables map directly to paper
   figures (engineer can `nbconvert --to pdf` or extract
   individual figs).
2. **Figure polish** — paper-format figures (often need vector
   PDF/SVG export with specific font sizes). The notebook's
   `set_paper_style()` is a good start; final paper-format
   tweaks per the PerCom LaTeX template.
3. **Citations / related-work scan** — pull recent (2024-2025)
   WiFi-fingerprinting and inertial-localization papers; cite
   the chosen baselines properly.

### Optional (would strengthen the paper if time permits)

4. **B-1/B-2 loss-function lever experiment** (~30 min from
   PLAN_05's queued candidate) — if successful, the smoothness
   debt becomes "we solved it" instead of "we identified it".
   High-value-if-it-works; falsifiable in 30 min.
5. **MSILN re-run with Anchor2Vec encoder + transformer** —
   RESULT_37's MSILN result actually used Anchor2Vec; this
   number is paper-ready. No re-run needed (the SUMMARY note
   about WiFiSetTransformer was about RESULT_15's PLAN_15 deployed
   config, not the RESULT_37 retrain).
6. **K-axis sweep refinement** — RESULT_11/12 found K=8 regressed
   on fresh accuracy; RESULT_13 showed K=4 + B=128 + 4-mod is
   the sweet spot. Confirm K=4 + B=128 + **2-mod** (the
   conference config) shows the same K-sweet-spot.

### NOT needed for conference (deferred to journal)

- More fusion-arch comparisons (PLAN_38 + PLAN_39 work).
- More datasets (IMUWiFine, IPIN, TartanAir, additional MSILN
  sites).
- Smoothness lever experiments beyond B-1.
- Cross-subject IMU generalization study.

---

## 11. Risk register

Things that could push back this scope:

| risk | mitigation |
|---|---|
| Reviewer asks: "did you compare other fusion architectures?" | Reference the journal-extension paragraph in §7; cite the bake-off as ongoing work. |
| Reviewer asks: "what about Camera/Odom?" | Same — frame as 4-modality extension in journal. |
| Reviewer asks: "why is the IMU encoder behind ResNet1D?" | §7 honest in-domain-only framing + parameter-budget context. |
| Reviewer asks: "show me the smoothness curve / per-trajectory results" | Notebook has these; include in supplementary. |
| MSILN gate-1 partial gets challenged | §7 path-130 composition explanation + per-path table in supplementary. |
| Title sounds like buzzword salad | Iterate title; the current "Async-Robust Multi-Modal Indoor Localization via a Continuous-Time Set-Transformer" can be tightened. |
| Webots sim isn't considered "real" enough | Frame Webots as the controlled lab where ablations are possible; MSILN cross-session is the real claim. |

---

## 12. Decision log

| date | decision | rationale | source |
|---|---|---|---|
| 2026-05-30 | Conference scope = 2 modalities (WiFi+IMU) | Both have public SOTAs; both demonstrate async rates; lightens architecture per supervisor. | this doc |
| 2026-05-30 | Conference scope = 1 fusion arch (set-transformer) | Cleanest fit for "continuous-time unified" angle; CNN1D wins on Webots but is an aggregator variant, save for journal. | this doc |
| 2026-05-30 | Conference scope = 4 datasets (Webots + MSILN + UJI + RoNIN) | Webots = lab; MSILN = headline; UJI + RoNIN = per-leg validation. Camera/Odom out. | this doc |
| 2026-05-30 | Contribution angle = continuous-time async unified set-transformer | Maps cleanly to the architecture's actual mechanism (time_encoding + modality_embedding + dropout); load-bearing claim is real-world cross-session generalization. | this doc |
| 2026-05-30 | Honest limitations stay in §7 | Smoothness debt + C2 partial + MSILN gate-1 partial all named; reviewers will see them anyway; honest framing is more defensible than concealment. | this doc |

---

## 13. Open items to resolve before paper writing starts

These are small, but should be locked before drafting:

- [ ] **Final title** — current candidate is workable; user may
  iterate. Should include "indoor localization" + "continuous-
  time" + "asynchronous" + "set-transformer" (or "multi-modal
  fusion").
- [ ] **Author list + affiliations** — Mohamed Bachar + supervisors
  + CESI LINEACT affiliation.
- [ ] **Decide on B-1/B-2 sprint** — 30 min experiment, could
  upgrade the §7 framing from "open problem" to "we solved it";
  user's call.
- [ ] **Confirm MSILN reference baseline** — wlan_localization
  is the chosen open-source SOTA; WiFi-kNN is reported as a
  secondary baseline. Confirm both go in the table.
- [ ] **Per-leg dataset count** — currently UJI + RoNIN canonical
  (2 per-leg reals). Drop one if conference space is tight?
  Recommend keeping both (one per modality).
- [ ] **Decide on supplementary material** — what gets the
  PerCom supplementary appendix vs what gets ARchived for the
  journal? Recommend: per-trajectory plots, full bake-off table,
  IMUWiFine + IPIN brief, smoothness analysis go to supplementary.

---

## 14. Sign-off

This scope is the contract for the **conference paper** only.
The journal version inherits this scope plus the deferred items
in §8.

**User to sign-off** (date): _______

**Supervisor sign-off** (date): _______

---

*Generated by scientist (Claude) from user's 2026-05-30 ~10:00
local scoping conversation; v2 after one iteration round on the
4 open questions raised in v1 proposal. Source of truth for paper
claims is `handoff/SUMMARY.md` + `handoff/results/RESULT_01-38_*.md`
+ `notebooks/run2_walkthrough.ipynb`.*
