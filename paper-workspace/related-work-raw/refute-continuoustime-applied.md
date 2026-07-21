# Adversarial Novelty Refutation — Angle: continuous-time APPLIED to localization

**Date:** 2026-06-05
**Refuter angle:** Continuous-time / irregular-sampling models APPLIED to localization or WiFi/IMU fusion.
**Primary notebook:** "Continuous-time / async / irregularly-sampled" (id `49ffba8a-0198-4c43-99a9-34c487d983ae`, 15 sources).
**Our claim under attack:** No prior work, for WiFi+IMU (or comparable multi-sensor) indoor localization, combines ALL of
- (i) continuous-time / real-valued Delta-t handling WITHOUT resampling and WITHOUT an ODE solver,
- (ii) a SINGLE unified permutation-invariant attention/set block doing cross-modal AND cross-time fusion (not per-modality branches),
- (iii) explicit missing/stale-modality robustness (modality dropout) with cross-session generalization.

---

## Method

1. Inventoried all 15 sources in the assigned notebook (titles + source_ids + domain).
2. Hard-probed the only two *localization-applied* papers (Eang & Lee; Feng et al.) for the full conjunction.
3. Hard-probed the four strongest *architectural* matches (SeFT, STraTS, mTAN, Raindrop) — these have set/triplet + continuous-time + (some) missing-data handling, but in clinical domain.
4. Checked ContiFormer + DGODE (continuous-time transformer; multimodal ODE) for WiFi/IMU + localization + missing-modality.
5. Ran a cross-notebook query over all 5 relevant corpora to guard against the "it exists in another corpus" risk.
All quotes below are verbatim from NotebookLM grounded answers (source_ids retained). No numbers invented.

---

## Source inventory (assigned notebook)

| # | Paper | source_id | Domain | Relevance |
|---|-------|-----------|--------|-----------|
| 1 | Che 2018 GRU-D | `974ad9a2-f9f6-4961-a704-c4d3faa53c27` | clinical | missing-value RNN, discrete |
| 2 | Chen 2018 Neural ODE | `107ff554-9251-474b-b6df-0b702d209c4c` | generic | ODE solver (we explicitly avoid) |
| 3 | Chen 2024 ContiFormer | `d4fc0e36-b596-432d-aa89-570df59ec51c` | generic irregular TS | cont-time transformer but ODE-based |
| 4 | De Brouwer 2019 GRU-ODE-Bayes | `793ea3c3-c41f-451a-91fa-a26f4eeedca8` | clinical/climate | ODE-based |
| 5 | Eang & Lee 2024 DNN-EKF UWB | `369ed09c-e1b9-4864-b48a-dfc1480a0c41` | **indoor localization** | LOCALIZATION-APPLIED #1 |
| 6 | Feng 2023 KF+NN review | `eba958aa-3ab3-472f-ba9c-9f81fdaa3bb0` | state estimation review (incl. positioning) | LOCALIZATION-APPLIED #2 |
| 7 | Horn 2020 SeFT | `2c3d6b02-6b5b-440e-8d86-2adaac65c3b5` | clinical | ARCHITECTURE MATCH (i)+(ii) |
| 8 | Kazemi 2019 Time2Vec | `6155e781-d6f4-4bf7-b90b-389f1c6a3b86` | generic | the time-encoding primitive we build on |
| 9 | Kidger 2020 Neural CDE | `5e708532-97cc-47cf-9c67-ebb79cfdcd8f` | generic irregular | CDE-based |
| 10 | Rubanova 2019 Latent ODE | `cbcd353a-7b15-4fc7-875e-719b3c3648b2` | generic irregular | ODE-based |
| 11 | Shou 2024 DGODE | `92ad96ff-e6e2-4b3e-adec-02664f914869` | multimodal emotion (T/A/V) | multimodal + missing-modality but ODE + not localization |
| 12 | Shukla 2019 IPN | `586bbf61-d165-453a-8b7e-ac12d8bfb3c5` | clinical | interpolation onto grid (resampling) |
| 13 | Shukla 2020 mTAN | `cfe566d2-e92a-41ca-8f51-5ac914fdf4c1` | clinical + HAR | ARCHITECTURE-ADJACENT (separate-stage) |
| 14 | Tipirneni 2022 STraTS | `a7665dca-e7b5-4e66-a1ad-522f0fc4b8ba` | clinical | ARCHITECTURE MATCH (i)+(ii) |
| 15 | Zhang 2022 Raindrop | `a068da78-aa9a-4607-96cb-c5f6f8b032f6` | clinical + HAR | ROBUSTNESS MATCH (iii) + cross-group gen |

Only 2 of 15 are localization-applied (Eang & Lee, Feng). Neither is WiFi+IMU; both are KF/MLP, discrete-time.

---

## Candidate-by-candidate verdict

### Eang & Lee 2024 (DNN-EKF UWB) — `369ed09c...`
- (i) cont-time Delta-t: **LACKS.** "For each time step t: ... Apply EKF()" — discrete steps.
- (ii) unified set-attn: **LACKS.** "At the core of the algorithm is the Multi-layer Perceptron (MLP) Regressor" + EKF. No transformer/set.
- (iii) missing-modality robustness + cross-session: **LACKS.** Uses random split "training set of 60% and a testing set of 40%." No modality dropout, no cross-session.
- Modalities: UWB ranges (NOT WiFi+IMU): "By employing distance measurements between the UWB tag and the anchors..."
- **Verdict: NOT a counterexample. Has only the localization *domain*; lacks all three pillars.**

### Feng et al. 2023 (KF+NN review) — `eba958aa...`
- (i): LACKS — "X_{k+1}=AX_k+Bu_k+w_k; Z_k=HX_k+v_k" discrete state-space.
- (ii): LACKS — reviews "artificial NN, feedforward NN, recurrent NN and back propagation NN" + Kalman. No set-transformer.
- (iii): LACKS — generic "generalization ability is also enhanced," no modality dropout / cross-session protocol.
- Modalities: generic state estimation (battery SOC, wind, positioning), not WiFi+IMU.
- **Verdict: NOT a counterexample (it is a survey; no method combining the three).**

### Horn 2020 SeFT — `2c3d6b02...`  [CLOSEST ARCHITECTURE]
- (i) cont-time Delta-t no-ODE: **HAS.** "the time encoding converts the 1-dimensional time axis ... through multiple trigonometric functions of varying frequencies." No ODE.
- (ii) single permutation-invariant set, cross-var + cross-time jointly: **HAS.** "we propose to rephrase the problem of classifying time series as classifying a set of observations"; "an unordered set of measurements, where all information is conserved because the observation time is included for each set element." Explicitly "not having to observe all modalities at each time point."
- (iii) explicit missing-modality robustness test + cross-session: **LACKS (no explicit leave-sensor-out benchmark; no cross-session protocol).** Naturally handles non-sync but does not benchmark dropped sensors.
- Domain: **clinical only** — "predict patient mortality on two datasets ... predict the onset of sepsis." NOT localization, NOT WiFi+IMU.
- **Verdict: NOT a counterexample. Has (i)+(ii) but clinical, no (iii)-test, no localization.**

### Tipirneni 2022 STraTS — `a7665dca...`  [CLOSEST ARCHITECTURE]
- (i): **HAS.** Continuous Value Embedding "to embed continuous times and measured values" via FFN, no ODE, no discretization.
- (ii): **HAS.** "our model regards each time-series as a set of observation triplets (time, variable, value) without ... aggregation or imputation" -> single Transformer over the set.
- (iii): **LACKS** explicit leave-sensor-out / modality dropout test; no cross-session.
- Domain: **clinical only** — "two real-world EHR databases for the mortality prediction task." NOT localization, NOT WiFi+IMU.
- **Verdict: NOT a counterexample. (i)+(ii) clinical, no (iii)-test, no localization.**

### Shukla 2020 mTAN — `cfe566d2...`
- (i): HAS (learned continuous-time sine+linear embedding, no ODE).
- (ii): **PARTIAL/LACKS unified block** — attention is computed per-dimension then linearly combined: "softmax ... over the observed time points t_id for dimension d" + "final linear combination across ... data dimensions." So cross-time and cross-variable are *separate stages*, not one permutation-invariant set block.
- (iii): LACKS explicit modality-dropout test; no cross-session.
- Domain: clinical + Human Activity. NOT localization, NOT WiFi+IMU.
- **Verdict: NOT a counterexample.**

### Zhang 2022 Raindrop — `a068da78...`  [CLOSEST ON ROBUSTNESS]
- (i): HAS (timestamps through trigonometric functions, no ODE).
- (ii): **LACKS the *single unified* block** — hierarchical/separate stages: "embeds individual observations considering inter-sensor dependencies ... aggregates them into a sensor embedding using temporal attention ... finally integrates sensor embeddings into a sample embedding." Cross-sensor (graph msg-passing) then cross-time (temporal attn) are distinct levels.
- (iii) missing-modality robustness: **HAS.** "Leave-fixed-sensors-out ... we test whether RAINDROP can achieve good performance when a subset of sensors are completely missing." AND cross-group generalization: "split the data into two groups ... use one group as a train set and ... the other group [as] validation and test set."
- Domain: **clinical + human activity** ("three healthcare and human activity datasets"). NOT localization, NOT WiFi+IMU.
- **Verdict: NOT a counterexample. Has (i)+(iii) incl. a cross-group generalization protocol, but architecture is multi-stage (lacks (ii)) and domain is clinical/HAR.**

### Chen 2024 ContiFormer — `d4fc0e36...`
- (i): cont-time transformer but **ODE-based** ("incorporates ... continuous dynamics of Neural ODEs with the attention") — we explicitly avoid ODE; also relies on cubic-spline interpolation for queries.
- (ii): single transformer but designed for generic single-stream irregular TS, not declared cross-modal-set fusion.
- (iii): only random-observation drop ("randomly drop either 30%, 50% or 70% observations"), NOT whole-modality leave-out; no cross-session.
- Domain: generic ("interpolation, classification, and prediction"). NOT localization, NOT WiFi+IMU.
- **Verdict: NOT a counterexample (ODE-based; not localization).**

### Shou 2024 DGODE — `92ad96ff...`
- Multimodal (Text/Audio/Video) + tests modality subsets (T, A, V, T+A, T+V, T+A+V) -> has a missing-modality flavour.
- But ODE/graph-ODE based (not the no-ODE claim), emotion recognition (NOT localization), not WiFi+IMU, not a single permutation-invariant set block.
- **Verdict: NOT a counterexample.**

---

## Cross-notebook sweep (guard against "exists in another corpus")

Ran `cross_notebook_query` over WiFi fingerprinting / Inertial-IMU / Multimodal-fusion(localization) / Attention-transformer-set / Continuous-time. All 5 notebooks returned: **no full-conjunction paper for WiFi+IMU localization.** Newly surfaced partial matches worth recording as residual-risk candidates (source_ids from their own notebooks):

- **Aristorenas 2025** (WiFi fingerprinting, `f6417660-...`): permutation-invariant Set Transformer for indoor localization — but **single-modality WiFi RSSI scans only**, static (no cross-time tracking), no IMU, no continuous Delta-t. Quote: "permutation-invariant neural architecture for indoor localization using RSSI scans ... unordered set of (BSSID, RSSI) pairs."
- **Bhatia 2025 Locaris** (WiFi fingerprinting, `c4f1526c-...`): unified decoder-only Transformer fusing WiFi RSSI+FTM with explicit missing-modality robustness ("provide whichever modality is available ... without requiring placeholders") — but **no IMU, no continuous-time encoding, no set-transformer (LLM token sequence).**
- **Cohen & Klein 2024 A-KIT / ST-BeamsNet** (Attention/Inertial, `8b22a8f8-...` / survey `cd200533-...`): Set-Transformer for navigation fusion with complete-sensor-outage robustness — but **underwater IMU+DVL, fixed 1-s synchronized windows, EKF-based fusion**, not indoor WiFi+IMU.
- **Wei et al. 2021** (Multimodal-fusion-localization, `75d15e66-...`): WiFi+IMU smartphone tracking that handles missing WiFi via a null vector ("when there is no WiFi scan ... the WiFi input is a vector with all components value of 0 ... balance entirely on the inertial sensors") — but **separate LSTM/DNN branches, linear interpolation onto fixed windows**, no set-transformer, no continuous Delta-t.
- **AFT-VO (Kaygusuz 2022)** (Attention/transformer/set, `44dce69c-...`): transformer fusing asynchronous cameras with time-discrepancy handling — but **camera-only, uses a Discretiser (binning), no modality dropout.**
- **Neverova 2014 ModDrop** (Multimodal-fusion-localization, `d8cc9acd-...`): the canonical modality-dropout mechanism — but **gesture recognition, per-modality branches, synchronized frames.**

Each is a single-pillar (or two-pillar) match; none has (i)+(ii)+(iii) for WiFi+IMU localization.

---

## VERDICT

**conjunction_holds = TRUE** — no single paper in any of the 5 corpora has the full (i)+(ii)+(iii) conjunction for WiFi+IMU indoor localization.

**Closest partial matches (this angle):**
- *Architecture (i)+(ii):* SeFT (`2c3d6b02...`) and STraTS (`a7665dca...`) — single permutation-invariant set/triplet + continuous-time no-ODE — but clinical-only, no explicit modality-dropout/cross-session test, not localization.
- *Robustness (iii) + generalization:* Raindrop (`a068da78...`) — explicit leave-sensor-out + cross-group generalization + continuous-time — but multi-stage architecture (not a single unified block) and clinical/HAR, not localization.

**Residual risk a reviewer could raise:** Each of the three pillars is individually well-established in the *clinical* irregular-time-series literature (SeFT/STraTS for unified set + continuous-time; Raindrop for leave-sensor-out + cross-group transfer; ModDrop for modality dropout). A skeptic can argue our contribution is "porting a known clinical set-transformer recipe to WiFi+IMU localization" — i.e., novel *application* + *conjunction* but no fundamentally new mechanism. Defense must lean on: (a) it is the FIRST to bring this unified continuous-time set-transformer to WiFi+IMU indoor localization, (b) the specific async profile (1 Hz WiFi vs 30 Hz IMU, stale-WiFi degradation) and (c) real-world CROSS-SESSION evaluation, which none of the architecture-matching papers (SeFT/STraTS) perform and which is the actual hard part for WiFi fingerprints.
