# Adversarial Novelty Refutation — Fusion Domain

**Paper under attack:** "Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization" (ICINCO 2026)

**Novelty claim being tested:** NO prior WiFi+IMU (or comparable multi-sensor) indoor-localization
work combines ALL THREE of:
- (i) continuous-time / real-valued-Delta-t handling WITHOUT resampling/ODE,
- (ii) a SINGLE unified permutation-invariant attention/set block doing cross-modal AND cross-time fusion (not per-modality branches),
- (iii) explicit missing/stale-modality robustness (e.g. modality dropout) WITH cross-session generalization.

**Angle:** fusion-domain. **Notebook:** "Multimodal fusion (localization)"
(id `2d35a60b-d383-411e-89ad-2286bbe68580`, 16 sources).

**Method:** grounded NotebookLM queries; every load-bearing fact backed by a verbatim quote +
source_id. Verbatim quotes captured below.

---

## Verdict

**conjunction_holds = TRUE.** No single paper in this notebook holds the full (i)+(ii)+(iii) conjunction.

Two structural facts collapse the search before per-paper scoring even matters:
- **NO paper in this notebook uses a transformer or set-transformer at all.** Every DL fusion paper
  uses per-modality CNN/LSTM/DNN branches merged by concatenation / weighted sum / attention layer,
  or a classical Kalman/EKF/particle filter / factor graph. So criterion (ii) — a SINGLE
  permutation-invariant attention block over (modality,time) tokens — is met by ZERO papers.
- **NO paper encodes real-valued elapsed time (Delta-t / Time2Vec / sinusoidal continuous-time
  embedding) into a token.** Every async-aware paper resamples / windows / interpolates to a fixed
  grid, or uses an analytical solver. So criterion (i) is met by ZERO papers.

Because (i) and (ii) are each met by zero papers, the conjunction is unbreakable from this notebook
regardless of how many papers satisfy (iii).

---

## Per-candidate scoring (HAS / LACKS, with source_id + quote)

### Threat tier 1 — WiFi+IMU deep fusion WITH some attention/robustness

**SmartFPS (Hua, Zhuang, Yang, Frontiers in Neurorobotics 2023)** — src `6b420277-3636-484c-a42c-f8a1bde08581`
- HAS: an attention layer; cross-device / cross-pedestrian generalization via GAN transfer learning
  ("A transfer learning strategy for SmartFPS is proposed based on the generative adversarial network
  (GAN) to deal with device heterogeneity..."; "average positioning accuracy of 0.575 meters for
  different pedestrians and mobile phones").
- LACKS (ii): separate branches, NOT a unified set block — "SmartFPS is formed by four modules:
  (1) inertial encoder ... LSTM ...; (2) wireless encoder ... CNN ...; (3) attention layer;
  (4) fusion decoder ... LSTM network."
- LACKS (i): resamples to fixed 1s windows — "should also be down-sampled due to their uneven
  sampling frequencies. In this study, the inputs of both encoders are 1 s of the signal sequence."
- LACKS (iii)-as-modality-dropout: no modality dropout; relies on inertial encoder to carry through
  ("decreasing the number of beacons has a much smaller influence on SmartFPS...").
- Note: fuses Bluetooth+IMU (wireless = BLE/RSS), not WiFi specifically.

**Multi-Modal Recurrent Fusion (Yu, Wang, Koike-Akino, Orlik, MERL 2022)** — src `53d5d1d1-413f-4b4d-af7c-e6b3f4682cff`
- HAS (iii, robustness only): learned per-modality "relative importance weights" from immediate past
  hidden states to down-weight degraded sensors (NLOS / failures) — "we propose to project a
  concatenated 'immediately preceding' hidden states ... into a measure of modality quality, u, and
  normalize these ... to reflect the relative importance of each sensor at time t." Fuses WiFi(RSSI+CSI)+IMU+UWB.
- LACKS (ii): multi-stream LSTM with weighted sum, NOT a unified set block — "h_fusion = sum_m alpha_m h_t^m".
- LACKS (i): discrete fixed time steps — "infer the coordinate from the multi-sensor data over a
  certain time interval ... where T is the number of time steps".
- LACKS (iii, cross-session): random 80/10/10 split of one dataset — "an 80/10/10 splitting of
  Dataset1 among the training, validation, and testing sets". NOT cross-session.

**MM-Loc / Hybrid Multimodal DNN (Wei, Wei, Radu, Sensors 2021)** — src `75d15e66-b24b-4fc7-88e0-835ae66ccec0`
- HAS (iii, robustness only): explicit missing-WiFi handling via null vector (-100 dBm) shifting
  inference to inertial branch — "when there is no WiFi scan in the system, the WiFi input is a
  vector with all components value of 0 (normalised to -100 dBm). This null vector causes the
  inference to balance entirely on the inertial sensors side." Handles imbalanced sampling rates.
  Fuses WiFi RSS + inertial.
- LACKS (ii): two parallel branches (LSTM for sensors, DNN for WiFi) merged by concatenation —
  "two parallel single-modality feature extractors and a joint network structure to merge latent
  features at the top". NOT a unified set/attention block (no attention at all).
- LACKS (i): linear interpolation + resampling to fixed windows — "We use linear interpolation to
  fill the missing values ... grouped in time windows"; "we adjust the WiFi scan rate ... to every
  100 ms."
- LACKS (iii, cross-session): random 65/25/10 split across 2 buildings, NOT zero-shot cross-session
  — "with the following split radio: 65%, 25% and 10%".
- THIS IS THE CLOSEST PARTIAL MATCH on the robustness+async-handling combination (it is the only WiFi+IMU
  paper here that explicitly handles missing WiFi AND imbalanced rates), but it has neither the unified
  set-transformer, the continuous-time Delta-t, nor a cross-session split.

### Threat tier 2 — robustness / cross-session present, other modalities or classical fusion

**WIO-EKF (Zhou, Wang, Sun, Gravina, IEEE IoT-J 2024)** — src `de3074ce-40fb-443c-a4b9-860664fcac1a`
- HAS (iii, cross-session): explicit cross-DAY eval — "The time interval between data collection for
  the training and test sets is ten days." Plus per-AP masking-noise robustness via convolutional
  denoising autoencoder — "Wi-Fi fingerprint data with artificially added Gaussian noise and masking
  noises to reconstruct the original Wi-Fi fingerprints." Fuses WiFi+IMU.
- LACKS (ii): EKF fusion of two separate deep branches (CDAELoc + DbDIO), NOT a unified attention
  block — "an EKF-based Wi-Fi and inertial odometry (WIO-EKF) fusion ... utilizing the predicted
  results from the proposed CDAELoc and DbDIO models as the system observations".
- LACKS (i): fixed 1s time windows — "time window is 200 (i.e., 1 s)".
- LACKS (iii)-as-modality-dropout: masking is per-AP within WiFi, not whole-modality dropout.
- Strong on the cross-session axis; this is the second-closest partial match. But classical EKF,
  fixed windows, no set-transformer.

**DamLoc (Qinghu Wang et al., Future Gener. Comput. Syst. 2024)** — src `94456ab2-d40b-4ba2-9d0a-28f8d1e9a034`
- HAS: an attention mechanism in fusion; cross-device & cross-user eval — "DamLoc is more tolerant of
  different devices and users." A "kill all neurons" context-zeroing trick for kidnapped-robot.
- LACKS (ii): two-branch CNN — "we invoke a two-branch CNN ... to extract magnetic and BLE's ...
  features separately". NOT a unified set block.
- LACKS (i): interpolation — "we adopt the interpolation method to augment the BLE values".
- LACKS modality match: fuses magnetic + BLE (+context), not WiFi+IMU. Context-zeroing is a runtime
  state control, not random train-time modality dropout.

**WiMU (MobiSys 2025 demo, Yang et al.)** — src `31231955-1d3c-4f2e-8a14-64ca9c7c33f3`
- LACKS (ii): GNN (WiFi) + PDR (IMU) fused by particle filter — "the results from the WiFi module and
  the PDR algorithm are fused together by a particle filter".
- LACKS (i): PDR compensates between sparse WiFi scans; no Delta-t token encoding.
- LACKS (iii): no modality dropout reported (UNSUPPORTED). Robustness to RP reduction shown, but no
  explicit cross-session zero-shot split in provided text.

**ModDrop (Neverova, Wolf, Taylor, Nebout, 2014)** — src `d8cc9acd-5226-4ef3-b79c-b92226354d8a`
- HAS (iii, the canonical modality-dropout): "random dropping of separate channels (dubbed ModDrop)"
  / each modality component "is dropped (set to 0) with a certain probability ...". This is the
  prior art our modality-dropout descends from — must be cited as the origin of the technique.
- LACKS domain: gesture recognition, NOT localization (no WiFi, no IMU).
- LACKS (ii): parallel modality-specific CNN paths, late additive fusion — "single-scale paths
  connected in parallel ... Predictions from all paths are aggregated through additive late fusion".
- LACKS (i): fixed temporal-stride windows — "sampled with a given temporal stride s and concatenated
  to form a spatio-temporal 3D volume".

### Threat tier 3 — explicit "asynchronous" in title (async axis only)

**Asynchronous Multi-Sensor Fusion for 3D Mapping (Geneva, Eckenhoff, Huang, ICRA 2018)** — src `59d94997-2d16-4a78-801d-ea9df0ccfaab`
- HAS (i, partial — the closest anyone gets to async-without-resampling): analytical linear 3D pose
  interpolation in a factor graph, adds asynchronous measurements without new nodes — "we accurately
  align both asynchronous unary and binary graph factors ... based on our analytically derived linear
  3D pose interpolation ... without the need for extra nodes ... or for the naive ignoring of the
  measurement delay."
- LACKS (i, the DL form): it is an ANALYTICAL solver inside a factor graph, NOT a learned
  continuous-time neural Delta-t token encoding. Different mechanism entirely.
- LACKS (ii): factor-graph optimization, not a set-transformer.
- LACKS modality match & (iii): fuses LIDAR + stereo vision + RTK GPS, not WiFi+IMU — "we fuse
  odometry measurements from ... stereo and LIDAR modules ... with a RTK GPS unit". No modality dropout.

### Other sources (dataset/review/classical — not architecture threats)
- `1bb05d0a` Abdalla et al. (Data in Brief 2025): dataset (WiFi+IMU+CCTV), W3KNN+KF/PF baseline. Not a fusion architecture.
- `1c454340` Chen et al. (Sensors 2015): WiFi+PDR+landmarks via linear Kalman filter. Classical, no DL, no attention.
- `b0e54375` Silva et al. (Data 2023): industrial WiFi+IMU+odometry dataset; kNN + dead reckoning baselines. Dataset.
- `4d20d630` PEOPLEx (Lajoie, Beltrame 2023): IMU+UWB+BLE+WiFi via factor-graph loop closures. Classical optimization, no set block, no Delta-t token.
- `5fcd4079` Zhang et al. (IEEE IoT-J 2021): WiFi+PDR via single LSTM on displacement features; unifies rates by resampling to 20 Hz — "the unified sampling frequency is set to 20Hz." No attention, no Delta-t token, no modality dropout.
- `51755f40` Wang & Ahmad (review 2025): survey of AI AMR localization. Review.
- `c3c7d669` Lukasik et al. (Sensors 2024): systematic review of image-based multimodal localization. Review.
- `59d94997` covered above.

---

## Closest partial matches (ranked)

1. **MM-Loc (src 75d15e66)** — closest on the (iii)-robustness + async-rate-handling combination for
   WiFi+IMU: explicitly handles missing WiFi (null vector) and imbalanced sampling rates in one
   network. But it does this with parallel branches + interpolation/resampling, has NO attention/set
   block, NO Delta-t encoding, and uses a random in-building split (no cross-session). Misses (i) and (ii) outright.
2. **WIO-EKF (src de3074ce)** — closest on the cross-session axis (genuine 10-day train/test gap) for
   WiFi+IMU, with DAE robustness. But classical EKF fusion of separate branches, fixed 1s windows.
   Misses (i) and (ii) outright.
3. **Geneva et al. 2018 (src 59d94997)** — closest on the "asynchronous without resampling" idea, but
   via analytical factor-graph interpolation (not a learned continuous-time token), wrong modalities
   (LIDAR/stereo/GPS), no robustness mechanism. Misses (ii), (iii), and the LEARNED form of (i).

---

## Residual risk a reviewer could raise

- A reviewer could argue our three pieces are each individually well-known prior art assembled here:
  modality dropout = ModDrop (src d8cc9acd, verbatim "ModDrop" channel dropping); missing-modality /
  imbalanced-rate handling for WiFi+IMU = MM-Loc (src 75d15e66, null-vector trick); cross-session /
  cross-day WiFi+IMU evaluation = WIO-EKF (src de3074ce, 10-day gap) and SmartFPS (src 6b420277,
  cross-device GAN transfer); asynchronous-without-resampling = Geneva 2018 (src 59d94997, analytical
  interpolation). The reviewer's line: "each ingredient exists; the contribution is incremental
  engineering."
- DEFENSE the paper must make explicit: the novelty is the CONJUNCTION plus the specific mechanism —
  a SINGLE permutation-invariant set-transformer where self-attention over (modality,time) tokens IS
  the fusion, with a LEARNED continuous-time Delta-t token encoding (no resampling, no ODE), trained
  with modality+instant dropout, evaluated cross-session. From THIS notebook, zero papers use a
  transformer/set-transformer and zero use a learned Delta-t token, so (ii) and the DL form of (i) are
  uncontested in the fusion-domain corpus.
- Secondary risk: ModDrop, the origin of modality dropout, must be CITED (not claimed as ours). Our
  per-instant dropout is the temporal-axis extension; cite ModDrop and frame ours as the extension to
  the (modality,time) token set.
- Caveat on grounding: WiMU and ModDrop cross-session status returned UNSUPPORTED in the provided
  source text; we did NOT claim either as a counterexample.
