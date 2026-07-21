# Related Work Evidence — Group: Attention / Transformer / Set

Notebook: `f635555d-7cf3-4eb6-ad14-6e2441559b2b`
Gatherer pass for ICINCO 2026 paper "Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization".
Our 3 contributions: (i) continuous-time learned sinusoidal encoding of real-valued Delta-t (no resampling, no ODE); (ii) ONE permutation-invariant set-transformer doing cross-modal AND cross-time fusion at once; (iii) async-robustness via modality-dropout (0.4) + instant-dropout (0.45) demonstrated by real-world cross-session generalization.

Source map:
- f7cf37ed = Vaswani et al. 2017 Attention is All You Need (PILLAR)
- a85a3ae2 = Lee et al. 2019 Set Transformer (PILLAR — our (ii))
- 2e61db11 = Zaheer et al. 2017 Deep Sets (PILLAR — permutation-invariance theory)
- 44dce69c = Jaegle et al. 2021 Perceiver (PILLAR/context)
- a66f7a26 = Jaegle et al. 2022 Perceiver IO (PILLAR/context)
- 8b22a8f8 = Kaygusuz et al. 2022 AFT-VO (COMPETITOR — async + attention fusion; highest priority)
- fb531c11 = Cohen & Klein 2024 A-KIT (COMPETITOR)
- 1d12937b = Xiao et al. 2024 EffLoc (COMPETITOR)
- 5cd52555 = Lin & Evans 2025 SCM-PR (COMPETITOR)
- 3b94d89f = Diaz-Guerra et al. 2023 PI-RNN (COMPETITOR; actually Forum Acusticum 2023, NOT 2024)

---

## A. CAPSULES

### COMPETITOR 1 (highest priority) — AFT-VO (Kaygusuz et al. 2022) [8b22a8f8]
- **Method**: Per-camera Mixture Density Network estimates 6-DoF pose + uncertainty; a transformer-based fusion module ("AFT-VO") fuses these asynchronous per-camera pose estimates. Two novel encodings: a "Discretiser" (positionally encodes continuous timestamps by binning) and "Source Encoding" (distinguishes which camera each prediction came from).
- **Modalities**: Multiple ASYNCHRONOUS cameras only (multi-view visual odometry). NOT WiFi, NOT IMU in the fusion module. "Our framework combines predictions from asynchronous multi-view cameras and accounts for the time discrepancies of measurements coming from different sources." [8b22a8f8]
- **Time handling**: Discretises continuous time into BINS, then uses Vaswani-style positional encoding indexed by bin index. NOT real-valued continuous time. "To address this issue we propose to discretise the continuous time domain into bins... We then divide the time axis into smaller chunks, Z, and group the measurements into bins that have been captured close in time." [8b22a8f8] Explicitly contrasts with interpolation: "our AFT-VO model does not need such an interpolation step due to the proposed time encoding and hence can fuse asynchronous camera information in real-time." [8b22a8f8]
- **Robustness**: Multiple cameras give robustness to individual CAMERA failure — but no modality-dropout / no missing-modality training mechanism. "employing multiple cameras is a clear way to provide robustness to individual camera failures." [8b22a8f8]
- **Datasets**: nuScenes, KITTI.
- **Headline result (quote-grounded)**: nuScenes Avg RPE (Mean +/- std): Daylight 0.031 +/- 0.023, Rain 0.029 +/- 0.024, Night 0.053 +/- 0.037. "AFT-VO (ours) ... 0.031 +/- 0.023 [Daylight] ... 0.029 +/- 0.024 [Rain] ... 0.053 +/- 0.037 [Night]" [8b22a8f8]. "AFT-VO achieves the best overall mean performance across all sequences" on KITTI [8b22a8f8].
- **Limitation**: Vision-only; time is bucketed/quantised (loses real-valued Delta-t resolution; bin size Z is a hyperparameter); evaluated across weather/lighting splits, not cross-session relocalization; no missing-modality robustness mechanism.
- **diff_vs_ours**: This is the closest async+attention competitor. We DIFFER on all three contributions: (i) we encode REAL-VALUED Delta-t with a learned sinusoidal embedding — no binning/quantisation; (ii) AFT-VO fuses per-camera pose PREDICTIONS (late fusion of homogeneous vision streams) whereas we run ONE set-transformer over heterogeneous (modality, time) tokens of WiFi+IMU; (iii) we add modality/instant dropout for missing-modality robustness and demonstrate cross-session generalization. AFT-VO is single-modality (vision) — no WiFi/IMU localization.
- **key_quote**: "we propose to discretise the continuous time domain into bins." [8b22a8f8]

### COMPETITOR 2 — A-KIT (Cohen & Klein 2024) [fb531c11]
- **Method**: Adaptive Kalman-Informed Transformer. A set-transformer regresses the EKF process-noise covariance matrix online; hybrid of EKF + deep set-transformer ("Kalman-informed loss").
- **Modalities**: Inertial Navigation System (IMU specific force + angular velocity) + Doppler Velocity Log (velocity). "our approach to nonlinear sensor fusion based on an inertial navigation system and Doppler velocity log." [fb531c11]
- **Time handling**: Different rates handled by the EKF, NOT by the transformer; fixed 1-second windows of 100 samples. The EKF predicts in a loop until a valid observation arrives. "The INS operates at a rate of 100 [Hz] and the DVL at 1 [Hz]." [fb531c11]; "For each, a one-second window was taken, meaning one hundred samples." [fb531c11]; "The A-KIT cycle begins with the initialization of the EKF, followed by the execution of the prediction phase in a loop until a valid observation is obtained." [fb531c11] => multi-rate is a Kalman filter mechanism + fixed-rate windows, NOT a learned real-valued time embedding.
- **Robustness**: No explicit missing/stale-modality dropout; relies on EKF prediction loop between observations. No modality-dropout mechanism quoted.
- **Datasets**: Real AUV data, Snapir AUV, Mediterranean Sea (INS/DVL).
- **Headline result (quote-grounded)**: "A-KIT outperforms the conventional EKF by more than 49.5% and model-based adaptive EKF by an average of 35.4% in terms of position accuracy." [fb531c11]; per-trajectory: "67.8% improvement in VRMSE and a 49.5% improvement in PRMSE compared to the EKF" [fb531c11]; "more than 90% ... VRMSE and 87.7% ... PRMSE" on the other test trajectory [fb531c11]. (Relative improvements only — no absolute MAE quoted; flag as relative.)
- **Limitation**: The transformer does NOT do the fusion — the EKF does; the set-transformer only tunes Q. Underwater INS/DVL domain, not indoor WiFi/IMU. Multi-rate handled by Kalman machinery, not a continuous-time token embedding. Reports only relative improvement over EKF.
- **diff_vs_ours**: Both use a SET-TRANSFORMER, so this is an important comparator for (ii). But A-KIT is a Kalman-informed hybrid: the transformer regresses noise covariance, the EKF performs the actual state fusion. We instead use ONE set-transformer as the END-TO-END fusion block (no EKF, no ODE). A-KIT handles multi-rate via the EKF + fixed windows; we use a learned real-valued Delta-t embedding. A-KIT has no modality-dropout robustness and reports per-trajectory (not cross-session) tests.
- **key_quote**: "Built upon a set-transformer network, A-KIT is designed for real-time adaptive regression of the process noise covariance matrix." [fb531c11]

### COMPETITOR 3 — EffLoc (Xiao et al. 2024) [1d12937b]
- **Method**: Lightweight efficient Vision Transformer (EfficientViT backbone) for single-image 6-DoF camera relocalization; introduces Sequential Group Attention (SGA), Overlap Patch Embedding.
- **Modalities**: SINGLE camera image only. "We propose EffLoc, a novel efficient Vision Transformer for single-image camera relocalization." [1d12937b]; "learn 6-DoF camera poses from single images." [1d12937b]
- **Time handling**: NONE. Static single-image pose regression; no asynchronous/multi-rate/continuous-time handling at all.
- **Robustness**: No missing/stale-sensor mechanism (single modality). Robustness claims are to visual disturbances (lighting/occlusion/dynamic objects).
- **Datasets**: Oxford RobotCar (also 7-Scenes implied by AtLoc/MapNet baselines).
- **Headline result (quote-grounded)**: "The mean position accuracy is enhanced from 25.39m to 7.58m on LOOP1 and from 28.89m to 7.89m on LOOP2." [1d12937b]; "reduces the mean rotation error from 17.49 deg to 3.72 deg on LOOP1 and from 19.65 deg to 4.19 deg on LOOP2." [1d12937b]; "33.9% and 24.7% improvements in FULL1 and FULL2 routes compared with MapNet by using only a single image." [1d12937b]; "using 49.7% fewer Flops with 9.6% lower position and rotation error vs. AtLoc." [1d12937b]
- **Limitation**: Single-modality vision; no fusion at all; no temporal/async modelling; outdoor driving relocalization, not indoor WiFi/IMU.
- **diff_vs_ours**: Shows the transformer/ViT trend in localization but is the OPPOSITE of our problem — single static image, no sensor fusion, no time, no robustness to missing modalities. Useful only as evidence that "transformer-for-localization" is established but NOT applied to async multi-rate WiFi+IMU.
- **key_quote**: "We propose EffLoc, a novel efficient Vision Transformer for single-image camera relocalization." [1d12937b]
- **NOTE**: Cross-session — EffLoc evaluates on Oxford RobotCar across days/weather (train/test on different recording dates, sunlight vs cloudy). It DOES report cross-session/cross-weather generalization, but single-modality. "captured biweekly for more than a year, encompassing a diverse array of environmental conditions" [1d12937b]; "testing datasets in LOOP sequence recorded under direct sunlight, whereas FULL datasets are captured under cloudy conditions." [1d12937b]

### COMPETITOR 4 — SCM-PR (Lin & Evans 2025) [5cd52555]
- **Method**: Semantic-Enhanced Cross-Modal Place Recognition. VMamba backbone for RGB; pre-trained 3D semantic segmentation for LiDAR; NetVLAD global descriptors + cross-modal semantic attention; contrastive learning. Task = retrieve correct LiDAR-map location for a query RGB image (place recognition / R@1), not (x,y) regression.
- **Modalities**: Monocular RGB image + pre-built 3D LiDAR point-cloud map. "The SCM-PR framework takes a query RGB image I_RGB and a pre-built 3D LiDAR point cloud map P_LiDAR as inputs." [5cd52555]
- **Time handling**: NONE quoted; no asynchronous/multi-rate handling.
- **Robustness**: Robustness to illumination/seasonal change (via semantic cues), NOT missing/stale-modality dropout. "To evaluate this robustness, we curated challenging subsets from the KITTI-360 dataset, specifically focusing on sequences captured at different times of day and across distinct seasons." [5cd52555]
- **Datasets**: KITTI, KITTI-360.
- **Headline result (quote-grounded)**: "on the KITTI dataset, our method achieves a Recall@1 of 62.58%, outperforming ModalLink's 61.22%." [5cd52555]; "on KITTI-360, we achieve 53.45% Recall@1 compared to ModalLink's 52.18%." [5cd52555]; night-time "43.19% R@1" vs ModalLink "39.55%"; winter "44.50%" vs "40.21%" [5cd52555].
- **Limitation**: Place-recognition retrieval (R@1), not metric (x,y) regression; cross-modal RGB-to-LiDAR (uses attention but only cross-modal, not cross-time); no async/multi-rate; no missing-modality dropout.
- **diff_vs_ours**: Uses cross-modal attention but the modalities (RGB, LiDAR map) and task (retrieval) differ entirely; no time handling; robustness is to appearance change, not sensor dropout. Confirms attention-for-cross-modal-localization is a trend but leaves async + missing-modality + (x,y) regression open.
- **key_quote**: "augmented with a novel cross-modal semantic attention mechanism that enables the RGB image descriptor to focus on corresponding semantic regions within the LiDAR map" [5cd52555]

### COMPETITOR 5 — PI-RNN (Diaz-Guerra et al. 2023) [3b94d89f]
- **Method**: Permutation-Invariant RNN for sound-source tracking. Replaces RNN input/state vectors with unordered SETS of embeddings; uses a multi-head attention module to match new input embeddings to state embeddings; invariant to input-set permutations, equivariant to state-set permutations.
- **Modalities/inputs**: Spatial detections from a sound-source-localization network (simulated speech from LibriSpeech, reverberant rooms). Acoustic, not WiFi/IMU.
- **Att vs RNN**: HYBRID — recurrent architecture using multi-head attention internally for the set matching. "In order to match every new embedding of the input set with the embeddings of the state set, we can use a multi-head attention module [..] which is well known for its use in transformer models and is invariant to the permutation of the elements of its input sets" [3b94d89f]
- **Permutation invariance**: "the generated set C(t) is invariant to the permutations of X(t) and equivariant to the permutations of H(t-1)" [3b94d89f]
- **Task**: multi-source localization and tracking.
- **Headline result (quote-grounded)**: "the proposed PI-RNN clearly outperforms the baselines in terms of localization error and the frequency of the identity switches" [3b94d89f]. (No absolute number quoted; results in a figure.)
- **Limitation**: Acoustic SST domain; permutation invariance is over the set of SOURCES, not over (modality, time) tokens; recurrent (not a single set-transformer); no WiFi/IMU; no continuous-time Delta-t embedding; no missing-modality dropout.
- **diff_vs_ours**: Demonstrates permutation-invariant attention applied to a localization/tracking task — supports our (ii) lineage — but it is RNN-based, single-domain (acoustic), and permutation invariance is over sources not over heterogeneous sensor tokens. No real-valued time embedding, no modality dropout.
- **key_quote**: "we present a permutation-invariant recurrent neural network (PI-RNN) that takes an unordered set of embeddings as input ... that is also an unordered set of embeddings with the information of every tracked trajectory." [3b94d89f]
- **VENUE CORRECTION**: Forum Acusticum 2023 / 10th Convention of the European Acoustics Association; "Copyright: (c)2023 David Diaz-Guerra et al." [3b94d89f]. The brief said 2024 — the source says 2023. Use 2023.

### PILLAR — Transformer (Vaswani et al. 2017, NIPS 2017) [f7cf37ed]
- **Method**: First sequence-transduction model based entirely on attention; self-attention multi-head; sinusoidal positional encoding.
- **key facts**: "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely." [f7cf37ed]. Positional encoding: sine/cosine of position index; "each dimension of the positional encoding corresponds to a sinusoid." [f7cf37ed]. CRITICAL for our gap: standard positional encoding indexes by DISCRETE position and "assume[s] the consecutive items to be equidistant in the time domain" (AFT-VO's restatement [8b22a8f8]). Venue: "31st Conference on Neural Information Processing Systems (NIPS 2017)" [f7cf37ed].
- **diff_vs_ours / what we borrow**: We borrow self-attention as the fusion operator but replace the fixed equidistant positional encoding with a learned sinusoidal encoding of REAL-VALUED elapsed time Delta-t — exactly the assumption (equidistant positions) we must break for asynchronous multi-rate sensors.

### PILLAR — Set Transformer (Lee et al. 2019, ICML 2019) [a85a3ae2]
- **Method**: Attention-based permutation-invariant neural network for set-input problems; encoder+decoder both attention-based; ISAB (induced set attention block, O(nm)) + PMA (pooling by multihead attention with k seed vectors). Proven permutation invariant + universal approximator for permutation-invariant functions.
- **key facts**: "We present an attention-based neural network module, the Set Transformer, specifically designed to model interactions among elements in the input set." [a85a3ae2]; "Proposition 1. The Set Transformer is permutation invariant." [a85a3ae2]; PMA: "We instead propose to aggregate features by applying multihead attention on a learnable set of k seed vectors S" [a85a3ae2]. Venue: ICML 2019 (PMLR 97) [a85a3ae2].
- **diff_vs_ours / what we borrow**: This is the direct architectural pillar for our contribution (ii). We instantiate a set-transformer over (modality, time) tokens so ONE self-attention block does cross-modal AND cross-time fusion at once. Set Transformer itself has NO notion of time/Delta-t and was not applied to async sensor fusion / localization (our novelty is the application + the continuous-time tokens + the dropout robustness).

### PILLAR — Deep Sets (Zaheer et al. 2017, NIPS 2017) [2e61db11]
- **Method**: Theory of permutation-invariant functions on sets. Theorem: any permutation-invariant set function decomposes as rho(sum phi(x)).
- **key facts**: "Theorem 2 A function f(X) operating on a set X having elements from a countable universe, is a valid set function, i.e., invariant to the permutation of instances in X, iff it can be decomposed in the form rho(sum_{x in X} phi(x)), for suitable transformations phi and rho." [2e61db11]; venue NIPS 2017 [2e61db11].
- **diff_vs_ours / what we borrow**: The theoretical justification that a permutation-invariant block over the set of (modality, time) tokens is well-founded — sensor order/rate must not matter. Deep Sets uses sum-pooling (no pairwise interactions); we use attention (Set Transformer) so tokens interact. Pure context/theory pillar.

### PILLAR/context — Perceiver (Jaegle et al. 2021, ICML 2021) [44dce69c]
- **Method**: Transformer-based general-perception model; asymmetric cross-attention distills large inputs into a small latent bottleneck; modality-agnostic.
- **key facts**: "the Perceiver -- a model that builds upon Transformers and hence makes few architectural assumptions about the relationship between its inputs" [44dce69c]; "leverages an asymmetric attention mechanism to iteratively distill inputs into a tight latent bottleneck" [44dce69c]; venue ICML 2021 (PMLR 139) [44dce69c].
- **diff_vs_ours**: Demonstrates the modality-agnostic, set-of-inputs attention paradigm we adopt, but Perceiver is general perception (images/audio/point clouds/video), NOT async multi-rate sensor fusion with real-valued time, NOT localization, NO missing-modality dropout. Context pillar.

### PILLAR/context — Perceiver IO (Jaegle et al. 2022, ICLR 2022) [a66f7a26]
- **Method**: Generalizes Perceiver to arbitrary structured outputs via a flexible output-query attention mechanism.
- **key facts**: "augments the Perceiver with a flexible querying mechanism that enables outputs of various sizes and semantics" [a66f7a26]; "decoding structured outputs ... directly from the Perceiver latent space" [a66f7a26]; venue ICLR 2022 [a66f7a26].
- **diff_vs_ours**: The query-based readout idea parallels our cross-attention readout, but again general-purpose I/O, not async sensor fusion / localization. Context pillar.

---

## B. COMPETITOR RUBRIC (gap matrix)

| Competitor (bibkey) | MODS | ATT | CT (real-valued Delta-t, no resample/ODE) | ROB (missing/stale) | XSESS | UNIFIED |
|---|---|---|---|---|---|---|
| AFT-VO (kaygusuz2022aftvo) | Multi-view cameras only (vision); NOT WiFi/IMU | yes (transformer fusion) | partial — discretises time into BINS + positional encoding; not real-valued; but no resampling/ODE | no (multi-camera redundancy only; no modality dropout) | no (weather/lighting splits, not cross-session) | hybrid — per-camera MDN pose preds then one transformer fuses (late fusion of homogeneous streams) |
| A-KIT (cohen2024akit) | IMU/INS + DVL (velocity) | yes (set-transformer) but only to regress EKF noise cov | no — multi-rate handled by EKF + fixed 1s/100-sample windows | no (EKF prediction loop, no modality dropout) | partial — held-out test trajectory segments (same vehicle/session) | branches/hybrid — EKF does fusion; transformer regresses Q (not the fusion block) |
| EffLoc (xiao2024effloc) | Single camera image (single-modality) | yes (Vision Transformer) | no — static single image, no time | no (single modality) | yes — Oxford RobotCar cross-day/cross-weather | n/a (single modality, no fusion) |
| SCM-PR (lin2025scmpr) | RGB image + 3D LiDAR map (cross-modal) | yes (cross-modal semantic attention + NetVLAD) | no | no (robust to illumination/season, not sensor dropout) | yes — KITTI-360 day/night + seasonal subsets | branches — separate RGB and LiDAR branches, attention at matching |
| PI-RNN (diazguerra2023pirnn) | Acoustic source detections (single-domain) | partial — RNN with internal multi-head attention | no | no | no (synthetic SST, no cross-session protocol) | unified over the set of SOURCES (but recurrent, not a set-transformer) |

Justify-quotes:
- AFT-VO MODS: "combines predictions from asynchronous multi-view cameras" [8b22a8f8]
- AFT-VO ATT: "a novel transformer-based fusion module, AFT-VO, is introduced" [8b22a8f8]
- AFT-VO CT: "we propose to discretise the continuous time domain into bins" [8b22a8f8]
- AFT-VO ROB: "employing multiple cameras is a clear way to provide robustness to individual camera failures" [8b22a8f8]
- AFT-VO XSESS: "we split the test set into three categories defined as: daylight, rain and night" [8b22a8f8]
- AFT-VO UNIFIED: "employs a Mixture Density Network (MDN) to estimate the probability distributions of the 6-DoF poses for every camera ... Then a novel transformer-based fusion module" [8b22a8f8]
- A-KIT MODS: "nonlinear sensor fusion based on an inertial navigation system and Doppler velocity log" [fb531c11]
- A-KIT ATT/UNIFIED: "Built upon a set-transformer network, A-KIT is designed for real-time adaptive regression of the process noise covariance matrix" [fb531c11]
- A-KIT CT: "The INS operates at a rate of 100 [Hz] and the DVL at 1 [Hz]" + "a one-second window was taken, meaning one hundred samples" [fb531c11]
- A-KIT XSESS: "an additional two 400 [sec] segments of the data that are not present in the training set, referring to them as the test set" [fb531c11]
- EffLoc MODS/ATT: "a novel efficient Vision Transformer for single-image camera relocalization" [1d12937b]
- EffLoc XSESS: "captured biweekly for more than a year, encompassing a diverse array of environmental conditions" [1d12937b]
- SCM-PR MODS/ATT: "takes a query RGB image I_RGB and a pre-built 3D LiDAR point cloud map P_LiDAR as inputs" + "cross-modal semantic attention mechanism" [5cd52555]
- SCM-PR XSESS: "sequences captured at different times of day and across distinct seasons" [5cd52555]
- PI-RNN ATT/UNIFIED: "a multi-head attention module ... invariant to the permutation of the elements of its input sets" [3b94d89f]

---

## C. ATOMIC CLAIMS

1. Standard transformer positional encoding assumes equidistant token positions, which breaks for asynchronous multi-rate sensors. Supports gap for (i). Quote: "We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE_{pos+k} can be represented as a linear function of PE_{pos}." [f7cf37ed] AND AFT-VO's restatement: "they assume the consecutive items to be equidistant in the time domain e.g. the distance between every consecutive input is equal. ... However, this is not applicable to our problem as the information comes from multiple asynchronous sources where the inputs are not equally distant from each other." [8b22a8f8]

2. The closest async+attention competitor (AFT-VO) handles asynchronous time by BINNING/QUANTISING continuous time, not by encoding real-valued Delta-t. Supports our (i) novelty. Quote: "To address this issue we propose to discretise the continuous time domain into bins." [8b22a8f8]

3. AFT-VO is single-modality (multi-view vision) and does late fusion of per-camera pose predictions, not a unified set-transformer over heterogeneous sensor tokens. Supports (ii) + WiFi/IMU gap. Quote: "Our approach first employs a Mixture Density Network (MDN) to estimate the probability distributions of the 6-DoF poses for every camera ... Then a novel transformer-based fusion module, AFT-VO, is introduced, which combines these asynchronous pose estimations." [8b22a8f8]

4. A-KIT uses a set-transformer, but only to regress the EKF process-noise covariance — the EKF, not the transformer, performs the state fusion, and multi-rate is handled by the EKF + fixed-length windows. Supports (i)+(ii) gap (no end-to-end unified set-transformer; no learned time embedding). Quote: "Built upon a set-transformer network, A-KIT is designed for real-time adaptive regression of the process noise covariance matrix." [fb531c11] + "a one-second window was taken, meaning one hundred samples." [fb531c11]

5. A-KIT fuses inertial + DVL (underwater), not WiFi+IMU indoor localization, and reports only relative improvement over EKF. Supports domain/contribution gap. Quote: "A-KIT outperforms the conventional EKF by more than 49.5% and model-based adaptive EKF by an average of 35.4% in terms of position accuracy." [fb531c11]

6. The Set Transformer is a permutation-invariant attention architecture and a universal approximator of permutation-invariant functions — the pillar for our (ii). Quote: "Proposition 1. The Set Transformer is permutation invariant." [a85a3ae2]

7. Deep Sets gives the theoretical guarantee that permutation-invariant set functions have a canonical decomposition — justifies treating sensor observations as an unordered set. Quote: "A function f(X) operating on a set X ... is a valid set function, i.e., invariant to the permutation of instances in X, iff it can be decomposed in the form rho(sum_{x in X} phi(x))." [2e61db11]

8. EffLoc shows transformers achieve SOTA single-image relocalization but it is single-modality with no temporal/async/fusion component. Supports the gap that transformer-localization exists but not for async WiFi+IMU fusion. Quote: "We propose EffLoc, a novel efficient Vision Transformer for single-image camera relocalization." [1d12937b]

9. SCM-PR uses cross-modal attention for localization (place recognition) but only cross-modal (RGB to LiDAR map), no cross-time/async handling, and the task is retrieval (Recall@1), not metric (x,y). Supports the gap that cross-modal attention exists but not unified cross-modal-AND-cross-time over async streams. Quote: "cross-modal semantic attention mechanism that enables the RGB image descriptor to focus on corresponding semantic regions within the LiDAR map." [5cd52555]

10. Permutation-invariant attention has been applied to a localization/tracking task (sound-source tracking) — but RNN-based and permutation-invariant over sources, not over (modality, time) sensor tokens. Supports (ii) lineage + remaining gap. Quote: "we present a permutation-invariant recurrent neural network (PI-RNN) that takes an unordered set of embeddings as input." [3b94d89f]

11. NONE of the competitors combines a learned real-valued Delta-t embedding AND a single unified permutation-invariant set-transformer AND missing/stale-modality dropout with cross-session evaluation — the conjunction is the novelty. (Synthesis claim grounded by the per-paper rubric quotes above; AFT-VO=binned time, A-KIT=EKF+windows, EffLoc/SCM-PR=no time, PI-RNN=RNN/no time/no dropout.) Quote anchor: "we propose to discretise the continuous time domain into bins." [8b22a8f8]

---

## D. GROUP GAP SYNTHESIS

This group is the architectural backbone of our method and its nearest neighbours: the Transformer [f7cf37ed], Deep Sets [2e61db11] and the Set Transformer [a85a3ae2] establish attention and permutation-invariant set processing, and Perceiver / Perceiver IO [44dce69c, a66f7a26] generalize attention to modality-agnostic sets and structured queries — collectively justifying treating asynchronous sensor observations as an unordered set fused by one attention block. The applied competitors show a clear trend toward attention/transformers for localization and odometry: EffLoc [1d12937b] (single-image ViT relocalization), SCM-PR [5cd52555] (cross-modal RGB-to-LiDAR attention), PI-RNN [3b94d89f] (permutation-invariant attention for source tracking), A-KIT [fb531c11] (a set-transformer that tunes an EKF for INS/DVL fusion), and most closely AFT-VO [8b22a8f8], the asynchronous fusion transformer for multi-view visual odometry. However, each leaves a precise gap against our three contributions: on (i) continuous-time, the only async-aware competitor (AFT-VO) QUANTISES time into bins rather than encoding real-valued Delta-t [8b22a8f8], A-KIT delegates multi-rate to an EKF with fixed 100-sample windows [fb531c11], and the rest are static (EffLoc/SCM-PR) or RNN-recurrent (PI-RNN); on (ii) a single unified permutation-invariant set-transformer doing cross-modal AND cross-time fusion at once, AFT-VO does late fusion of per-camera pose predictions, A-KIT's set-transformer only regresses noise covariance while the EKF fuses, and SCM-PR keeps separate per-modality branches; on (iii) explicit missing/stale-modality robustness (modality + instant dropout) demonstrated by real-world cross-session generalization, no competitor trains with modality dropout — their robustness is to camera failure via redundancy (AFT-VO), or to appearance change (EffLoc/SCM-PR), and cross-session evaluation when present is single-modality (EffLoc) or same-vehicle held-out segments (A-KIT). Crucially, NONE of them targets asynchronous WiFi RSSI + IMU indoor (x,y) localization, and none combines all three contributions — the conjunction of real-valued Delta-t encoding, one permutation-invariant set-transformer, and dropout-based async robustness with cross-session evaluation is the open space this paper occupies.
