# Related-Work Evidence — Group: Inertial / IMU

Notebook: `0cbfc89a-1f81-4e8c-8a6d-712b70e0c6b6`
Group focus: learned inertial navigation / pedestrian dead-reckoning. Contains the IMU benchmark
(RoNIN), the closest overall competitor (iMoT), and the transformer-based inertial methods.

Our paper (ICINCO 2026): "Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor
Localization". Three contributions: (i) learned continuous-time encoding of real-valued Delta-t;
(ii) one permutation-invariant set-transformer doing cross-modal AND cross-time fusion at once;
(iii) async-robustness via modality-dropout + instant-dropout, shown by real cross-session generalization.

> NOTE ON DATES: the brief labels iMoT as "Nguyen et al. 2024". The source itself reads
> "Copyright (c) 2025, Association for the Advancement of Artificial Intelligence" and cites
> arXiv 2024 -> AAAI 2025. Both forms are defensible; I record the grounded AAAI **2025** copyright.

---

## A. CAPSULES

### iMoT — Nguyen et al., AAAI 2025 (CLOSEST COMPETITOR)
- **source_id:** 7b6e4a06
- **citation:** Son Minh Nguyen, Linh Duy Tran, Duc Viet Le, Paul J.M. Havinga (2025). iMoT:
  Inertial Motion Transformer for Inertial Navigation. AAAI (Assoc. for the Advancement of AI).
  (Authors at University of Twente + Viettel AI.)
- **role:** competitor
- **method:** Transformer encoder-decoder for inertial odometry. Encoder uses self-attention over
  temporal variate tokens of acceleration and angular velocity; decoder uses cross-attention with a
  set of learnable "query motion particles" to model motion uncertainty; Progressive Series Decoupler,
  Adaptive Positional Encoding, Dynamic Scoring Mechanism.
- **modalities:** inertial-only: acceleration + angular velocity (2 IMU channels). No WiFi.
- **time_handling:** fixed 1-second windows; token dimension set to 100 (100 Hz) or 200 (200 Hz)
  per dataset. NO continuous-time Delta-t embedding; rate is baked into the token dimension.
  "Adaptive Positional Encoding ... dynamically modifies positional embeddings for temporal
  discrepancies between different modalities" — but this is positional encoding of token index,
  not a learned function of real-valued elapsed time.
- **robustness:** no explicit missing/stale-sensor mechanism; robustness is to motion uncertainty
  via learnable query motion particles.
- **datasets:** RIDI, RoNIN, OxIOD, IDOL (four benchmarks).
- **headline_result:** RoNIN dynamic, UNSEEN subjects: ATE 5.31 m (vs RIDI 15.75, CTIN 6.89,
  TLIO 6.77). IDOL: +15.43% T-RTE, +12.50% D-RTE over second-best (RoResnet18).
- **limitation:** inertial-only; fixed-rate windows; no absolute reference (drift bounded only by
  the learned prior); no missing-modality robustness; cross-SUBJECT generalization only (not cross-session WiFi).
- **diff_vs_ours:** iMoT is the closest in spirit (transformer + cross-modal attention between accel
  and gyro, even uses 128 query particles), but (a) modalities are the two IMU channels, NOT WiFi+IMU;
  (b) time is fixed-rate windows with index positional encoding, NOT learned continuous-time Delta-t;
  (c) attention is split encoder(self)/decoder(cross), NOT one unified permutation-invariant set block;
  (d) no modality/instant dropout, no stale-sensor degradation, no cross-session real-world eval.
- **key_quote:** "We propose iMoT, an innovative Transformer-based inertial odometry method that
  retrieves cross-modal information from motion and rotation modalities for accurate positional
  estimation." (7b6e4a06)
- **quote (modalities):** "Inertial odometry methods aim to reconstruct a traveled trajectory from
  corresponding IMU sequences ... Aa ... and Ag ... represent acceleration and angular velocity ...
  recorded over D = 3 channels along x-, y-, z-axes within 1 second, respectively." (7b6e4a06)
- **quote (rate baked into token dim):** "Depending on the sampling rate of each dataset, the token
  dimension is set to 100 for IMU sequences recorded at 100 Hz and to 200 for sequences recorded at
  200 Hz." (7b6e4a06)
- **quote (cross-subject generalization):** "Throughout the experiment, our proposed method
  consistently outperforms other SoTA approaches, particularly in its ability to generalize to unseen
  subjects." (7b6e4a06)
- **quote (headline):** "in dynamic scenarios from the RoNIN dataset ... our method achieves an ATE
  of 5.31 m for unseen subjects, significantly outperforming RIDI (15.75 m) by 66.29%, and other
  robust methods that account for motion uncertainties, such as CTIN (6.89 m) and TLIO (6.77 m)."
  (7b6e4a06)

### RoNIN — Yan, Herath & Furukawa, 2019 (IMU BENCHMARK; our ResNet1D baseline)
- **source_id:** 9b151e20
- **citation:** Hang Yan, Sachini Herath, Yasutaka Furukawa (2019). RoNIN: Robust Neural Inertial
  Navigation in the Wild: Benchmark, Evaluations, and New Methods. arXiv:1905.12853 (also cited
  as ICRA 2020 / Herath, Yan, Furukawa 2020 by downstream papers).
- **role:** benchmark
- **method:** Three regression backbones (ResNet-18 1D, stacked LSTM, TCN) that regress a 2D
  velocity vector from a window of IMU data; velocities are integrated to a trajectory.
- **modalities:** IMU-only (200x6 tensor = 6 IMU channels). Explicitly no WiFi/non-inertial.
- **time_handling:** fixed window: at frame i, IMU from i-200 to i as a 200x6 tensor; predict every
  five frames at test. Fixed-rate, no Delta-t embedding.
- **robustness:** none for missing modalities (single modality).
- **datasets:** RIDI dataset, OxIOD, and their new RoNIN dataset (>40 h, 100 subjects per iMoT).
- **headline_result:** RoNIN dataset, UNSEEN subjects: ResNet ATE 5.14 / RTE 4.37; LSTM 5.32 / 3.58;
  TCN 5.70 / 4.07. (Table 1.)  [These are the canonical numbers we cite for the baseline.]
- **limitation:** IMU-only; integrates velocity so drift accumulates without absolute reference;
  fixed-rate windows.
- **diff_vs_ours:** our ResNet1D baseline mirrors RoNIN-ResNet; RoNIN provides no fusion, no
  continuous-time handling, no missing-modality robustness, no WiFi anchor.
- **key_quote:** "Inertial navigation is the problem of estimating the position and orientation of a
  moving subject only from a sequence of IMU sensor data." (9b151e20)
- **quote (architectures):** "We present three RoNIN variants based on ResNet, LSTM or TCN.
  RoNIN ResNet: We take the 1D version of the standard ResNet-18 architecture and add one fully
  connected layer with 512 units at the end to regress a 2D vector. At frame i, the network takes IMU
  data from frame i-200 to i as a 200x6 tensor and produces a velocity vector at frame i." (9b151e20)
- **quote (headline, Table 1):** "RoNIN Dataset ... Unseen ATE 458.06 27.67 15.66 32.03 5.14 5.32
  5.70  RTE 117.06 23.17 18.91 26.93 4.37 3.58 4.07" (9b151e20)
  [columns: NDI, PDR, RIDI, IONet, RoNIN-ResNet, RoNIN-LSTM, RoNIN-TCN]

### CTIN — Rao et al., AAAI 2022 (contextual transformer for inertial)
- **source_id:** 88da0ef4
- **citation:** Bingbing Rao, Ehsan Kazemi, Yifan Ding, Devu M. Shila, Frank M. Tucker, Liqiang Wang
  (2022). CTIN: Robust Contextual Transformer Network for Inertial Navigation. AAAI.
- **role:** competitor
- **method:** ResNet-based encoder enhanced by local + global multi-head self-attention for spatial
  context; Transformer decoder fuses spatial reps with temporal knowledge via multi-head attention;
  multi-task learning with uncertainty reduction.
- **modalities:** inertial-only: 6D IMU (gyroscope + accelerometer). No WiFi.
- **time_handling:** sliding window size m=200, overlap step varies per dataset; IMU rotated to
  navigation frame. Fixed-rate windows, no Delta-t embedding.
- **robustness:** robustness to IMU noise/bias via random perturbation of samples; NO missing/stale
  modality mechanism (single modality).
- **datasets:** RIDI, OxIOD, RoNIN, IDOL, and own CTIN dataset.
- **headline_result:** CTIN dataset ATE 1.28 m, T-RTE 1.29 m, D-RTE 0.08 m (Table 3); avg ATE
  improvement over R-LSTM/R-ResNet/R-TCN of 34.74%/21.78%/37.46% (seen).
- **limitation:** inertial-only; fixed-rate; no absolute reference; no missing-modality robustness.
- **diff_vs_ours:** attention used but split across a ResNet spatial encoder + transformer temporal
  decoder (hybrid, not one unified set block); single modality; no continuous-time; no modality dropout.
- **key_quote:** "we first design a ResNet-based encoder enhanced by local and global multi-head
  self-attention to capture spatial contextual information from IMU measurements. Then we fuse these
  spatial representations with temporal knowledge by leveraging multi-head attention in the Transformer
  decoder." (88da0ef4)
- **quote (windowing):** "we leverage a sliding window with size m to prepare datasets at timestamp t,
  denoted by X(1:m)_t = [x_{t-m+1}, ..., x_t]." (88da0ef4)
- **quote (noise robustness, not modality):** "we also increase the robustness of the network against
  IMU measurements noise and bias by random perturbation of samples." (88da0ef4)

### RIOT — Brotchie et al., Sensors 2023 (recursive inertial odometry transformer)
- **source_id:** 13fe55b9
- **citation:** James Brotchie, Wenchao Li, Andrew D. Greentree, Allison Kealy (2023). RIOT: Recursive
  Inertial Odometry Transformer for Localisation from Low-Cost IMU Measurements. Sensors 23(6), 3217.
- **role:** competitor
- **method:** self-attention encoder-decoder; embeds 9D IMU, encoder self-attention -> context vector
  -> decoder estimates 3D position; recursive use of prior position estimates as input.
- **modalities:** inertial-only but 9D: 3D accel + 3D gyro + 3D magnetometer. No WiFi.
- **time_handling:** data synchronised at 100 Hz; sliding window of 100 measurements every 50;
  window sizes 50-500 (0.5-5 s). Fixed-rate, no Delta-t embedding.
- **robustness:** sliding window "averages out noise and fill in gaps" — mitigates noise/missing
  *measurements within a window*, NOT a missing-modality mechanism. Magnetometer dependence makes
  calibration region-specific (their stated limitation).
- **datasets:** OxIOD smartphone data (158 sequences, >42 km); unseen-sequence evaluation.
- **headline_result:** sequence-length-weighted mean ATE 0.0865 m, mean RTE 0.0091 m (RIOT vs ARIOT/GRU).
- **limitation:** magnetometer-dependent (region-specific); single (inertial) modality; fixed-rate;
  no absolute anchor; no modality dropout.
- **diff_vs_ours:** transformer + self-attention, but inertial-only, fixed-rate, no continuous-time,
  no WiFi, no cross-modal fusion, no modality-dropout robustness.
- **key_quote:** "To the best of our knowledge, our approaches are the only networks that leverage
  self-attention and all available IMU information (from a 3D accelerometer, 3D gyroscope and 3D
  magnetometer) to provide an end-to-end, 3D inertial odometry solution." (13fe55b9)
- **quote (architecture):** "The Recursive Inertial Odometry Transformer is a self-attention-based
  encoder-decoder network." (13fe55b9)
- **quote (fixed rate):** "The IMU data was collected and synchronised with a frequency of 100 Hz."
  (13fe55b9)
- **quote (headline):** "RIOT outperforms ARIOT and a GRU in terms of position error metrics, with a
  sequence length weighted mean Absolute Trajectory Error (ATE) of 0.0865 m and sequence length
  weighted mean Relative Trajectory Error (RTE) of 0.0091 m." (13fe55b9)

### NeurIT — Zheng et al., 2024 (neural inertial tracking, indoor robotic IoT)
- **source_id:** 01c12ae2
- **citation:** Xinzhe Zheng, Sijie Ji, Yipeng Pan, Kaiwen Zhang, Chenshu Wu (2024). NeurIT: Pushing
  the Limit of Neural Inertial Tracking for Indoor Robotic IoT. (Univ. of Hong Kong; venue/year not
  stated verbatim in the indexed text — UNSUPPORTED venue.)
- **role:** competitor
- **method:** TF-BRT = Time-Frequency Block-recurrent Transformer; combines block-recurrent attention
  (RNN-in-Transformer) + multi-head self-attention + convolution + time-frequency learning; outputs
  velocity/orientation sequences that are integrated.
- **modalities:** inertial-only but 9D: accelerometer + gyroscope + MAGNETOMETER (body-frame
  magnetometer differentiation). No WiFi as input (WiFi only discussed as prior work).
- **time_handling:** sliding window, fixed length L; combined 3-modality time series segmented into
  fixed-length sequences. Fixed-rate, no Delta-t embedding.
- **robustness:** data augmentation by random rotation for heading robustness; NO missing/stale
  modality dropout mechanism.
- **datasets:** own NeurIT robot dataset, 4 floors across 3 buildings; train/val/test-seen in
  Building A, test-unseen in Buildings B and C (cross-BUILDING generalization).
- **headline_result:** ~1 m tracking error over 300 m; +48.21% accuracy vs best baseline on unseen
  data; avg drift rate 0.62%.
- **limitation:** inertial-only (no absolute WiFi anchor); magnetometer dependence; fixed-rate;
  no missing-modality robustness.
- **diff_vs_ours:** strong on transformer + cross-environment generalization, but inertial-only,
  fixed-rate windows, no continuous-time Delta-t, no WiFi fusion, no modality dropout.
- **key_quote:** "NEURIT employs a Time-Frequency Block-recurrent Transformer (TF-BRT) at its core,
  combining both RNN and Transformer to learn representative features in both time and frequency
  domains." (01c12ae2)
- **quote (modalities/magnetometer):** "NEURIT incorporates magnetometer data as an additional
  feature with accelerometer and gyroscope measurements." (01c12ae2)
- **quote (cross-building):** "We collect the training, validation, and test-seen sets in Building A,
  and build the test-unseen set in Building B and C." (01c12ae2)
- **quote (headline):** "compared to the best baseline, NEURIT improves the tracking accuracy by
  48.21% on unseen data and achieves an average drift rate of 0.62%." (01c12ae2)
- **quote (WiFi only as prior work):** "Other approaches, such as combining IMU estimations with
  WiFi or acoustic signals, offer limited improvements and are affected by ambient interference."
  (01c12ae2)

### IMUNet — Zeinali et al., 2022 (efficient regression architecture)
- **source_id:** dcc6888f
- **citation:** Behnam Zeinali, Hadi Zandizari, J. Morris Chang (2022). IMUNet: Efficient Regression
  Architecture for IMU Navigation and Positioning. (Univ. of South Florida; manuscript received
  July 4, 2022 — exact venue UNSUPPORTED in indexed text.)
- **role:** competitor
- **method:** CNN with MobileResNet (MRBlock) blocks: depth-wise + point-wise convolution + batch
  norm + residuals; edge-efficient regression of position from raw IMU.
- **modalities:** inertial-only: raw IMU (6x200). No attention/transformer. No WiFi.
- **time_handling:** input 6x200 (window of 200 steps). Fixed-rate, no Delta-t.
- **robustness:** per-dimension convolution isolates noise across IMU channels; NO missing-modality.
- **datasets:** RoNIN, OxIOD, RIDI, PX4, + own proposed dataset.
- **headline_result:** own Proposed Dataset (seen): ATE 2.59 m, RTE 2.97 m (IMUNet column); RoNIN
  dataset (seen): ATE 3.52 m. (Table I.)
- **limitation:** no attention; single modality; fixed-rate; efficiency-focused, not fusion.
- **diff_vs_ours:** CNN, no attention at all, inertial-only, fixed-rate; orthogonal to all 3 contributions.
- **key_quote:** "This paper introduces a new architecture called IMUNet which is accurate and
  efficient for position estimation on edge device implementation receiving a sequence of raw IMU
  measurements." (dcc6888f)
- **quote (no attention / CNN):** "A new block which is the MobileResNet block has been proposed in
  this paper. inside the block, depth-wise and point-wise convolution along with a batch normalization."
  (dcc6888f)
- **quote (headline, Table I):** "Proposed Dataset Seen ATE 2.73 2.98 3.03 2.75 2.67 2.59  RTE 3.03
  3.42 3.55 3.19 3.48 2.97" (dcc6888f) [final column = IMUNet]

### NILoc — Herath et al., CVPR 2022 (neural inertial localization)
- **source_id:** 506570fb
- **citation:** Sachini Herath, David Caruso, Chen Liu, Yufan Chen, Yasutaka Furukawa (2022). Neural
  Inertial Localization. (Simon Fraser University + Reality Labs, Meta; CVPR-era — exact venue
  UNSUPPORTED in indexed text, presented as CVPR 2022 in literature.)
- **role:** competitor
- **method:** two-stage: RoNIN-ResNet turns IMU into velocity, then a two-branch Transformer
  (velocity branch with TCN compressor + auto-regressive location branch) maps velocity sequence to a
  location likelihood; scene-specific (trained per scene).
- **modalities:** IMU-only (explicitly avoids WiFi/camera). No WiFi as input.
- **time_handling:** 2-layer TCN with receptive field 10 compresses velocity sequence by factor 10;
  resample velocities by distance of travel to handle stationary periods. Distance-based resampling,
  not continuous-time Delta-t embedding.
- **robustness:** distance-based resampling for low-motion periods; NO missing-modality mechanism.
  NOTE: explicitly frames WiFi as a complementary anchor used "once in a few minutes" with IMU
  in-between — exactly the WiFi+IMU scenario our paper addresses with one fused model.
- **datasets:** own inertial localization dataset: 2 university buildings + 1 office; 53 h.
- **headline_result:** success-rate metrics (SR at distance thresholds), not ATE in meters; e.g.
  inertial localization SR@1m 69.9 etc.; "competitive accuracy at significantly lower run time"
  (20-30x faster than floorplan methods).
- **limitation:** scene-specific (trained per scene), absolute-localization not odometry; IMU-only;
  uses absolute map prior implicitly via scene training.
- **diff_vs_ours:** transformer-based and explicitly motivates WiFi-as-occasional-anchor + IMU, but
  (a) does NOT actually fuse WiFi as input; (b) per-scene transformer, not one unified
  permutation-invariant set block over modality/time tokens; (c) no continuous-time Delta-t; (d) no
  modality dropout.
- **key_quote:** "We only use an IMU sensor, which is energy efficient and privacy preserving compared
  to WiFi, cameras, and other data sources." (506570fb)
- **quote (transformer):** "NILoc employs a neural architecture with two Transformer-based network
  branches, capable of using long history of complex motion data to reduce uncertainty." (506570fb)
- **quote (WiFi-anchor motivation):** "The task represents a scenario where one uses WiFi to obtain a
  global position once in a few minutes, while re-localizing oneself in-between with an IMU sensor for
  energy efficiency." (506570fb)

### RNIN-VIO — Chen et al., ISMAR 2021 (neural inertial + visual-inertial)
- **source_id:** 1b3ac22c
- **citation:** Danpeng Chen, Nan Wang, Runsen Xu, Weijian Xie, Hujun Bao, Guofeng Zhang (2021).
  RNIN-VIO: Robust Neural Inertial Navigation Aided Visual-Inertial Odometry in Challenging Scenes.
  IEEE ISMAR (DOI 10.1109/ISMAR52148.2021.00043).
- **role:** competitor
- **method:** RNIN (ResNet + LSTM) neural inertial network producing relative position + covariance;
  RNIN-VIO tightly couples visual + IMU + NIN measurements in an EKF; outlier removal via Mahalanobis.
- **modalities:** vision (camera) + IMU (multi-sensor). Most relevant in this group as a fusion system,
  but the *fusion* is an EKF, not attention. No WiFi.
- **time_handling:** sliding window N IMU samples; LINEAR INTERPOLATION at 100 Hz to force fixed
  frequency (explicit). No continuous-time Delta-t embedding.
- **robustness:** GRACEFUL DEGRADATION to IMU-only when visual constraints drop — closest analog to
  our missing-modality robustness in this group, but achieved by EKF design (drop visual updates),
  not modality-dropout training.
- **datasets:** IDOL + own data; IDOL split into 'known'/'unknown' (cross user/building/device).
- **headline_result:** IDOL ATE (Ours, by building, known/unknown): 2.71/3.62, 6.19/5.23, 4.57/3.38 (Table 1);
  Vicon indoor ATE Normal00 0.124 m (RNIN-VIO col, Table 5).
- **limitation:** fusion is filter-based (EKF), not a single attention block; no continuous-time;
  no WiFi; robustness is hand-engineered (outlier removal + drop visual), not learned dropout.
- **diff_vs_ours:** does fuse two modalities AND degrades to IMU-only (graceful), but via EKF + outlier
  rejection rather than one permutation-invariant set-transformer with modality/instant dropout;
  fixed-rate (interpolated to 100 Hz); vision+IMU not WiFi+IMU; no learned Delta-t.
- **key_quote:** "we further develop a multi-sensor fusion system RNIN-VIO, which tightly couples the
  visual, IMU and NIN measurements." (1b3ac22c)
- **quote (graceful degradation):** "In our system, the visual constraints can be removed at any time,
  and state estimation can also be carried out only based on IMU measurements." (1b3ac22c)
- **quote (fixed-rate interpolation):** "On each sequence, we perform linear interpolation at 100 Hz
  to ensure that all input data are at a fixed frequency." (1b3ac22c)

### EqNIO — Jayanth et al. (subequivariant neural inertial odometry)
- **source_id:** c90706f3
- **citation:** Royina Karegoudra Jayanth, Yinshuang Xu, Ziyun Wang, Evangelos Chatzipantazis, Daniel
  Gehrig, Kostas Daniilidis. EqNIO: Subequivariant Neural Inertial Odometry. (Univ. of Pennsylvania;
  venue/year not stated verbatim in indexed text — UNSUPPORTED; ICLR-era preprint.)
- **role:** competitor
- **method:** subequivariant canonicalization wrapper around off-the-shelf nets (TLIO, RoNIN): predicts
  an O(2)-equivariant gravity-aligned frame with learnable yaw using O(2)-equivariant MLPs + convolution
  (NOT attention); maps IMU into canonical frame, outputs back to original frame.
- **modalities:** inertial-only: accelerometer + gyroscope (single IMU). No WiFi, no magnetometer.
- **time_handling:** 1 s window at 200 Hz -> n=200 samples; gravity-aligned. Fixed-rate, no Delta-t.
- **robustness:** gravity-perturbation robustness analysis; equivariance gives generalization — but
  NO missing-modality mechanism.
- **datasets:** trained on TLIO -> tested on TLIO + Aria; RoNIN variant trained on RoNIN -> tested on
  RoNIN, RIDI, OxIOD (cross-dataset generalization).
- **headline_result:** +O(2) Eq.Frame beats TLIO on no-EKF metrics by 57%/12%/11% (MSE*/ATE*/RTE*)
  on Aria; on OxIOD beats +J+TTT by 56%/43% (ATE*/RTE*).
- **limitation:** inertial-only; convolution/MLP-based (no attention); fixed-rate; no WiFi; no missing-modality.
- **diff_vs_ours:** strong cross-dataset generalization via equivariance, but not a transformer,
  inertial-only, fixed-rate, no continuous-time, no modality dropout. Orthogonal generalization route.
- **key_quote:** "This paper targets neural inertial odometry using data from a single IMU, comprised of
  an accelerometer (giving linear accelerations a_i) and gyroscope (giving angular velocity w_i)." (c90706f3)
- **quote (cross-dataset generalization):** "Our TLIO variant is trained on the TLIO Dataset and
  tested on TLIO and Aria Everyday Activities (Aria) Datasets ... Our RONIN variant is trained on RONIN
  Dataset ... We test our RONIN variant on three popular pedestrian datasets RONIN, RIDI and OxIOD."
  (c90706f3)

### IONet — Chen et al., AAAI 2018 (PILLAR; foundational learned inertial)
- **source_id:** 4eb0c9d6
- **citation:** Changhao Chen, Xiaoxuan Lu, Andrew Markham, Niki Trigoni (2018). IONet: Learning to
  Cure the Curse of Drift in Inertial Odometry. AAAI (Thirty-Second AAAI Conf., New Orleans,
  Feb 2-7, 2018).
- **role:** pillar
- **method:** first DNN to do inertial odometry from IMU only; segments IMU into independent windows,
  two-layer Bi-LSTM regresses polar displacement (delta_l, delta_psi); breaks the integration cycle.
- **modalities:** inertial-only (IMU). No WiFi.
- **time_handling:** segments inertial data into independent fixed windows; no continuous-time Delta-t.
- **robustness:** none for missing modality.
- **datasets:** OxIOD-style + multi-user/device + trolley tests.
- **headline_result:** "maximum error of our IOnet stayed around 2 meter within 90% testing time"
  (30-40% better than PDR).
- **limitation:** poor in large environments (per TLIO/RoNIN follow-ups); IMU-only; fixed windows.
- **diff_vs_ours:** the foundational learned-inertial pillar; LSTM not attention; single modality;
  no continuous-time; no fusion.
- **key_quote:** "To the best of our knowledge, our IONet is the first neural network framework to
  achieve inertial odometry using inertial data only." (4eb0c9d6)
- **quote (windowing):** "We propose to break the cycle of continuous integration, and instead segment
  inertial data into independent windows." (4eb0c9d6)

### TLIO — Liu et al., RA-L/IROS 2020 (PILLAR; tight learned inertial odometry)
- **source_id:** 50b563d3
- **citation:** Wenxin Liu, David Caruso, Eddy Ilg, Jing Dong, Anastasios I. Mourikis, Kostas
  Daniilidis, Vijay Kumar, Jakob Engel (2020). TLIO: Tight Learned Inertial Odometry. IEEE Robotics
  and Automation Letters (DOI 10.1109/LRA.2020.3007421), 2020 IEEE.
- **role:** pillar
- **method:** network (RoNIN-ResNet backbone) regresses 3D displacement + uncertainty; tightly fused
  into a stochastic-cloning EKF to solve pose, velocity, IMU biases.
- **modalities:** IMU-only (consumer-grade, head-mounted). No WiFi.
- **time_handling:** gravity-aligned IMU buffer segments; e.g. "200hz-05s-3s" = 3 s of 200 Hz input,
  regress displacement over last 0.5 s. Fixed-rate; no Delta-t.
- **robustness:** EKF fuses learned displacement + uncertainty; no missing-modality (single modality).
- **datasets:** own head-mounted pedestrian dataset.
- **headline_result:** "reduces average yaw and position drift by 27% and 33% respectively ... comparing
  to the best performing RoNIN velocity concatenation baseline."
- **limitation:** IMU-only; EKF can fail under highly dynamic motion (noted by RIOT); fixed-rate.
- **diff_vs_ours:** pillar of learned-inertial-in-a-filter; uses EKF not attention; single modality;
  no continuous-time; no learned modality dropout.
- **key_quote:** "This letter demonstrates a network that regresses 3D displacement estimates and its
  uncertainty, giving us the ability to tightly fuse the relative state measurement into a stochastic
  cloning EKF to solve for pose, velocity and sensor biases." (50b563d3)
- **quote (headline):** "This tight fusion approach reduces average yaw and position drift by 27% and
  33% respectively on our test dataset comparing to the best performing RoNIN velocity concatenation
  baseline approach." (50b563d3)

### RIDI — Yan, Shan & Furukawa, ECCV 2018 (BENCHMARK)
- **source_id:** e8c59e76
- **citation:** Hang Yan, Qi Shan, Yasutaka Furukawa (2018). RIDI: Robust IMU Double Integration.
  ECCV (Proc. European Conf. on Computer Vision, 621-636).
- **role:** benchmark
- **method:** regress a velocity vector from accel/gyro history, correct low-frequency bias in linear
  accelerations, then double-integrate to positions. First supervised training for inertial navigation.
- **modalities:** IMU-only.
- **time_handling:** velocity regression over history windows; fixed-rate (200 Hz).
- **datasets:** 10 subjects, 4 smartphone placements, >150 min at 200 Hz.
- **headline_result:** "mean positional errors below 3%."
- **diff_vs_ours:** classic IMU benchmark; double integration, no fusion, no continuous-time, no WiFi.
- **key_quote:** "Our datasets consist of various motion trajectories over 150 minutes at 200Hz ...
  RIDI produces motion trajectories comparable to the ground truth, with mean positional errors below 3%."
  (e8c59e76)
- **quote (WiFi as orthogonal prior work):** "WiFi signals are another information source for motion
  tracking without cameras in indoor environments. A particle filter is applied on IMU, WiFi, and the
  map data to enable reliable motion tracking ... Our research is orthogonal and directly benefits
  these techniques." (e8c59e76)

### OxIOD — Chen et al., 2018 (BENCHMARK / dataset)
- **source_id:** e20813a5
- **citation:** Changhao Chen, Peijun Zhao, Chris Xiaoxuan Lu, Wei Wang, Andrew Markham, Niki Trigoni
  (2018). OxIOD: The Dataset for Deep Inertial Odometry. arXiv:1809.07491.
- **role:** benchmark
- **method/contents:** large inertial dataset; 158 sequences, >42 km; 4 attachments, 4 motion modes,
  5 users, 4 phones; motion-capture ground truth at 0.5 mm.
- **modalities:** IMU.
- **headline_result:** dataset paper (size/diversity), no single ATE headline; mocap accuracy 0.5 mm.
- **key_quote:** "Our dataset contains 158 sequences totalling more than 42 km in total distance, much
  larger than previous inertial datasets." (e20813a5)

### IDOL — Sun, Melamed & Kitani, AAAI 2021 (BENCHMARK / method)
- **source_id:** 9a712c1a
- **citation:** Scott Sun, Dennis Melamed, Kris Kitani (2021). IDOL: Inertial Deep Orientation-
  Estimation and Localization. AAAI, vol. 35, 6128-6137 (Carnegie Mellon University).
- **role:** benchmark
- **method:** two-stage data-driven pipeline: RNN+EKF orientation module rotates raw IMU into reference
  frame, then a second RNN localizes. Addresses inaccurate phone orientation estimates.
- **modalities:** IMU incl. magnetometer (accel + gyro + mag at 100 Hz).
- **datasets:** own: 20 h, 3 buildings, 15 subjects.
- **headline_result:** "outperforms state-of-the-art methods in both orientation and position error";
  orientation ~0.08 rad (4.6deg) Bldg 1.
- **key_quote:** "Our proposed method outperforms state-of-the-art methods in both orientation and
  position error on a large dataset we constructed that contains 20 hours of pedestrian motion across
  3 buildings and 15 subjects." (9a712c1a)

### Cohen & Klein, 2024 (SURVEY / context)
- **source_id:** cd200533
- **citation:** Nadav Cohen, Itzik Klein (2024). Inertial Navigation Meets Deep Learning: A Survey of
  Current Trends and Future Directions. Results in Engineering 24, 103565. (Univ. of Haifa.)
- **role:** context
- **one-line:** Survey of DL methods used exclusively in inertial sensing and sensor-fusion algorithms,
  organized by land/air/sea platforms; covers pure inertial nav, aided inertial nav, and learning fusion
  filter parameters.
- **key_quote:** "this paper examines DL methods utilized exclusively in inertial sensing and sensor
  fusion algorithms, focusing entirely on vehicles regardless of their operating environment." (cd200533)
- **NOTABLE:** survey cites a Set-Transformer-based network ("ST-BeamsNet") for sensor-outage recovery
  in marine DVL — i.e. set-transformers-for-missing-modality exist outside indoor localization, which
  supports our framing that nobody has applied this set-transformer + continuous-time idea to WiFi+IMU.
  Quote: "In the case of a complete DVL outage ... the authors introduced 'ST-BeamsNet', which is a
  Set-Transformer based network that uses inertial reading and past DVL measurements to regress the
  current velocity." (cd200533)

---

## B. COMPETITOR RUBRIC

MODS = modalities fused | ATT = attention/transformer | CT = continuous-time / irregular Delta-t w/o
resample/ODE | ROB = explicit missing/stale modality robustness | XSESS = cross-session/subject/env
real-world eval | UNIFIED = single unified fusion block vs branches.

| bibkey | MODS | ATT | CT | ROB | XSESS | UNIFIED |
|---|---|---|---|---|---|---|
| nguyen2025imot | inertial-only (accel+gyro) | yes (enc self-attn + dec cross-attn) | no (fixed 1 s windows; token dim per rate; index APE) | no | yes (cross-subject: RoNIN/IDOL unseen) | branches (separate motion/rotation streams + enc/dec) |
| rao2022ctin | inertial-only (6D IMU) | yes (ResNet+local/global self-attn enc, transformer dec) | no (sliding window m=200) | no (noise perturb only) | yes (unseen subjects on RIDI/OxIOD/RoNIN/IDOL) | hybrid (ResNet spatial enc + transformer temporal dec) |
| brotchie2023riot | inertial-only (9D: accel+gyro+mag) | yes (self-attn enc-dec) | no (synced 100 Hz; window 50-500) | no (window fills within-window gaps only) | yes (unseen users/activities/devices) | unified-ish (one self-attn enc-dec, but single modality) |
| zheng2024neurit | inertial-only (9D: accel+gyro+mag) | yes (block-recurrent attn + multi-head self-attn) | no (fixed-length sliding window) | no (rotation augmentation only) | yes (cross-building: train A, test B/C) | hybrid (RNN+Transformer+conv stack) |
| zeinali2022imunet | inertial-only (6D IMU) | no (CNN MobileResNet) | no (6x200 window) | no | yes (unseen subjects; PX4 unseen) | n/a (single CNN, single modality) |
| herath2022niloc | IMU-only (velocity) | yes (two-branch transformer) | no (TCN compress; distance-based resample) | no | yes (multi-building dataset, per-scene) | branches (velocity branch + location branch, per-scene) |
| chen2021rninvio | vision + IMU (+NIN) | partial (ResNet+LSTM; fusion is EKF, not attn) | no (linear interpolation to fixed 100 Hz) | yes (drop visual -> IMU-only graceful) | yes (cross user/building/device on IDOL) | branches (EKF tightly-coupled filter, not one attn block) |
| jayanth2024eqnio | inertial-only (accel+gyro) | no (O(2)-equivariant MLP+conv wrapper) | no (1 s @200 Hz window) | no (gravity-perturb robustness only) | yes (cross-dataset: train RoNIN, test RIDI/OxIOD; TLIO->Aria) | n/a (wrapper around single-modality nets) |
| chen2018ionet (pillar) | inertial-only | no (Bi-LSTM) | no (independent fixed windows) | no | partial (new users/devices, trolley) | n/a (single modality) |
| liu2020tlio (pillar) | inertial-only | no (ResNet + EKF) | no (gravity-aligned fixed windows) | no | no (single dataset) | branches (network + EKF filter) |

### Rubric justify-quotes (per competitor)
- **nguyen2025imot CT=no:** "Depending on the sampling rate of each dataset, the token dimension is set
  to 100 for IMU sequences recorded at 100 Hz and to 200 for sequences recorded at 200 Hz." (7b6e4a06)
- **nguyen2025imot UNIFIED=branches:** "two cross-attention modules are employed singly for retrieving
  features regarding specific motion modes from both motion Aa and rotation tokens Ag." (7b6e4a06)
- **rao2022ctin ATT=yes / UNIFIED=hybrid:** "a ResNet-based encoder enhanced by local and global
  multi-head self-attention ... Then we fuse these spatial representations with temporal knowledge by
  leveraging multi-head attention in the Transformer decoder." (88da0ef4)
- **rao2022ctin ROB=no:** "we also increase the robustness of the network against IMU measurements
  noise and bias by random perturbation of samples." (88da0ef4)
- **brotchie2023riot MODS / ATT:** "the only networks that leverage self-attention and all available
  IMU information (from a 3D accelerometer, 3D gyroscope and 3D magnetometer)." (13fe55b9)
- **brotchie2023riot CT=no:** "The IMU data was collected and synchronised with a frequency of 100 Hz."
  (13fe55b9)
- **zheng2024neurit MODS:** "NEURIT incorporates magnetometer data as an additional feature with
  accelerometer and gyroscope measurements." (01c12ae2)
- **zheng2024neurit XSESS:** "We collect the training, validation, and test-seen sets in Building A,
  and build the test-unseen set in Building B and C." (01c12ae2)
- **zeinali2022imunet ATT=no:** "A new block which is the MobileResNet block ... depth-wise and
  point-wise convolution along with a batch normalization." (dcc6888f)
- **herath2022niloc MODS / ATT:** "We only use an IMU sensor ... NILoc employs a neural architecture
  with two Transformer-based network branches." (506570fb)
- **chen2021rninvio MODS / ROB=yes:** "we further develop a multi-sensor fusion system RNIN-VIO, which
  tightly couples the visual, IMU and NIN measurements." + "the visual constraints can be removed at
  any time, and state estimation can also be carried out only based on IMU measurements." (1b3ac22c)
- **chen2021rninvio CT=no:** "we perform linear interpolation at 100 Hz to ensure that all input data
  are at a fixed frequency." (1b3ac22c)
- **jayanth2024eqnio MODS / ATT=no:** "data from a single IMU, comprised of an accelerometer ... and
  gyroscope" + "specialized O(2) equivariant MLPs and convolution to process vector features."
  (c90706f3)

---

## C. ATOMIC CLAIMS (see structured output)

Key cross-cutting grounded facts:
1. No paper in this group fuses WiFi RSSI with IMU as input — supports contributions (ii)+(iii)
   uniqueness for WiFi+IMU. (multi-source)
2. Every method uses fixed-rate windows / interpolation / per-rate token dims; none uses a learned
   continuous-time Delta-t embedding without resampling/ODE — supports contribution (i). (multi-source)
3. The only graceful-degradation-to-fewer-sensors example (RNIN-VIO) achieves it via EKF + outlier
   rejection, not learned modality dropout — supports contribution (iii). (1b3ac22c)
4. iMoT is the closest transformer/cross-modal-attention competitor but is inertial-only (accel/gyro),
   fixed-rate, cross-subject only — pinpoints exactly what we add. (7b6e4a06)
5. The Cohen-Klein survey notes set-transformers used for sensor-outage recovery in a *different*
   domain (marine DVL), confirming our conjunction is novel for indoor WiFi+IMU. (cd200533)

---

## D. GROUP GAP SYNTHESIS

This group is the deep-learned inertial-navigation lineage: from foundational LSTM odometry (IONet)
and double-integration regression (RIDI), through ResNet velocity regression with the RoNIN benchmark
and EKF-tight fusion (TLIO), to the transformer wave (CTIN, RIOT, NeurIT, NILoc) and the closest
competitor iMoT, plus generalization-oriented EqNIO and the vision-fusion RNIN-VIO. The collective
trend is clear: attention/transformers have become the default backbone, and the field has shifted from
seen-subject accuracy toward cross-subject and cross-building generalization. However, three precise
gaps remain relative to our contributions. First, every method consumes IMU on a fixed-rate, resampled
grid (1 s windows at 100/200 Hz, linear interpolation, or rate-specific token dimensions) and uses at
most index-based positional encoding (iMoT's "Adaptive Positional Encoding") -- none learns a
continuous-time embedding of real-valued elapsed Delta-t to ingest asynchronous multi-rate streams
without resampling or an ODE solver (our contribution i). Second, fusion, where it exists, is split into
per-modality branches or a learned-network-plus-EKF filter (iMoT's separate motion/rotation streams,
CTIN's spatial-encoder+temporal-decoder, RNIN-VIO's EKF), not a single permutation-invariant
set-transformer doing cross-modal and cross-time fusion at once (our contribution ii). Third, robustness
to a missing or stale modality is essentially absent: only RNIN-VIO degrades gracefully to IMU-only, and
it does so through hand-engineered EKF outlier rejection rather than learned modality/instant dropout,
and none of these inertial works fuses WiFi RSSI nor reports cross-session WiFi generalization (our
contribution iii). In short, this group is inertial-only (or vision+IMU), fixed-rate, branch- or
filter-fused, and lacks both continuous-time tokenization and learned missing-modality robustness for
WiFi+IMU -- exactly the conjunction our paper targets.
