# Related Work Raw Evidence — Group: Multimodal fusion (localization)

Notebook: 2d35a60b-d383-411e-89ad-2286bbe68580 ("Multimodal fusion (localization)")
Gatherer pass for ICINCO 2026 paper: "Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization".
Our 3 contributions: (i) learned continuous-time Delta-t encoding (no resampling, no ODE);
(ii) single permutation-invariant set-transformer doing cross-modal AND cross-time fusion at once (unified, no per-modality branches);
(iii) async-robustness via modality-dropout (0.4) + instant-dropout (0.45), demonstrated by real-world CROSS-SESSION generalization.

Every quote below is verbatim from NotebookLM grounded answers; source_id is the 8-char prefix used in the prompt (full UUIDs in notebook). All 16 sources grounded; nothing UNSUPPORTED.

---

## CAPSULES

### Yu et al. 2022 — Multi-Modal Recurrent Fusion (COMPETITOR; near competitor) [53d5d1d1]
- Citation: Yu, J.; Wang, P.; Koike-Akino, T.; Orlik, P.V. (2022). Multi-Modal Recurrent Fusion for Indoor Localization. MERL (ICASSP-class signal-processing venue).
- Method: multi-stream LSTM (one RNN stream per modality), fused by learned "relative importance weights" (softmax over modality quality estimated from each stream's immediately-preceding hidden states); FC regression head.
- Modalities: **WiFi (RSSI + CSI) + IMU + UWB** (up to M=3). NOTE: not WiFi+IMU only — adds CSI and UWB.
- Time handling: fixed number of discrete time steps T (T=10 or T=30) with stepsize; NO interpolation/resampling mechanism described, NO real-valued Delta-t. "{xMM i }ti=t−T+1 → {xt, yt}, where T is the number of time steps".
- Robustness: YES — learned per-modality importance weights gauge real-time data quality (NLOS/sensor failures). This is a soft attention-like quality gate, a precedent for graceful degradation but NOT modality dropout.
- Datasets: SPAWC2021 multi-modal localization dataset; 80/10/10 random split (no cross-session).
- Headline result: best RF(F=1,seq10,biTrue) = 0.06 m mean / 0.05 m median / 0.12 m CDF@0.9 on SPAWC2021 (Table 1).
- Limitation vs ours: no continuous-time encoding (fixed step grid); per-modality LSTM branches (NOT a unified set), no permutation invariance; no missing-modality dropout training; no cross-session eval (random split).
- Key quote: "By formulating the localization as a multi-modal sequence regression problem, a multi-stream recurrent fusion method is proposed to combine the current hidden state of each modality in the context of recurrent neural networks while accounting for the modality uncertainty which is directly learned from its own immediate past states." [53d5d1d1]

### Zhou et al. 2024 — WIO-EKF (COMPETITOR) [de3074ce]
- Citation: Zhou, P.; Wang, H.; Gravina, R.; Sun, F. (2024). WIO-EKF: Extended Kalman Filtering-Based Wi-Fi and Inertial Odometry Fusion Method for Indoor Localization. IEEE (IoT-J / TIM class).
- Method: two deep models — CDAELoc (convolutional denoising autoencoder, WiFi) + DbDIO (dual-branch deep inertial odometry, IMU) — whose outputs are fused by an **EKF** (loosely coupled).
- Modalities: **WiFi fingerprint + IMU**.
- Time handling: NO async mechanism; matches rates by extracting fixed 1 s IMU windows (200 samples) to align with 1 Hz WiFi. Fixed-rate windows.
- Robustness: YES (partial) — CDAE trained with masking noise → robust to failed/replaced APs; WIO-EKF robust to initial-heading error.
- Datasets: UJIIndoorLoc, RoNIN, self-collected (SIAT).
- Headline result: WIO-EKF APE = 2.53 m (outperforms CDAELoc/DbDIO by ~34%/42%).
- Cross-session: YES (cross-day) — on UJIIndoorLoc, train/test collected 10 days apart.
- Limitation vs ours: EKF fusion, not attention/transformer; loosely coupled per-modality pipelines (branches), not a unified token set; no continuous-time Delta-t; robustness is AP-masking, not stale/missing-token dropout.
- Key quote: "this article adopts EKF as the fusion method ... using Wi-Fi fingerprint localization results and deep IO outputs as observation values" [de3074ce]; cross-day: "The time interval between data collection for the training and test sets is ten days." [de3074ce]

### Yang et al. 2025 — WiMU (COMPETITOR) [31231955]
- Citation: Yang, Q.; Xu, H.; Song, M.; Li, M. (2025). Demo: WiMU: Real-time Indoor Localization via Wi-Fi/IMU Fusion with Minimal Site Survey. MobiSys '25 (demo).
- Method: GNN/VGAE+MLP WiFi module (AP proximity graph) + PDR (IMU) for intermediate steps; fused by a **particle filter** with dynamically-adjusted parameters.
- Modalities: **WiFi RSSI + IMU**.
- Time handling: PDR compensates for missing intermediate steps between low-rate WiFi scans; no learned Delta-t; particle-filter time stepping.
- Robustness: partial — filter parameters adjusted dynamically using magnetic strength, posture, RSSI; no explicit missing-modality mechanism stated.
- Datasets: Microsoft Indoor Location and Navigation dataset; self-collected (university campus, warehouse).
- Headline result: avg localization error 4.6 m (campus), 4.61 m / 5.22 m (warehouse storage/bulk).
- Cross-session: NO reported.
- Limitation vs ours: filter-based, not attention/transformer; modular per-modality branches; no continuous-time encoding; no missing-modality dropout; no cross-session protocol.
- Key quote: "the results from the WiFi module and the PDR algorithm are fused together by a particle filter for more accurate location estimation" [31231955]

### Hua et al. 2023 — SmartFPS (COMPETITOR) [6b420277]
- Citation: Hua, ... (2023). SmartFPS: Neural network based wireless-inertial fusion positioning system.
- Method: end-to-end NN: inertial encoder (LSTM) + wireless encoder (CNN) + **attention layer** + fusion decoder (LSTM). GAN-based unsupervised transfer learning for device/pedestrian heterogeneity.
- Modalities: **Bluetooth (RSS) + inertial (IMU)**. NOTE: Bluetooth, NOT WiFi.
- Time handling: down-samples both streams to fixed 1 s windows due to uneven sampling frequencies (BT down-sampled to 100 Hz, inertial 200 steps). Fixed windows, no Delta-t.
- Robustness: YES (partial) — attention selects reliable wireless signals; degrading beacon count hurts SmartFPS much less than the wireless encoder (inertial carries it).
- Datasets: self-collected (Chuangzhi Building, Nanjing, one floor + lab).
- Headline result: 0.506 m mean positioning error (whole-floor BT-inertial); 0.575 m across different pedestrians and phones (after transfer learning).
- Cross-session: NO (cross-device / cross-pedestrian via GAN transfer, not cross-day).
- Limitation vs ours: attention is a sub-layer inside per-modality branches (hybrid, not a single unified set-transformer over all tokens); fixed-window resampling not continuous-time; Bluetooth not WiFi; robustness/transfer is GAN domain adaptation, not modality/instant dropout.
- Key quote: "SmartFPS is formed by four modules: (1) inertial encoder ... LSTM; (2) wireless encoder ... CNN; (3) attention layer; (4) fusion decoder ... LSTM." [6b420277]

### Herath et al. 2021 — Fusion-DHL (COMPETITOR) [583c1128]
- Citation: Herath, S.; Irandoust, S.; Chen, B.; Qian, Y.; Kim, P.; Furukawa, Y. (2021). Fusion-DHL: WiFi, IMU, and Floorplan Fusion for Dense History of Locations in Indoor Environments. (ICRA-class).
- Method: non-linear least-squares optimization (geo-localize RoNIN IMU trajectory against sparse WiFi/FLP constraints) + CNN floorplan-consistency refinement; iterated twice.
- Modalities: **WiFi (Google FLP geo-positions) + IMU + floorplan image**.
- Time handling: keeps IMU trajectory dense (50 Hz) and uses heavily-subsampled WiFi constraints (1/60 Hz) in the optimization; rate disparity handled by sparse-constraint optimization, not learned Delta-t. Penalty only applied beyond FLP-reported error radius.
- Robustness: partial — sparse/stale WiFi handled by treating it as a sparse constraint (battery-realistic 1/60 Hz); robust to WiFi noise via error-radius gating.
- Datasets: new benchmark, 4 university buildings (train) + 3 shopping malls (test), 15.2 h / 42.3 km.
- Headline result: RMSE roughly 5 m (vs ~12 m prior state-of-the-art), 50 Hz density.
- Cross-session: cross-ENVIRONMENT / cross-building (train univ → test malls), not cross-day/session.
- Limitation vs ours: optimization+CNN, not attention/transformer; requires a floorplan (extra modality/prior); per-stage pipeline not a unified token set; no learned continuous-time encoding; no missing-modality dropout training.
- Key quote: "The paper proposes a novel fusion of WiFi, IMU, and floorplan data by a combination of optimization and a convolutional neural network (CNN)." [583c1128]; "RMSE roughly 5m instead of 12m" [583c1128]

### Wei et al. 2021 — MM-Loc hybrid MDNN (COMPETITOR) [75d15e66]
- Citation: Wei, ... (2021). Sensor-Fusion for Smartphone Location Tracking Using Hybrid Multimodal Deep Neural Networks. Sensors (MDPI).
- Method: hybrid multimodal DNN — LSTM branch (inertial) + DNN branch (WiFi RSS), fused by concatenation (256-d) → FC joint layers → 2D regression. No attention/transformer.
- Modalities: **WiFi RSS + inertial (accel/gyro/magnetometer)**.
- Time handling: linear interpolation of inertial into fixed time windows; WiFi scan rate artificially adjusted to every 100 ms to align. Resampling + fixed windows; no learned Delta-t.
- Robustness: YES (notable precedent) — missing WiFi handled by feeding a NULL vector (all −100 dBm), which shifts inference entirely onto the inertial branch. Explicit missing-modality handling.
- Datasets: self-collected, two crowded office buildings, 65/25/10 split.
- Headline result: 1.9 m median error (1.98 m on test); 90% of errors < 4 m.
- Cross-session: NO (random split across two buildings).
- Limitation vs ours: concatenation fusion, no attention/transformer; per-modality branches (hybrid), not a unified permutation-invariant set; resampling not continuous-time; NULL-vector is a hand-set missing flag, not learned modality/instant dropout; no cross-session eval.
- Key quote: "when there is no WiFi scan in the system, the WiFi input is a vector with all components value of 0 (normalised to −100 dBm). This null vector causes the inference to balance entirely on the inertial sensors side." [75d15e66]

### Zhang et al. 2021 — LSTM WiFi+PDR fusion (COMPETITOR) [5fcd4079]
- Citation: Zhang, M.; Jia, J.; Chen, J.; Deng, Y.; Wang, X.; Aghvami, A.H. (2021). Indoor Localization Fusing WiFi With Smartphone Inertial Sensors Using LSTM Networks. IEEE.
- Method: single LSTM over displacement features ("first LSTM-based indoor fusion localization algorithm"); sliding-window displacement scheme. No attention/transformer.
- Modalities: **WiFi (RSS) + PDR/inertial**.
- Time handling: unify to 20 Hz; linearly interpolate WiFi, then moving-average filter to suppress the repeated (stale) up-sampled RSS; fixed sliding window W. Resampling + smoothing; no learned Delta-t.
- Robustness: partial — handles STALE (repeated) up-sampled WiFi via moving-average filter; no dedicated mechanism for fully MISSING modality.
- Datasets: self-collected (single + multi-floor paths).
- Headline result: average error 0.42 m "at best"; multi-floor A-A ~1.87 m.
- Cross-session: cross-USER (train user A, test users A and B), cross-posture; NOT cross-day/session.
- Limitation vs ours: single LSTM, not a permutation-invariant set-transformer; resampling to a fixed grid is exactly what our Delta-t avoids; stale handling is a smoothing filter not learned per-instant dropout; cross-user not cross-session.
- Key quote: "the unified sampling frequency is set to 20Hz" + "a moving average filter operation is performed to resist the effect of repeated data" [5fcd4079]

### Wang et al. 2024 — DamLoc multi-modal multi-scale fusion (COMPETITOR) [94456ab2]
- Citation: Wang, Q.; et al. (2024). Robust indoor localization based on multi-modal information [DamLoc]. Future Generation Computer Systems 155.
- Method: multi-branch CNN + **attention mechanism** for fusion; online data augmentation for multi-scale (variable-speed) sequences; context (previous prediction) as third modality with a state controller.
- Modalities: **Magnetic + BLE + context** (previous-prediction). NOTE: not WiFi (authors note BLE "can be replaced with Wi-Fi"). Not IMU.
- Time handling: piecewise-average interpolation to align magnetic (≤100 Hz) and BLE (5 Hz); fixed-length sliding sub-sequences. Interpolation + fixed windows; no learned Delta-t.
- Robustness: partial — context input zeroed to (0,0) via state controller to avoid "robot kidnapping" / tracking failure (analogous to dropping a stale modality), then relies on magnetic+BLE.
- Datasets: self-collected (author's lab building, 2nd floor); 3349 sequences, 163 volunteers, 3 phones, 3 attitudes, speeds 0-3 m/s.
- Headline result: ~63% average accuracy improvement over SOTA; mean error 0.30-1.38 m across paths (Table 4).
- Cross-session: NO (single environment); cross-user/device/speed/attitude only.
- Limitation vs ours: attention inside a multi-branch CNN (branches, not unified set); modalities are magnetic+BLE, not WiFi+IMU; interpolation not continuous-time; context-zeroing is hand-coded not learned dropout; no cross-session.
- Key quote: "we propose a multi-branch and attention mechanism-based end-to-end localization model to extract and efficiently fuse the significant features of the multi-modal data" [94456ab2]

### Lajoie et al. 2023 — PEOPLEx (COMPETITOR) [4d20d630]
- Citation: Lajoie, P.-Y.; Baghi, B.H.; Herath, S.; Hogan, F.; Liu, X.; Dudek, G. (2023). PEOPLEx: PEdestrian Opportunistic Positioning LEveraging IMU, UWB, BLE and WiFi. Samsung (IPIN/IROS-class).
- Method: **nonlinear factor-graph optimization** with IMU (RoNIN/PDR) backbone; opportunistic UWB/BLE/WiFi factors; novel adaptive-scaling and coarse-loop-closure factors. No attention/transformer/NN fusion.
- Modalities: **IMU + UWB + BLE + WiFi** (opportunistic; WiFi/BLE used as coarse loop closures).
- Time handling: factor graph incorporates measurements "as they become available" (asynchronous by construction, no fixed-rate resampling); coarse loop closing adapted to low scan rates and step discretization. Handles async WITHOUT resampling, but via classical optimization not learned Delta-t.
- Robustness: YES (strong, explicit opportunistic) — "In absence of sufficient radio-frequency signals, our technique performs inertial navigation alone"; "opportunistic in nature ... consistently delivering a solution ... even when it's not available."
- Datasets: self-collected (Samsung S22, Decawave DW1001 UWB anchors; BLE 1 Hz, WiFi 3 Hz; ARCore ground truth).
- Headline result: RMSE 1.05 ± 0.52 m (coarse loop closure, no UWB) vs RoNIN-alone 2.88 ± 1.55 m; multi-agent 1.06 ± 0.53 m.
- Cross-session: NO reported (10 trajectories in one environment).
- Limitation vs ours: classical factor-graph optimization, NOT a learned attention/transformer; async handled by graph factors, not a learned continuous-time token embedding; opportunistic fallback is architectural (graph), not learned modality/instant dropout; no cross-session protocol. NOTE: shares author Herath with our other competitors; strongest "async without resampling" precedent in the group but non-learned.
- Key quote: "our methodology employs IMU-based pedestrian inertial navigation as the backbone for sensor fusion, opportunistically integrating Ultra-Wideband (UWB), Bluetooth Low Energy (BLE), and WiFi signals when they are available" [4d20d630]

### Chen et al. 2015 — WiFi+PDR+landmarks Kalman (COMPETITOR; classical) [1c454340]
- Citation: Chen, ... (2015). Fusion of WiFi, Smartphone Sensors and Landmarks Using the Kalman Filter for Indoor Localization. Sensors (MDPI) 2015, 15.
- Method: **linear Kalman filter** fusing WiFi WPL output (observation) with PDR (state transition); landmarks reset accumulated drift.
- Modalities: **WiFi + PDR (accel/magnetometer/gyro/barometer) + landmarks**.
- Time handling: assumes aligned time step t; NO explicit async/interpolation mechanism described.
- Robustness: partial — when WiFi unavailable, PDR+landmarks give coarse position; landmarks reset drift.
- Datasets: self-collected (NTU research lab + testbed).
- Headline result: average ~1 m (0.9945 m lab, 0.8492 m testbed).
- Cross-session: NO reported.
- Limitation vs ours: classical KF, not learned/attention; linear, no async modeling; requires landmark semantics; no missing-modality training; no cross-session. Baseline-of-record for "classical filter fusion."
- Key quote: "we formulate the problem in a linear perspective. Then, a Kalman filter can be leveraged to solve this fusion problem effectively." [1c454340]

### Geneva et al. 2018 — Asynchronous Multi-Sensor Fusion (SPECIAL: async precedent) [59d94997]
- Citation: Geneva, P.; Eckenhoff, K.; Huang, G. (2018). Asynchronous Multi-Sensor Fusion for 3D Mapping and Localization. (ICRA/IROS-class).
- Method: factor-graph optimization (iSAM2 incremental or full batch). Core contribution = out-of-sequence (asynchronous) measurement alignment WITHOUT adding new graph nodes.
- Modalities: LIDAR (LOAM) + stereo vision (ORB-SLAM2) + RTK GPS. (Autonomous driving, NOT indoor WiFi/IMU.)
- Time handling — CRITICAL FOR US: it does NOT fuse at the true asynchronous timestamps; instead it ALIGNS asynchronous measurements to existing backbone state timestamps via analytically-derived linear 3D pose INTERPOLATION (unary factors) and EXTRAPOLATION / "stretching" (binary factors) under constant-velocity assumption. So async is handled by interpolation-to-fixed-states, explicitly to avoid adding nodes. Explicitly contrasts with spline-based continuous-time trajectories (Patron-Perez) which it says add complexity.
- Robustness: not framed as missing-modality robustness; about delay/rate alignment.
- Headline result: GPS-denied localization RMSE 0.71 m (vs naive 0.93 m); odometry-only 7.026 m (vs naive 26.74 m).
- Limitation vs ours: async = interpolation/extrapolation to a fixed state backbone, NOT a learned continuous-time Delta-t token embedding; classical optimization not a transformer; driving domain not WiFi+IMU indoor; no modality dropout. THIS IS THE KEY CONTRAST: the canonical "async fusion" precedent still resamples/aligns to fixed states; our learned Delta-t needs no alignment.
- Key quote: "To limit the addition of new graph nodes when receiving asynchronous data, we interpolate between two sequential 3D pose measurements to a given state timestamp." [59d94997]; "we accurately align both asynchronous unary and binary graph factors to existing states based on our analytically derived linear 3D pose interpolation." [59d94997]

### Neverova et al. 2014 — ModDrop (SPECIAL: modality-dropout ancestor) [d8cc9acd]
- Citation: Neverova, N.; Wolf, C.; Taylor, G.; Nebout, F. (2014). ModDrop: Adaptive Multi-Modal Gesture Recognition. (IEEE TPAMI).
- Method: multi-scale/multi-modal CNN; ModDrop = during fusion training, randomly drop ENTIRE modality channels (set whole modality input to 0 with Bernoulli probability) to (i) learn cross-modality correlations without false co-adaptation and (ii) handle missing channels at test time.
- Modalities (task): gesture recognition — intensity/RGB video + depth video + articulated pose + audio.
- Robustness: YES — this is THE precedent. "ensures robustness of the classifier to missing signals in one or several channels to produce meaningful predictions from any number of available modalities."
- Applied at input/fusion level (drops whole modalities, not random neurons); rate cited in paper: input ModDrop 10% per segment / input dropout 20%.
- Diff/relation vs ours: ModDrop drops WHOLE MODALITIES at the input of a fusion net for gesture recognition. Our contribution (iii) generalizes this lineage to (a) localization with WiFi+IMU, (b) ADDS per-instant (token) dropout — dropping individual time-stamped observations, not just whole modalities — and (c) ties it to async/stale robustness with cross-session eval. We cite ModDrop as the explicit ancestor of our modality-dropout.
- Key quote: "gradual fusion involving random dropping of separate channels (dubbed ModDrop) for learning cross-modality correlations while preserving uniqueness of each modality-specific representation." [d8cc9acd]; "the proposed ModDrop training technique ensures robustness of the classifier to missing signals in one or several channels to produce meaningful predictions from any number of available modalities." [d8cc9acd]

### Silva et al. 2023 — Industrial multi-sensor dataset (BENCHMARK / dataset) [b0e54375]
- Citation: Silva, I.; Pendao, C.; Torres-Sospedra, J.; Moreira, A. (2023). Industrial Environment Multi-Sensor Dataset for Vehicle Indoor Tracking with Wi-Fi, Inertial and Odometry Data. Data (MDPI) 2023, 8, 157. DOI 10.3390/data8100157.
- Modalities: WiFi (4 interfaces + radio map, ~0.62 Hz / 1.614 s mean) + 2 IMUs (20 Hz) + absolute wheel encoder/odometry (50 Hz) + CV ground truth.
- Platform/env: manually-pushed trolley emulating an industrial vehicle, factory-like open space (PIEP, Univ. of Minho, 20x50 m).
- Structure: 6 trajectories T1-T6 + offline RadioMap (40 samples/RP, 4 directions). Supports radio-map vs online-trajectory split (cross-phase).
- Headline baselines (what we cite): dead reckoning 8.25 m (IMU1) / 11.66 m (IMU2); WiFi fingerprinting (kNN, 5 interfaces) 2.19 m.
- Relevance to us: a real WiFi+IMU+odometry multi-rate indoor dataset with explicit asynchronous sampling rates — exactly the regime our Delta-t targets; vehicle/AMR analogue to our TIAGO++ setting.
- Key quote: "The average positioning error for simple dead reckoning ... is 8.25 m and 11.66 m for IMU1 and IMU2, respectively. The average positioning error for simple Wi-Fi fingerprinting is 2.19 m when combining the RSSI information from five Wi-Fi interfaces." [b0e54375]

### Abdalla et al. 2025/2026 — Multi-modal hybrid positioning dataset (BENCHMARK / dataset) [1bb05d0a]
- Citation: Abdalla, B.A.; Maghdid, H.S.; Sabir, A.T. (2025). A multi-modal dataset for hybrid indoor positioning using Wi-Fi RSS, embedded inertial sensors, and CCTV images. Data in Brief (Elsevier). DOI 10.1016/j.dib.2025.112370. NOTE: text says 2025 (prompt said 2026) — record as 2025 per "© 2025 The Author(s)."
- Modalities: WiFi RSSI (10 Hz / 100 ms, 12 APs) + embedded inertial (accel/gyro/magnetometer, ~5 Hz / 200 ms) + 8 CCTV cameras (20 fps / 50 ms).
- Platform/env: indoor university corridor (39x49 m), Koya University, Huawei Nova 5T smartphone; 1 m grid, 10 repeats/point (8 train / 2 test).
- Multi-rate handling (relevant to async): resamples IMU to 5 Hz, index-based nearest-neighbor alignment of CCTV/WiFi/IMU; bounded temporal offsets (±50/±100/±150 ms). i.e. the dataset itself relies on resampling+nearest-neighbor alignment — the very thing our Delta-t removes.
- Headline baseline: NONE quantitative — only "preliminary trials" with W3KNN + Kalman/Particle filters; "preliminary results show ... can significantly decrease the localization error" (no number). Record headline as "not grounded."
- Relevance to us: real WiFi RSS + inertial multi-rate dataset; cross-session NOT supported (static grid, per-location split).
- Key quote: "All IMU streams are resampled to a consistent 5 Hz grid ... Using index-based nearest-neighbour alignment for each coordinate, the maximum temporal offset is bounded by half the sampling period of each stream" [1bb05d0a]

### Wang & Ahmad 2025 — AI for AMR indoor localization (SURVEY / context) [51755f40]
- Citation: Wang, S.; Ahmad, N.S. (2025). AI-based approaches for improving autonomous mobile robot localization in indoor environments: A comprehensive review. Engineering Science and Technology, an International Journal 63 (2025) 101977, Elsevier.
- Scope: AI for AMR/UAV indoor localization; categorizes SLAM, odometry, multi-sensor fusion; DL is the dominant model class; multi-sensor fusion the dominant sensor strategy.
- Gap relevant to us: does NOT discuss asynchronous fusion (the word "asynchronous" only appears re: A3C RL); mentions transformers only as "high complexity" examples, not as a localization gap; frames missing-sensor robustness as a benefit of multimodal fusion, not an open gap. -> supports our framing that learned continuous-time async fusion + transformer is under-explored for indoor localization.
- Key quote: "AI methods facilitate the optimization and enhancement of multimodal data fusion from on-board sensors ... enabling robots to rely on visual or IMU data for supplementary localization when laser sensors fail" [51755f40]

### Lukasik et al. 2024 — Multimodal image-based indoor localization (SURVEY / context) [c3c7d669]
- Citation: Lukasik, S.; Szott, S.; Leszczuk, M. (2024). Multimodal Image-Based Indoor Localization with Machine Learning — A Systematic Review. Sensors (MDPI) 2024, 24, 6051.
- Scope: ~40 papers on multimodal indoor positioning that include camera imagery fused with motion sensors / radio (WiFi, UWB, RFID) / LiDAR. CNN/YOLO dominant; notes several attention/cross-modal-attention studies (Wen 2023, Liu 2021, Yan 2020).
- Conclusion relevant to us: multimodal consistently beats unimodal ("future indoor positioning systems will combine at least two input modalities"); attention appears but no single unified permutation-invariant set-transformer for WiFi+IMU; no async/continuous-time framing. Context for "attention is emerging but not yet a unified async set-transformer."
- Key quote: "research clearly shows that the multimodal approach outperforms a unimodal one. ... Machine learning methods for indoor localization using sensor fusion often leverage CNNs and deep CNNs to integrate data from multiple sensors." [c3c7d669]

---

## COMPETITOR RUBRIC (gap matrix)

| bibkey | MODS | ATT | CT (async/real-valued Delta-t, no resample/ODE) | ROB (missing/stale) | XSESS (cross-session real) | UNIFIED |
|---|---|---|---|---|---|---|
| yu2022multimodal | WiFi RSSI+CSI + IMU + UWB | no | no — fixed T discrete steps, no Delta-t | yes — learned per-modality importance weights | no — 80/10/10 random split | branches (per-modality LSTM streams) |
| zhou2024wioekf | WiFi fingerprint + IMU | no | no — fixed 1s IMU windows to match 1Hz WiFi | yes — CDAE masking-noise robust to failed APs | yes — cross-day (UJI, 10 days apart) | branches (CDAELoc + DbDIO via EKF) |
| yang2025wimu | WiFi RSSI + IMU | no | no — PDR fills gaps, particle-filter stepping | partial — dynamic filter params | no | branches (GNN WiFi + PDR via PF) |
| hua2023smartfps | Bluetooth RSS + IMU | partial — attention sub-layer | no — fixed 1s down-sampled windows | yes (partial) — attention selects reliable beacons; inertial fallback | no — cross-device/pedestrian (GAN), not cross-day | hybrid (LSTM/CNN branches + attention + LSTM decoder) |
| herath2021fusiondhl | WiFi (FLP) + IMU + floorplan | no | partial — sparse 1/60Hz WiFi constraints in optimization | partial — sparse/stale WiFi as sparse constraint | no — cross-building/environment | branches (optimization + CNN stages) |
| wei2021sensorfusion | WiFi RSS + inertial | no | no — interpolation + fixed windows, WiFi resampled to 100ms | yes — NULL (−100 dBm) vector for missing WiFi | no — random split, two buildings | branches (LSTM + DNN concat) |
| zhang2021lstm | WiFi RSS + PDR/inertial | no | no — unify to 20Hz, interpolate + moving-average, fixed window | partial — moving-average for stale repeated WiFi | no — cross-user / cross-posture only | hybrid (single LSTM over displacement features) |
| wang2024damloc | Magnetic + BLE + context (not WiFi/IMU) | yes — attention | no — piecewise interpolation + fixed sliding windows | partial — context zeroed (0,0) via state controller | no — single environment | branches (multi-branch CNN + attention) |
| lajoie2023peoplex | IMU + UWB + BLE + WiFi | no | partial — factor graph ingests async as available, coarse loop closure; NO learned Delta-t | yes — opportunistic, IMU-only fallback | no — 10 traj one environment | unified-ish (single factor graph) but classical, not learned |
| chen2015kalman | WiFi + PDR + landmarks | no | no — assumes aligned time step | partial — PDR+landmarks fallback when no WiFi | no | branches (KF: PDR state + WiFi observation) |

SPECIAL (characterized, not full competitors but rubric-relevant):
| geneva2018async | LIDAR + stereo vision + RTK GPS (driving) | no | partial — async ALIGNED by linear interpolation/extrapolation to FIXED states (no Delta-t, no ODE, but resamples-to-states) | n/a (delay alignment, not missing-modality) | n/a | unified factor graph (classical) |
| neverova2014moddrop | RGB+depth+pose+audio (gesture) | no (CNN late fusion) | n/a | YES — modality-dropout = the ancestor of our (iii) | n/a | shared fusion layers with modality-dropout |

Reading of the matrix: NO paper in the group has the conjunction (ATT=yes via a single unified set-transformer) AND (CT=yes via learned real-valued Delta-t, no resampling/ODE) AND (ROB=learned modality+instant dropout) AND (XSESS=yes). The two papers that touch true async (Geneva, PEOPLEx) are classical optimization, not learned/attention, and still align/interpolate rather than learn Delta-t. The only learned modality-dropout precedent (ModDrop) is gesture recognition, whole-modality only, no time/instant dropout, no localization. WIO-EKF is the only one with cross-day, but it is EKF + per-modality branches.

---

## ATOMIC CLAIMS

1. (supports gap for contribution i — continuous-time Delta-t) Deep WiFi+IMU fusion methods overwhelmingly RESAMPLE both streams onto a fixed-rate grid / fixed windows rather than modeling real-valued time gaps. [wei2021sensorfusion 75d15e66] quote: "we adjust the WiFi scan rate from the original sampling rate to every 100 ms"; [zhang2021lstm 5fcd4079] "the unified sampling frequency is set to 20Hz"; [zhou2024wioekf de3074ce] "time window is 200 (i.e., 1 s)".

2. (supports gap for i — even the canonical ASYNC precedent still aligns to fixed states) Geneva 2018, the named asynchronous-fusion precedent, handles async by interpolating/extrapolating measurements onto existing fixed state timestamps, explicitly to avoid adding nodes — it does not learn a continuous-time embedding. [geneva2018async 59d94997] quote: "we interpolate between two sequential 3D pose measurements to a given state timestamp."

3. (supports gap for i — datasets themselves resample) Even recent WiFi+inertial datasets pre-align modalities by resampling + nearest-neighbor index matching, the operation our Delta-t removes. [abdalla2026dataset 1bb05d0a] quote: "All IMU streams are resampled to a consistent 5 Hz grid ... Using index-based nearest-neighbour alignment".

4. (supports gap for ii — unified permutation-invariant set vs branches) The dominant fusion topology in this group is per-modality branches/streams later combined (concat, EKF, particle filter, weighted hidden states), NOT a single unified attention block over a set of (modality,time) tokens. [wei2021sensorfusion 75d15e66] "two parallel single-modality feature extractors and a joint network ... fused by concatenation"; [yu2022multimodal 53d5d1d1] "multi-stream recurrent fusion ... combine the current hidden state of each modality".

5. (supports gap for ii — attention exists but as a sub-layer in branches, not the whole fusion) Where attention is used, it is a sub-component inside modality-specific branches, not a single permutation-invariant set-transformer doing cross-modal AND cross-time fusion at once. [hua2023smartfps 6b420277] "(1) inertial encoder ... LSTM; (2) wireless encoder ... CNN; (3) attention layer; (4) fusion decoder ... LSTM"; [wang2024damloc 94456ab2] "multi-branch and attention mechanism-based ... model".

6. (supports contribution iii — direct ancestor) ModDrop is the explicit precedent for our modality-dropout: random Bernoulli dropping of WHOLE modality channels during training to make the model robust to missing signals at test. [neverova2014moddrop d8cc9acd] "random dropping of separate channels (dubbed ModDrop) ... ensures robustness of the classifier to missing signals in one or several channels".

7. (supports contribution iii — what the lineage LACKS) The localization-side robustness mechanisms in this group are hand-set or filter-based (NULL vector, moving-average, context-zeroing, opportunistic fallback), NOT a learned dropout over individual time-stamped tokens (our instant-dropout). [wei2021sensorfusion 75d15e66] "the WiFi input is a vector with all components value of 0 (normalised to −100 dBm)"; [wang2024damloc 94456ab2] "the context input ... is determined as (0, 0)".

8. (supports gap for iii — cross-session is rarely the eval) Most WiFi+IMU fusion papers evaluate with random splits or cross-user/cross-device, NOT cross-session/cross-day generalization. [yu2022multimodal 53d5d1d1] "80/10/10 splitting of Dataset1"; [wei2021sensorfusion 75d15e66] "65%, 25% and 10% for training, validation and testing". The one cross-day result is EKF-based: [zhou2024wioekf de3074ce] "time interval between data collection for the training and test sets is ten days."

9. (supports contribution iii benchmarking — real multi-rate datasets exist) Real WiFi+IMU(+odometry) indoor datasets exhibit exactly the multi-rate asynchrony we target (WiFi ~0.6-1 Hz vs IMU 20+ Hz). [silva2023dataset b0e54375] WiFi 1.614 s mean vs IMU 20 Hz vs encoder 50 Hz; baseline WiFi-FP 2.19 m, DR 8.25 m.

10. (supports the overall framing) Surveys confirm multimodal fusion + deep learning is the dominant trend, that multimodal beats unimodal, yet do not surface learned continuous-time asynchronous fusion or a unified set-transformer as solved. [lukasik2024survey c3c7d669] "the multimodal approach outperforms a unimodal one"; [wangahmad2025survey 51755f40] multi-sensor fusion is the dominant sensor strategy, "asynchronous" not discussed for fusion.

11. (sharpens closeness to ours among competitors) PEOPLEx is the closest on async-without-resampling AND opportunistic robustness, but it is a classical factor-graph optimizer (no learned attention, no learned Delta-t, no cross-session). [lajoie2023peoplex 4d20d630] "opportunistic in nature ... consistently delivering a solution ... even when it's not available."

12. (modality-mismatch note for fair comparison) Several "near competitors" do NOT actually fuse WiFi+IMU: Hua=Bluetooth+IMU, Wang2024=Magnetic+BLE+context, Yu adds CSI+UWB. Record precisely so Related Work does not over-claim them as WiFi+IMU baselines. [hua2023smartfps 6b420277] "Bluetooth-inertial positioning"; [wang2024damloc 94456ab2] "fuse magnetic with ... Bluetooth low energy (BLE), and context information".

---

## GROUP GAP SYNTHESIS

This group is the most on-problem cluster: deep and classical methods that fuse WiFi/BLE/magnetic RSS with inertial (and sometimes UWB, floorplan, vision, landmarks) to estimate indoor (x,y), spanning Kalman/particle filters (Chen 2015, WiMU 2025), EKF + deep encoders (WIO-EKF 2024), factor-graph optimization (PEOPLEx 2023, Geneva 2018), and end-to-end neural fusion with LSTM/CNN and occasional attention sub-layers (Yu 2022, SmartFPS 2023, Wei 2021, Zhang 2021, DamLoc 2024). The collective trend is clear and corroborated by both surveys: multimodal fusion plus deep learning dominates and reliably beats single-modality localization. However, three precise gaps remain relative to our contributions. First (vs i), every method either resamples/interpolates both streams onto a fixed-rate grid or fixed windows, or — in the only true asynchronous precedents (Geneva 2018, PEOPLEx 2023) — aligns measurements to fixed graph states via classical interpolation/optimization; none learns a continuous-time embedding of the real-valued elapsed time per observation, and even the WiFi+IMU datasets ship pre-resampled. Second (vs ii), the dominant topology is per-modality branches/streams fused late (concatenation, EKF, particle filter, weighted hidden states), and where attention appears it is a sub-layer inside those branches — no paper uses a single permutation-invariant set-transformer that performs cross-modal and cross-time fusion jointly over a set of (modality, time) tokens. Third (vs iii), the robustness mechanisms here are hand-set or filter-based (NULL-vector for missing WiFi, moving-average for stale RSS, context-zeroing, opportunistic IMU fallback) and the only learned modality-dropout precedent, ModDrop, is whole-modality dropout for gesture recognition with no per-instant/token dropout and no localization or cross-session evaluation; meanwhile most localization papers evaluate with random or cross-user/device splits, with only WIO-EKF reporting a cross-day result (and via an EKF, not a learned async transformer). Thus the conjunction we claim — learned continuous-time Delta-t + a single unified permutation-invariant set-transformer + modality-and-instant dropout validated by real cross-session generalization on WiFi+IMU — is not occupied by any single member of this group.
