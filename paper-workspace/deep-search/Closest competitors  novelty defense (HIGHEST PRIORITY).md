# Prior-Art and Novelty Positioning for a Continuous-Time Set-Transformer Fusing Asynchronous WiFi RSSI + IMU for Indoor (x, y) Localization

## TL;DR
- **No existing published work combines all three of your contributions** — continuous-time Δt encoding, a single unified self-attention block doing both cross-modal AND cross-time fusion, and modality/instant-dropout async-robustness — for WiFi+IMU indoor localization; the nearest competitors each hold only one or two of these properties, so your conjunction is defensibly novel.
- The closest architectural neighbors are **A-KIT (2024)** (set-transformer for sensor fusion, but EKF-coupled, INS/DVL not WiFi), **AFT-VO (2022)** (transformer fusing asynchronous continuous-timestamp sources, but visual odometry with per-source branches), **iMoT (2025)** (cross-modal transformer with adaptive positional encoding for temporal discrepancies, but IMU-only), and the **WiFi+IMU fusion family** (WiMU 2025, WIO-EKF 2024, Yu et al. 2022, Zhang et al. 2021) which use per-modality branches and filtering/RNN fusion rather than a unified continuous-time set-transformer.
- **All six of your held references are real and verified**; two need metadata corrections (Yu et al. is "Multi-Modal Recurrent Fusion for Indoor Localization," ICASSP 2022, not "recurrent fusion-based indoor localization"; AFT-VO is confirmed multi-camera visual odometry at IROS 2022, not WiFi).

## Key Findings

### Verification of your six held references
1. **Yu et al. 2022 — VERIFIED with title correction.** Exact title: "Multi-Modal Recurrent Fusion for Indoor Localization." Authors: Jianyuan Yu, Pu (Perry) Wang, Toshiaki Koike-Akino, Philip V. Orlik (Mitsubishi Electric Research Labs). Published at ICASSP 2022 (IEEE Int. Conf. on Acoustics, Speech and Signal Processing); arXiv:2203.00510; IEEE Xplore doc 9746071. Fuses WiFi RSSI, IMU, and UWB via a multi-stream RNN with learned per-modality uncertainty. It is explicitly **multi-stream (per-modality recurrent branches)**, not a single unified block. Your held title "A Multi-Modal Recurrent Fusion-based Indoor Localization" is incorrect — correct it.
2. **Antsfeld et al. 2020 — VERIFIED with venue nuance.** "Deep Smartphone Sensors-WiFi Fusion for Indoor Positioning and Tracking," Leonid Antsfeld, Boris Chidlovskii, Emilio Sansano-Sansano (NAVER Labs Europe / Univ. Jaume I de Castelló). arXiv:2011.10799 is the 2020 preprint; the **published version is IPIN 2021 (IEEE), pp. 1–8** (dblp confirms "IPIN 2021: 1-8"). Cite IPIN 2021 and note the 2020 preprint.
3. **WIO-EKF 2024 — VERIFIED.** Peng Zhou, Hao Wang, Raffaele Gravina, Fangmin Sun, "WIO-EKF: Extended Kalman Filtering-Based Wi-Fi and Inertial Odometry Fusion Method for Indoor Localization," IEEE Internet of Things Journal, vol. 11, no. 13, pp. 23592–23603, 2024. DOI 10.1109/JIOT.2024.3386889.
4. **WiMU 2025 — VERIFIED.** Qirui Yang, Huatao Xu, Mengxuan Song, Mo Li, "WiMU: Real-time Indoor Localization via Wi-Fi/IMU Fusion with Minimal Site Survey," Proc. ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), vol. 9, no. 4, Article 233, pp. 1–25, 2025. DOI 10.1145/3770667 (published 02 Dec 2025).
5. **AFT-VO 2022 — VERIFIED, and confirmed NOT WiFi.** Nimet Kaygusuz, Oscar Mendez, Richard Bowden, "AFT-VO: Asynchronous Fusion Transformers for Multi-View Visual Odometry Estimation," IEEE/RSJ IROS 2022; arXiv:2206.12946. It is multi-camera visual odometry (nuScenes/KITTI), not WiFi/IMU — exactly as your task anticipated.
6. **A-KIT 2024 — VERIFIED.** Nadav Cohen, Itzik Klein, "A-KIT: Adaptive Kalman-Informed Transformer." Built on a set-transformer to regress EKF process-noise covariance online for INS/DVL nonlinear sensor fusion (autonomous underwater vehicle). arXiv:2401.09987 (v1 Jan 18 2024; v2 Mar 7 2025). **Preprint only** as of this writing.

### Foundational references — all verified
- **Set Transformer**: Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh, "Set Transformer: A Framework for Attention-Based Permutation-Invariant Neural Networks," ICML 2019, PMLR 97:3744–3753. arXiv:1810.00825.
- **Deep Sets**: Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, Alexander J. Smola, "Deep Sets," NeurIPS (NIPS) 2017, pp. 3391–3401. arXiv:1703.06114.
- **Attention Is All You Need**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin, NeurIPS 2017, pp. 5998–6008. arXiv:1706.03762.
- **Neural ODE**: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud, "Neural Ordinary Differential Equations," NeurIPS 2018. (Cross-check the exact page range at camera-ready; the canonical venue is NeurIPS 2018.)
- **Neural CDE**: Patrick Kidger, James Morrill, James Foster, Terry Lyons, "Neural Controlled Differential Equations for Irregular Time Series," NeurIPS 2020 (Spotlight); arXiv:2005.08926.
- **mTAN**: Satya Narayan Shukla, Benjamin M. Marlin, "Multi-Time Attention Networks for Irregularly Sampled Time Series," ICLR 2021; arXiv:2101.10318.
- **IONet**: Changhao Chen, Xiaoxuan Lu, Andrew Markham, Niki Trigoni, "IONet: Learning to Cure the Curse of Drift in Inertial Odometry," AAAI 2018, pp. 6468–6476. DOI 10.1609/aaai.v32i1.12102.
- **RoNIN**: Hang Yan, Sachini Herath, Yasutaka Furukawa, "RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods," IEEE ICRA 2020, pp. 3146–3152. arXiv:1905.12853; DOI 10.1109/ICRA40945.2020.9196860.

## Details — Ranked Nearest Competitors

Each entry states which of your four target properties — (i) attention/transformer fusion, (ii) continuous-time OR async/multi-rate without resampling, (iii) missing/stale-modality robustness, (iv) cross-session/cross-day real-world generalization — the work HAS and LACKS; whether it uses a SINGLE UNIFIED fusion block vs PER-MODALITY branches; and whether it handles REAL-VALUED continuous time gaps vs fixed timesteps/resampling.

### Rank 1 — A-KIT (2024) · {attention/transformer/set; multimodal fusion}
**Metadata:** Nadav Cohen, Itzik Klein, "A-KIT: Adaptive Kalman-Informed Transformer," arXiv:2401.09987 (preprint only; v1 Jan 18 2024, v2 Mar 7 2025). Autonomous Navigation and Sensor Fusion Lab, University of Haifa.
**Method (2–3 sentences):** A-KIT couples an EKF with a tailored set-transformer that regresses the EKF's process-noise covariance online from time-series inertial + Doppler-velocity-log data, trained with a "Kalman-informed loss." On 86.6 minutes of real autonomous-underwater-vehicle data in the Mediterranean Sea, the authors report that "A-KIT outperforms the conventional EKF by more than 49.5% and model-based adaptive EKF by an average of 35.4% in terms of position accuracy."
**Relation / group:** The only verified work using a set-transformer for navigation sensor fusion; the authors explicitly cite permutation invariance as enabling consistent handling of "asynchronously arriving data." Group: **attention/transformer/set**.
- HAS: (i) attention/set-transformer; partial (ii) permutation-invariant handling of asynchronous arrivals; (iv) real-world (AUV) evaluation.
- LACKS: continuous-time Δt embedding of real-valued gaps; (iii) modality/instant dropout; WiFi/IMU domain.
- Fusion block: **set-transformer is auxiliary to a hand-built EKF** (estimates noise covariance, not position directly) — not an end-to-end unified fusion block.
- Time gaps: handled by the EKF propagation step, not a learned real-valued Δt embedding.
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** A-KIT is a hybrid model-based filter operating on INS/DVL, where the set-transformer only tunes EKF noise; it has no continuous-time Δt embedding, no modality/instant dropout, and the attention block does not itself fuse raw heterogeneous observations into a position.

### Rank 2 — AFT-VO (2022) · {attention/transformer/set; continuous-time/async}
**Metadata:** Nimet Kaygusuz, Oscar Mendez, Richard Bowden, "AFT-VO: Asynchronous Fusion Transformers for Multi-View Visual Odometry Estimation," IEEE/RSJ IROS 2022; arXiv:2206.12946.
**Method:** A Mixture Density Network estimates per-camera 6-DoF pose distributions; a transformer fusion module then combines asynchronous pose estimates from multiple unsynchronised cameras. A "Discretiser" module positionally encodes continuous timestamps and "Source Encoding" tags each sensor, so sources "need not be synchronised or at the same frequency."
**Relation / group:** Architecturally the closest to your async-fusion idea — a transformer fusing multi-source asynchronous signals with continuous-timestamp positional encoding. Group: **attention/transformer/set + continuous-time/async**.
- HAS: (i) transformer fusion; (ii) continuous-timestamp encoding of asynchronous, different-rate sources without resampling; (iii) robustness to per-sensor failure via uncertainty; (iv) evaluation on nuScenes/KITTI in challenging conditions.
- LACKS: WiFi/IMU domain; explicit modality/instant dropout training.
- Fusion block: transformer fuses **pre-computed per-camera pose estimates** (per-source MDN branches feed the fusion), not raw heterogeneous observations in one block.
- Time gaps: handles **real-valued continuous timestamps** (a genuine strength, via the Discretiser).
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** It is multi-camera visual odometry, not WiFi/IMU; it fuses per-source pose distributions rather than raw observations in a single block; and graceful degradation comes from uncertainty weighting, not modality/instant dropout.

### Rank 3 — WiMU (2025) · {WiFi fingerprinting; multimodal fusion}
**Metadata:** Qirui Yang, Huatao Xu, Mengxuan Song, Mo Li, "WiMU: Real-time Indoor Localization via Wi-Fi/IMU Fusion with Minimal Site Survey," Proc. ACM IMWUT, vol. 9, no. 4, Article 233, pp. 1–25, 2025. DOI 10.1145/3770667.
**Method:** WiMU integrates WiFi RSSI fingerprinting with IMU to enable real-time localization while drastically reducing the labor-intensive site survey, explicitly targeting the limited/irregular WiFi sampling-rate problem in practical deployments.
**Relation / group:** The most recent direct WiFi+IMU fusion competitor, addressing the same practical async/sampling-rate pain point you target. Group: **WiFi fingerprinting + multimodal fusion**.
- HAS: WiFi+IMU fusion; (iv) real-world real-time deployment focus.
- LACKS: (i) unified self-attention block (it is a fingerprinting+IMU system, not a single permutation-invariant transformer); (ii) continuous-time Δt embedding; (iii) modality/instant dropout.
- Fusion block: system-level integration of fingerprinting + IMU, **not a single unified attention block**.
- Time gaps: addresses low WiFi sampling rate at the system level, not via a real-valued Δt embedding.
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** WiMU is a survey-light WiFi/IMU system rather than a single permutation-invariant self-attention block; it uses neither a continuous-time Δt embedding nor modality/instant dropout for graceful degradation.

### Rank 4 — WIO-EKF (2024) · {WiFi fingerprinting; inertial/IMU; multimodal fusion}
**Metadata:** Peng Zhou, Hao Wang, Raffaele Gravina, Fangmin Sun, "WIO-EKF: Extended Kalman Filtering-Based Wi-Fi and Inertial Odometry Fusion Method for Indoor Localization," IEEE Internet of Things Journal, vol. 11, no. 13, pp. 23592–23603, 2024. DOI 10.1109/JIOT.2024.3386889.
**Method:** Two deep sub-models — a convolutional denoising-autoencoder WiFi regressor (CDAELoc) and a dual-branch deep inertial odometry net (DbDIO) — produce position estimates that an EKF fuses while mitigating DbDIO heading error. Evaluated on UJIIndoorLoc, RoNIN, and a self-collected set; CDAELoc alone cuts average positioning error (APE) by 12.5%, and "the APE of WIO-EKF is lower than those of CDAELoc and DbDIO by 34% and 42%."
**Relation / group:** Canonical recent WiFi+IMU deep fusion, but with classic per-modality branches + EKF late fusion. Group: **WiFi fingerprinting + inertial/IMU + multimodal fusion**.
- HAS: WiFi+IMU deep fusion; (iv) multi-dataset evaluation incl. a self-collected set.
- LACKS: (i) attention/transformer; (ii) continuous-time Δt; (iii) modality/instant dropout.
- Fusion block: **explicit per-modality deep branches fused by a hand-built EKF** — not a learned unified attention block.
- Time gaps: handled implicitly by the EKF propagation, not a real-valued Δt embedding.
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** It uses two separate deep branches glued by an EKF, has no continuous-time Δt encoding, and no modality/instant dropout regime.

### Rank 5 — iMoT (2025) · {inertial/IMU; attention/transformer/set}
**Metadata:** Son Minh Nguyen, Duc Viet Le, Paul J. M. Havinga, "iMoT: Inertial Motion Transformer for Inertial Navigation," Proc. AAAI Conf. on Artificial Intelligence, vol. 39, no. 6, pp. 6209–6217, 2025. DOI 10.1609/aaai.v39i6.32664; arXiv:2412.12190.
**Method:** A transformer encoder–decoder for inertial odometry that fuses acceleration and angular-velocity "modalities" via cross-modal attention; introduces a Progressive Series Decoupler, learnable query "motion particles" for motion uncertainty, and an Adaptive Positional Encoding (APE) that "dynamically modifies positional embeddings for temporal discrepancies between different modalities." In ablation, APE alone yields "∼11.11% in ATE reduction," underscoring its role in handling modal distinctions.
**Relation / group:** Strong on cross-modal transformer fusion and explicitly tackles temporal discrepancies between modalities. Group: **inertial/IMU + attention/transformer/set**.
- HAS: (i) cross-modal transformer fusion; partial (ii) adaptive positional encoding for temporal discrepancies; (iv) generalization evaluation on inertial benchmarks.
- LACKS: WiFi; real-valued continuous-time Δt for ~1 Hz vs ~30 Hz fusion; (iii) modality/instant dropout.
- Fusion block: cross-modal attention but the two "modalities" are **two inertial channels** (acc, gyro), not heterogeneous-rate sensors.
- Time gaps: APE handles token discrepancies but is **not a real-valued continuous-time Δt embedding** for cross-rate sensor fusion without resampling.
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** iMoT is IMU-only with two inertial channels; its adaptive positional encoding is not a continuous-time Δt embedding for asynchronous WiFi+IMU, and it has no modality/instant dropout.

### Rank 6 — Zhang et al. (2021) LSTM WiFi+Inertial fusion · {WiFi fingerprinting; inertial/IMU; multimodal fusion}
**Metadata:** Mingyang Zhang, Jie Jia, Jian Chen, Yansha Deng, Xingwei Wang, Abdol Hamid Aghvami, "Indoor Localization Fusing WiFi With Smartphone Inertial Sensors Using LSTM Networks," IEEE Internet of Things Journal, vol. 8, no. 17, pp. 13608–13623, 2021. DOI 10.1109/JIOT.2021.3067515.
**Method:** Formulates WiFi+PDR fusion as recursive function approximation; a sliding-window displacement scheme builds a time-series feature set fused by an LSTM for localization.
**Relation / group:** Direct WiFi+IMU deep-fusion predecessor using recurrence rather than attention. Group: **WiFi fingerprinting + inertial/IMU + multimodal fusion**.
- HAS: WiFi+IMU deep fusion in a single recurrent pipeline.
- LACKS: (i) attention/transformer; (ii) continuous-time Δt (uses resampled sliding windows); (iii) modality/instant dropout; (iv) explicit cross-session generalization.
- Fusion block: single LSTM pipeline, but **not permutation-invariant and requires regular spacing/alignment**.
- Time gaps: **fixed-timestep sliding windows (resampling to a common rate)**.
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** Recurrent fixed-timestep fusion requiring resampling; no continuous-time Δt, no permutation-invariant set block, no modality/instant dropout.

### Rank 7 — Yu et al. (2022) Multi-Modal Recurrent Fusion · {multimodal fusion; WiFi fingerprinting}
**Metadata:** Jianyuan Yu, Pu (Perry) Wang, Toshiaki Koike-Akino, Philip V. Orlik, "Multi-Modal Recurrent Fusion for Indoor Localization," IEEE ICASSP 2022; arXiv:2203.00510; IEEE Xplore doc 9746071.
**Method:** Localization framed as multi-modal sequence regression; a multi-stream recurrent fusion combines each modality's current hidden state while modeling per-modality uncertainty learned from its own past states. Fuses WiFi RSSI, IMU, and UWB on the SPAWC2021 dataset, beating trilateration, fingerprinting, and convolutional baselines.
**Relation / group:** Closest in spirit on per-modality uncertainty-aware fusion and graceful handling of unreliable modalities. Group: **multimodal fusion + WiFi fingerprinting**.
- HAS: WiFi+IMU(+UWB) fusion; per-modality uncertainty modeling (a soft form of robustness).
- LACKS: (i) attention/transformer; (ii) continuous-time Δt; (iii) explicit modality/instant dropout; (iv) cross-session generalization.
- Fusion block: **explicit multi-stream (per-modality) RNN branches** — not a single block; modality order fixed by the stream architecture (not permutation-invariant).
- Time gaps: recurrent fixed-timestep, **no real-valued Δt**.
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** Per-modality recurrent streams with uncertainty weighting rather than a single permutation-invariant block; no real-valued Δt and no modality/instant dropout.

### Rank 8 — mTAN (2021) · {continuous-time/async; attention/transformer/set}
**Metadata:** Satya Narayan Shukla, Benjamin M. Marlin, "Multi-Time Attention Networks for Irregularly Sampled Time Series," ICLR 2021; arXiv:2101.10318.
**Method:** Learns a continuous-time embedding and uses a time-attention mechanism (reference points as queries, observed times as keys) to produce fixed-length representations of sparse, irregularly sampled, multivariate series; encoder–decoder for interpolation/classification.
**Relation / group:** The canonical continuous-time attention model and a likely methodological building block / baseline for your Δt encoding. Group: **continuous-time/async + attention/transformer/set**.
- HAS: (i) attention; (ii) genuine continuous-time embedding of real-valued time, no resampling.
- LACKS: localization/WiFi/IMU application; (iii) modality/instant dropout; (iv) cross-session navigation evaluation.
- Fusion block: re-represents data at **fixed reference points via learned interpolation**, not a single permutation-invariant fusion block over modalities.
- Time gaps: handles **real-valued continuous time** (a strength).
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** General-purpose (health-record) time series; interpolates to fixed reference points rather than fusing modalities as a permutation-invariant set; no modality/instant dropout; not evaluated on cross-session localization.

### Rank 9 — Neural CDE (2020) · {continuous-time/async}
**Metadata:** Patrick Kidger, James Morrill, James Foster, Terry Lyons, "Neural Controlled Differential Equations for Irregular Time Series," NeurIPS 2020 (Spotlight); arXiv:2005.08926.
**Method:** Extends Neural ODEs to act directly on irregularly sampled, partially observed multivariate series via a continuous control path (cubic-spline/linear interpolation), with memory-efficient adjoint training across observations.
**Relation / group:** Foundational continuous-time model for irregular/asynchronous data; a baseline/justification for handling real-valued gaps without resampling. Group: **continuous-time/async**.
- HAS: (ii) continuous-time handling of irregular/partially observed series.
- LACKS: (i) attention/set; cross-modal WiFi/IMU fusion; (iii) modality/instant dropout; (iv) localization evaluation.
- Fusion block: ODE-based; interpolation path effectively **imputes between observations** rather than treating them as a permutation-invariant set.
- Time gaps: real-valued (via the control path).
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** ODE-based rather than attention/set-based; no cross-modal WiFi/IMU fusion and no modality/instant dropout.

### Rank 10 — Antsfeld et al. (2020/2021) Deep Sensors-WiFi Fusion · {WiFi fingerprinting; inertial/IMU}
**Metadata:** Leonid Antsfeld, Boris Chidlovskii, Emilio Sansano-Sansano, "Deep Smartphone Sensors-WiFi Fusion for Indoor Positioning and Tracking," arXiv:2011.10799 (2020); published IPIN 2021 (IEEE), pp. 1–8.
**Method:** Deep-learning PDR produces high-rate relative position, corrected by WiFi absolute-position predictions via a Kalman filter, then projected onto walkable paths. On the IPIN'19 Indoor Localization challenge dataset, the system "improves the winner's results by 20% using the challenge evaluation protocol."
**Relation / group:** Concrete WiFi+IMU asynchronous fusion baseline that explicitly contends with the multi-rate problem via Kalman filtering (WiFi corrects PDR drift "each time a WiFi scan is received"). Group: **WiFi fingerprinting + inertial/IMU**.
- HAS: WiFi+IMU async fusion (low-rate WiFi correcting high-rate PDR); (iv) real challenge-dataset evaluation.
- LACKS: (i) attention/transformer; (ii) learned continuous-time Δt; (iii) modality/instant dropout.
- Fusion block: **deep PDR + WiFi are separate models fused by a Kalman filter** — not a unified attention block.
- Time gaps: asynchronous WiFi updates handled by the KF, not a real-valued Δt embedding.
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** Two separate models glued by a Kalman filter; no continuous-time Δt embedding and no modality/instant dropout. (Note: the precise WiFi scan rate was not stated in the source; do not assert a specific Hz figure without the published PDF.)

### Rank 11 — RoNIN (2020) · {inertial/IMU}
**Metadata:** Hang Yan, Sachini Herath, Yasutaka Furukawa, "RoNIN: Robust Neural Inertial Navigation in the Wild," IEEE ICRA 2020, pp. 3146–3152. arXiv:1905.12853; DOI 10.1109/ICRA40945.2020.9196860.
**Method:** Benchmark + ResNet/LSTM/TCN architectures regressing velocity/position from IMU under natural motion. The dataset comprises 42.7 h of IMU-motion data over 276 sequences in 3 buildings from 100 subjects on three Android devices (Asus Zenfone AR, Samsung Galaxy S9, Google Pixel 2 XL); 85 subjects are used for train/val/test and 15 are held out for unseen-subject generalization.
**Relation / group:** The standard IMU-only deep inertial navigation baseline and a likely component/benchmark for your IMU stream; directly relevant to your cross-session generalization claim. Group: **inertial/IMU**.
- HAS: (iv) explicit unseen-subject/cross-device generalization evaluation.
- LACKS: (i) attention/set; (ii) continuous-time Δt; (iii) modality dropout; WiFi/fusion entirely.
- Fusion block: none (IMU-only); fixed-rate windows.
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** IMU-only with fixed-rate windows; no WiFi, no fusion, no continuous-time Δt, no attention/set block, no modality dropout.

### Rank 12 — IONet (2018) · {inertial/IMU}
**Metadata:** Changhao Chen, Xiaoxuan Lu, Andrew Markham, Niki Trigoni, "IONet: Learning to Cure the Curse of Drift in Inertial Odometry," AAAI 2018, pp. 6468–6476. DOI 10.1609/aaai.v32i1.12102.
**Method:** Segments inertial data into independent windows and uses LSTMs to regress polar displacement/heading change, breaking the continuous-integration drift cycle.
**Relation / group:** Seminal deep inertial odometry method underpinning the IMU side of most WiFi+IMU fusion work. Group: **inertial/IMU**.
- HAS: deep inertial odometry; generalization to non-periodic motion.
- LACKS: (i) attention/set; (ii) continuous-time Δt; (iii) modality dropout; WiFi/fusion entirely.
- Fusion block: none (IMU-only); fixed windows.
**How it differs from a continuous-time unified set-transformer with modality/instant dropout:** IMU-only recurrent windowed regression with no cross-modal fusion, no continuous-time Δt, no set/attention block, and no modality dropout.

### At-a-glance: the three architectural neighbors vs your design
| Property | A-KIT | AFT-VO | iMoT | **Your paper** |
|---|---|---|---|---|
| Attention/transformer (i) | Yes (set-transformer) | Yes | Yes | Yes (set-transformer) |
| Real-valued continuous-time Δt (ii) | No (EKF) | Yes (Discretiser) | Partial (APE) | **Yes** |
| Single unified fusion block | No (EKF-auxiliary) | No (per-source branches) | Cross-modal but IMU-only | **Yes (one self-attention block)** |
| Modality/instant dropout (iii) | No | No (uncertainty) | No | **Yes** |
| Domain | INS/DVL (AUV) | Multi-camera VO | IMU-only | **WiFi RSSI + IMU** |
| Cross-session real-world (iv) | Yes (AUV) | Yes (nuScenes/KITTI) | Yes (inertial) | **Yes (target)** |

## Recommendations
- **Stage 1 — Frame the novelty as the conjunction, not any single property.** No competitor holds all three contributions together. State explicitly that the continuous-time Δt embedding is shared (mTAN, AFT-VO's Discretiser, Neural CDE), the permutation-invariant set-transformer machinery is shared (A-KIT, Set Transformer), and the per-modality-branch-free single block plus modality/instant dropout for graceful degradation is the unfilled gap for WiFi+IMU localization.
- **Stage 2 — Organize related work by your five groups:** WiFi fingerprinting (WiMU, WIO-EKF, Antsfeld, Yu); inertial/IMU (RoNIN, IONet, iMoT); multimodal fusion (Yu, WIO-EKF, Zhang); attention/transformer/set (A-KIT, AFT-VO, iMoT, Set Transformer, Vaswani); continuous-time/async (mTAN, Neural CDE, Neural ODE, AFT-VO). Lead each group with the strongest 1–2 works.
- **Stage 3 — Include the contrast table above** to make the gap visually unambiguous for reviewers: A-KIT (set-transformer but EKF-auxiliary, INS/DVL), AFT-VO (async transformer but per-source VO branches), iMoT (cross-modal transformer but IMU-only, APE ≠ continuous Δt).
- **Stage 4 — Strengthen contribution (iii) and (iv) empirically.** Cross-day/cross-session generalization is under-evaluated in the WiFi+IMU literature; benchmark against WIO-EKF (UJIIndoorLoc/RoNIN), Yu (SPAWC2021), and Antsfeld (IPIN'19) on their own datasets, and report a missing/stale-modality ablation (drop WiFi entirely, stale-WiFi, dropped-IMU-windows) since none of the competitors do this directly.
- **Benchmark/threshold that would change the positioning:** If a reviewer surfaces a published WiFi+IMU (or WiFi-CSI+IMU) transformer that combines a real-valued time embedding with missing-modality training (none found here), narrow your headline claim to the *single-block, permutation-invariant, raw-observation* aspect. Before camera-ready, sweep IPIN 2023–2025, ACM IMWUT 2025, and IEEE IoT-J 2025 for any such work.

## Caveats
- **Preprint-only:** A-KIT (arXiv:2401.09987) appears to be preprint-only — do not assign it a journal/conference venue. Antsfeld's 2020 arXiv was published at IPIN 2021 — cite the IPIN 2021 version and note the 2020 preprint.
- **Metadata reconstructed from mirrors:** WiMU and the Zhang et al. LSTM-fusion metadata were confirmed via institutional/index mirrors (HKUST Research Portal, Semantic Scholar, dblp) because ACM DL and IEEE Xplore blocked direct fetches; volume/issue/pages are consistent across sources but verify against the publisher PDF at camera-ready. The Zhang et al. issue number (no. 17) is inferred from the volume-8 page range (13608–13623) and should be double-checked.
- **Title drift in your held list:** Reference (1) should be "Multi-Modal Recurrent Fusion for Indoor Localization" (ICASSP 2022), not "A Multi-Modal Recurrent Fusion-based Indoor Localization." Correct in your bibliography.
- **Neural ODE page range:** Confirm the exact NeurIPS 2018 pagination for Chen et al. at camera-ready; the canonical venue (NeurIPS 2018) is confirmed.
- **Search scope limit:** WiFi-CSI+IMU transformer fusion (as opposed to RSSI) was only lightly covered; I found no published WiFi-CSI+IMU unified transformer, but a targeted IPIN/IMWUT 2025 sweep is advisable before asserting absolute novelty. There also exist transformer-only WiFi-fingerprinting works (e.g., ViT-based RSSI/CSI localization) that are not WiFi+IMU fusion and are therefore weaker competitors than the ranked list above; cite them only as evidence that attention has reached WiFi localization, not as direct prior art for your fusion architecture.
- **Non-WiFi but methodologically central:** AFT-VO is visual odometry, clearly labeled — include it for its asynchronous-transformer methodology (Discretiser + Source Encoding), not as a WiFi competitor, exactly as your task anticipated.