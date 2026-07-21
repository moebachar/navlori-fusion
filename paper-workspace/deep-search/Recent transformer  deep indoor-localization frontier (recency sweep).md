# Related-Work References for a Continuous-Time Set-Transformer Fusing Asynchronous WiFi RSSI + IMU for Indoor (x,y) Localization

## TL;DR
- I verified **13 high-relevance references** (2016–2025, most 2023–2025) spanning all five requested categories; the strongest direct analogues are the **Set Transformer** (Lee et al., ICML 2019), a **set-transformer applied to RSSI localization** (Aristorenas, arXiv 2025), **ContiFormer** (NeurIPS 2023) and **mTAN** (ICLR 2021) for continuous-time/async modeling, and **WIO-EKF / WiMU / SmartFPS** for deep WiFi+IMU fusion.
- A genuine literature gap exists: no prominent *published* paper transfers the formal modality-dropout regularizer to WiFi/IMU/UWB indoor positioning, so the closest robustness anchors are **ModDrop** (IEEE TPAMI 2016, the canonical method) and **PEOPLEx** (IEEE ICC 2024, opportunistic graceful degradation) — you can legitimately claim this combination as novel.
- Every item below is a real publication with verified metadata; preprint-only items and any uncertain DOIs are flagged explicitly so you do not cite an unverified DOI.

## Key Findings
- The described paper's three contributions each have clear, citable precedents to contrast against: continuous-time Δt encoding (mTAN, ContiFormer), a single permutation-invariant attention block (Set Transformer, set-based RSSI localization), and async-robustness via dropout (ModDrop, PEOPLEx).
- Most existing WiFi+IMU fusion work still uses **filtering (EKF) or two-branch per-modality architectures** rather than a single joint cross-modal/cross-time attention block — which is precisely the structural novelty of the described work and the cleanest axis on which to differentiate it.
- Transformer-based inertial odometry (CTIN, iMoT, RIOT) is mature, but those models are **IMU-only** and do not fuse an absolute WiFi modality, so they cannot resolve the global (x,y) drift that the described WiFi anchor corrects.

## Details — Verified References (grouped by category)

### Group A — WiFi fingerprinting with attention/transformer

**1. All-embracing Transformers for fingerprint-based indoor localization**
- Son Minh Nguyen, Duc Viet Le, Paul J. M. Havinga. "Seeing the world from its words: All-embracing Transformers for fingerprint-based indoor localization." *Pervasive and Mobile Computing*, Elsevier, Vol. 100, Article 101912, 2024. DOI: 10.1016/j.pmcj.2024.101912.
- Method: Applies a Transformer self-attention mechanism to WiFi RSS fingerprints, learning to focus exclusively on relevant anchors/APs in an RSS sequence to extract subtle location-discriminative representations that are robust to environmental dynamics.
- Relation: Both use attention over WiFi RSS, but this is WiFi-only fingerprinting with no IMU, no continuous-time encoding, and no asynchronous fusion. Group: {WiFi fingerprinting | attention/transformer}.

**2. Set Transformer applied to RSSI localization**
- Aris J. Aristorenas. "Permutation-Invariant Transformer Neural Architectures for Set-Based Indoor Localization Using Learned RSSI Embeddings." arXiv preprint arXiv:2506.00656, 2025 (**preprint only**; Stanford University; no peer-reviewed version confirmed).
- Method: Models each WiFi scan as an unordered set of (BSSID, RSSI) pairs with learned BSSID embeddings, processed by a Set Transformer (SAB + PMA) to handle variable-length sparse inputs permutation-invariantly. Per the abstract, "a simple LSTM consistently outperformed all other models, achieving the lowest mean localization error across three tasks (E1–E3), with average errors as low as 2.23 m. The Set Transformer performed competitively, ranking second in every experiment."
- Relation: This is the closest published analogue to contribution (ii) — a set-transformer over WiFi observations — but it is WiFi-only, has no IMU modality, no Δt continuous-time encoding, and no modality dropout; it also reports the Set Transformer *under-performing* a plain LSTM, which the described joint cross-modal design can be positioned to overturn. Group: {WiFi fingerprinting | attention/transformer/set}.

### Group B — Deep multimodal WiFi + IMU / inertial fusion

**3. WIO-EKF**
- Ping Zhou, Hao Wang, Raffaele Gravina, Fangmin Sun. "WIO-EKF: Extended Kalman Filtering-Based Wi-Fi and Inertial Odometry Fusion Method for Indoor Localization." *IEEE Internet of Things Journal*, Vol. 11, No. 13, pp. 23592–23603, 2024. DOI: 10.1109/JIOT.2024.3386889.
- Method: Combines a convolutional denoising-autoencoder WiFi fingerprint regressor (CDAELoc, which reduces average positioning error by 12.5%) and a dual-branch deep inertial odometry network (DbDIO), fused by an EKF using both networks' outputs as observations; the fused WIO-EKF average positioning error is "lower than those of CDAELoc and DbDIO by 34% and 42%." Evaluated on UJIIndoorLoc, RoNIN, and a self-collected dataset.
- Relation: Same sensor pair (WiFi + IMU) and same (x,y) goal, but uses separate per-modality deep branches fused by a hand-designed EKF rather than a single joint-attention block; no continuous-time Δt token, no modality dropout. Group: {multimodal fusion | inertial/IMU | WiFi fingerprinting}.

**4. WiMU**
- Qirui Yang, Huatao Xu, Mengxuan Song, Mo Li. "WiMU: Real-time Indoor Localization via Wi-Fi/IMU Fusion with Minimal Site Survey." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, Vol. 9, No. 4, Article 233, pp. 1–25, 2025. DOI: 10.1145/3770667.
- Method: Fuses WiFi RSSI fingerprinting with IMU for real-time localization while minimizing the site-survey burden, explicitly targeting the low/asynchronous WiFi sampling-rate problem of practical deployments; per the IMWUT record it "achieves an average localization error of 5.247 meters" using minimal labeled data while outperforming three state-of-the-art methods.
- Relation: Same modality pair and same practical motivation (asynchronous, low-rate WiFi), but is a systems/fusion pipeline rather than a single permutation-invariant continuous-time transformer. Group: {multimodal fusion | WiFi fingerprinting | inertial/IMU}.

**5. SmartFPS**
- Liang Hua, Yuan Zhuang, Jun Yang. "SmartFPS: Neural network based wireless-inertial fusion positioning system." *Frontiers in Neurorobotics*, Vol. 17, Article 1121623, 2023. DOI: 10.3389/fnbot.2023.1121623.
- Method: An end-to-end neural network replaces empirical propagation models and filters for wireless-inertial fusion positioning, plus a transfer-learning strategy to handle distribution shift across sessions; in the whole-floor scenario "the average positioning accuracy of the fusion network is 0.506 meters," and transfer learning improved fusion accuracy by 31.6% (and pedestrian step/rotation accuracy by 53.3%).
- Relation: Shares the deep WiFi+inertial fusion goal and the cross-session generalization concern, but uses conventional neural nets (not attention/set), no Δt continuous-time encoding, and no async/dropout robustness mechanism. Group: {multimodal fusion | inertial/IMU | WiFi fingerprinting}.

### Group C — Continuous-time / irregular-time-series transformers (async fusion)

**6. ContiFormer**
- Yuqi Chen, Kan Ren, Yansen Wang, Yuchen Fang, Weiwei Sun, Dongsheng Li. "ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling." *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. arXiv:2402.10635.
- Method: Extends Transformer self-attention to continuous time by combining Neural-ODE latent trajectories with attention (CT-MHA), so observations at arbitrary/irregular timestamps can be modeled without resampling. The authors report ContiFormer "outperforms all the baselines by a large margin when 70% observations are dropped," confirming suitability for irregular series.
- Relation: Directly supports contribution (i) — a principled continuous-time attention alternative to the paper's Δt-elapsed-time encoding — but is a generic time-series model, not applied to localization or to cross-modal WiFi/IMU fusion. Group: {continuous-time/async | attention/transformer}.

**7. mTAN (Multi-Time Attention Networks)**
- Satya Narayan Shukla, Benjamin M. Marlin. "Multi-Time Attention Networks for Irregularly Sampled Time Series." *International Conference on Learning Representations (ICLR) 2021*. arXiv:2101.10318.
- Method: Introduces a learned continuous-time embedding plus a time-attention mechanism that re-represents sparse, irregularly sampled multivariate series at fixed reference points, replacing fixed interpolation kernels.
- Relation: A foundational precedent for contribution (i): learned continuous-time embeddings for unequal/asynchronous rates without resampling; however it targets clinical time series, is older (2021), and is not multimodal localization. Group: {continuous-time/async | attention/transformer}.

### Group D — Set transformers / permutation-invariant attention + transformer inertial odometry

**8. Set Transformer (the original)**
- Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh. "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks." *Proceedings of the 36th International Conference on Machine Learning (ICML 2019)*, PMLR Vol. 97, pp. 3744–3753. arXiv:1810.00825.
- Method: An attention-based set-input architecture (SAB/ISAB encoder, PMA decoder) that is permutation-invariant and a universal approximator of permutation-invariant functions, with inducing points reducing self-attention cost from quadratic to linear.
- Relation: The architectural foundation for contribution (ii) — permutation invariance over a set of observations; the described paper extends this to mixed-modality, continuous-time tokens. Group: {attention/transformer/set}.

**9. CTIN**
- Bingbing Rao, Ehsan Kazemi, Yifan Ding, Devu Manikantan Shila, Frank M. Tucker, Liqiang Wang. "CTIN: Robust Contextual Transformer Network for Inertial Navigation." *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 36, No. 5, pp. 5413–5421, 2022. DOI: 10.1609/aaai.v36i5.20479. arXiv:2112.02143.
- Method: A ResNet encoder enhanced with local/global multi-head self-attention extracts spatial context from IMU windows; a Transformer decoder fuses temporal knowledge, with multi-task uncertainty-weighted learning of velocity and trajectory. Evaluated on RIDI, OxIOD, RoNIN, IDOL, and the authors' own dataset.
- Relation: Transformer over IMU, but IMU-only (no WiFi), no continuous-time Δt token, no async robustness; demonstrates that attention is effective for inertial odometry. Group: {inertial/IMU | attention/transformer}.

**10. iMoT**
- Son Minh Nguyen, Duc Viet Le, Paul J. M. Havinga. "iMoT: Inertial Motion Transformer for Inertial Navigation." *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 39, No. 6, pp. 6209–6217, 2025. DOI: 10.1609/aaai.v39i6.32664. arXiv:2412.12190.
- Method: An encoder–decoder inertial transformer that retrieves cross-modal information between acceleration and angular-velocity modalities, with a Progressive Series Decoupler, Adaptive Positional Encoding for inter-modality temporal discrepancies, and learnable query "motion particles" for motion uncertainty.
- Relation: Closest in spirit to cross-modal attention and adaptive positional encoding for temporal discrepancies, but treats only the two IMU sub-modalities (no WiFi) and performs odometry (relative motion) rather than absolute (x,y) localization. Group: {inertial/IMU | attention/transformer/set | continuous-time/async}.

**11. RIOT**
- James Brotchie, Wenchao Li, Andrew D. Greentree, Allison Kealy. "RIOT: Recursive Inertial Odometry Transformer for Localisation from Low-Cost IMU Measurements." *Sensors (MDPI)*, Vol. 23, No. 6, Article 3217, 2023. DOI: 10.3390/s23063217.
- Method: Two self-attention-based frameworks for pose-invariant deep inertial odometry that incorporate position priors recursively to learn motion characteristics and systemic drift/bias (sequence-length-weighted relative trajectory error ≤ 0.4594 m).
- Relation: Transformer-based inertial localization, IMU-only, no WiFi fusion and no continuous-time encoding; a relevant attention-for-IMU baseline. Group: {inertial/IMU | attention/transformer}.

### Group E — Robustness to missing/stale modalities (modality dropout, graceful degradation)

**12. ModDrop (canonical modality-dropout method)**
- Natalia Neverova, Christian Wolf, Graham W. Taylor, Florian Nebout. "ModDrop: Adaptive Multi-Modal Gesture Recognition." *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, Vol. 38, No. 8, pp. 1692–1706, 2016. DOI: 10.1109/TPAMI.2015.2461544. arXiv:1501.00102.
- Method: Trains a multimodal network with gradual fusion and random dropping of entire modality channels, forcing the model to learn cross-modal correlations while keeping each modality's standalone representation, so inference degrades gracefully under missing/noisy channels.
- Relation: The canonical precedent for contribution (iii) — modality dropout for graceful degradation — but in gesture recognition, not localization, and without continuous-time fusion. Group: {multimodal fusion}.

**13. PEOPLEx (opportunistic, graceful degradation in localization)**
- Pierre-Yves Lajoie, Bobak H. Baghi, Sachini Herath, Francois Hogan, Xue Liu, Gregory Dudek. "PEOPLEx: PEdestrian Opportunistic Positioning LEveraging IMU, UWB, BLE and WiFi." *ICC 2024 – IEEE International Conference on Communications*, 2024 (IEEE Xplore document 10622566). arXiv:2311.18182. **DOI suffix/pages not independently verified — confirm on IEEE Xplore before camera-ready.**
- Method: A factor-graph optimization framework using IMU pedestrian inertial navigation as the backbone and opportunistically integrating UWB/BLE/WiFi only when available, explicitly rejecting the assumption of constant sensor availability and requiring no prior anchor/RF maps.
- Relation: Directly tackles graceful degradation under missing modalities for pedestrian positioning, but via classical optimization rather than a learned attention model with modality dropout. Group: {multimodal fusion | inertial/IMU | continuous-time/async}.

## Recommendations
- **Stage 1 — anchor the novelties:** Cite items 6, 7, 8 as the methodological lineage for the paper's core contributions (continuous-time embeddings → mTAN/ContiFormer; permutation-invariant attention → Set Transformer). Frame contribution (i) as bringing mTAN/ContiFormer-style continuous-time encoding into multimodal localization, and (ii) as extending the Set Transformer from single-modality sets to mixed-modality, time-stamped tokens.
- **Stage 2 — the direct comparison set:** Cite items 2, 3, 4, 5 as the WiFi+IMU localization baselines, emphasizing that all use either per-modality branches + EKF (WIO-EKF, SmartFPS) or systems pipelines (WiMU), versus your single joint-attention block; item 2 is the only set-transformer localization work and is WiFi-only.
- **Stage 3 — motivate async-robustness:** Cite 12 (ModDrop) for the method and 13 (PEOPLEx) for the localization analogue; explicitly state that combining learned modality/instant dropout with continuous-time attention for WiFi+IMU localization is unaddressed in prior work.
- **Benchmarks that would change the framing:** If a peer-reviewed venue version of item 2 appears, upgrade it from preprint; if you find a localization paper that *learns* modality dropout (not just opportunistic fusion), your novelty claim in Group E must be softened to "first continuous-time set-transformer with modality dropout" rather than "first modality dropout in WiFi/IMU localization."
- Items 9, 10, 11 (CTIN, iMoT, RIOT) should be cited together as the transformer-inertial-odometry baseline, noting they are IMU-only and cannot correct absolute drift — the motivation for adding the WiFi modality.

## Caveats
- **Item 2 (Aristorenas, arXiv:2506.00656)** is arXiv-only; no peer-reviewed version confirmed. Cite as a preprint.
- **Item 13 (PEOPLEx)**: the exact IEEE DOI suffix and page numbers were not independently confirmed; the arXiv ID (2311.18182) and IEEE Xplore document number (10622566) are verified — confirm the DOI on the publisher page before final submission.
- **Item 10 (iMoT)** and **Item 1 (All-embracing Transformers)** share authorship (Nguyen, Le, Havinga). Under double-blind review, citing several papers from one group is fine, but avoid phrasing that could de-anonymize you if any of these are your own prior work.
- **No published paper explicitly applies modality-dropout to WiFi/IMU/UWB indoor localization** — this is a real gap (confirmed across IEEE/ACM/Elsevier/arXiv searches), which strengthens contribution (iii) but means item 12 (gesture recognition) and item 13 (optimization-based) are the nearest available anchors rather than exact precedents.
- Two additional real, on-topic candidates surfaced but are **excluded** to keep the list tight and fully aligned: a *Future Generation Computer Systems* (Elsevier, 2024) robust-multimodal-localization paper by Qinghu Wang et al. (DamLoc; magnetic+BLE, exact DOI/pages unverified) and *EKF-Based Fusion of Wi-Fi/LiDAR/IMU* (arXiv:2509.23118, 2025, preprint, classical EKF not deep/attention). Verify DamLoc's DOI on ScienceDirect if you wish to add it; the LiDAR/EKF paper is less relevant given its non-learned fusion.