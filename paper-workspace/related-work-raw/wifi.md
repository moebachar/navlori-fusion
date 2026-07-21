# Related Work Evidence — Group: WiFi Fingerprinting

NotebookLM notebook: `0befa2ea-b3b3-4da3-8f45-0f768631969a`
Gatherer: evidence pass for ICINCO 2026 paper "Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization".
All quotes below are verbatim from NotebookLM grounded answers (source_id given for each). Numbers NOT quotable are marked UNSUPPORTED.

Our 3 contributions (the gap matrix targets these):
- (i) CONTINUOUS-TIME: learned sinusoidal encoding of real-valued elapsed Delta-t per observation; no resampling, no ODE.
- (ii) SINGLE PERMUTATION-INVARIANT SET-TRANSFORMER: one self-attention block does BOTH cross-modal AND cross-time fusion over (modality, time) tokens.
- (iii) ASYNC-ROBUSTNESS: modality-dropout (0.4) + instant-dropout (0.45); graceful degradation under missing/stale sensors, shown via real-world cross-session generalization.

---

## A. CAPSULES

### PILLAR / highest-priority competitor — Aristorenas 2025 (Set Transformer over RSSI) [f6417660]
- **Citation:** Aristorenas, A. J. (2025). Permutation-Invariant Transformer Neural Architectures for Set-Based Indoor Localization Using Learned RSSI Embeddings. Stanford University (preprint).
- **Method:** Each WiFi scan modeled as an unordered set of (BSSID, RSSI) pairs; BSSIDs -> learned embeddings concatenated with RSSI; processed by a Set Transformer.
- **Modalities:** WiFi RSSI ONLY. (Author lists BLE/Wi-Fi/other-sensor fusion as FUTURE work.)
- **Time handling:** NONE across time — each scan is an independent set, processed individually (batch=1), no padding/masking, no temporal sequence, no Delta-t.
- **Robustness:** Missing-AP robustness is inherent to the set formulation — undetected BSSIDs simply excluded, no imputation. NO modality dropout (single modality).
- **Datasets:** Campus environment, six buildings; E1 single-building/single-floor, E2 multi-building/first-floor, E3 single-building/multi-floor.
- **Headline result (quoted):** "a simple LSTM consistently outperformed all other models ... with average errors as low as 2.23 m." Set Transformer = 3.82 +/- 2.35 m on E1 (2nd to LSTM). [f6417660]
- **Limitation:** Single modality (RSSI), single time-step (no temporal fusion), its own Set Transformer was BEATEN by a plain LSTM; cross-building tested but NOT cross-session/cross-day.
- **diff_vs_ours:** Closest architectural cousin to our pillar (ii) but (a) single modality vs our WiFi+IMU, (b) NO continuous-time / no temporal fusion at all (our (i)), (c) no modality/instant dropout and no chronological cross-session protocol (our (iii)). We use one set-transformer for BOTH cross-modal AND cross-time; they use it for cross-AP within a single scan only.
- **key_quote:** "We propose a permutation-invariant neural architecture for indoor localization using RSSI scans from Wi-Fi access points." [f6417660] ; "To accommodate variable-length scans, each RSSI set is processed individually (batch size = 1), avoiding the need for explicit padding or masking." [f6417660]

### BENCHMARK — Torres-Sospedra et al. 2014 UJIIndoorLoc [46b80222]
- **Citation:** Torres-Sospedra, J., Montoliu, R., Martinez-Uso, A., Avariento, J. P., Arnau, T. J., Benedito-Bordonau, M., & Huerta, J. (2014). UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems. In Proc. 5th Int. Conf. on Indoor Positioning and Indoor Navigation (IPIN 2014), Busan, Korea, pp. 261-270.
- **Method/role:** The canonical public WiFi fingerprint benchmark we use.
- **Size (quoted):** "3 buildings with 4 or 5 floors"; "933" reference points; "21049 sampled points ... 19938 for training/learning and 1111 for validation/testing"; "520" WAPs. Each record = 529 numeric elements (520 RSSI + coords/IDs/timestamp).
- **CROSS-SESSION property (load-bearing for us):** "Dataset independence has been assured by taking Validation (or testing) samples 4 months after Training ones." [46b80222]
- **Modalities:** WiFi RSSI ONLY — no IMU/inertial. RSSI is 98% of each record (520/529).
- **Headline / canonical baseline (quoted):** "With the basic indoor location system the error is, in average, 7.9 meters when the fingerprint has been located in the correct building and floor." (1NN + Euclidean; success rate 89.92%.) [46b80222]
- **diff_vs_ours:** WiFi-only; we need WiFi+IMU + asynchronous time, so this benchmark is the WiFi half of our setup and the source of the 4-month cross-session split that motivates our (iii).

### BENCHMARK — Bahl & Padmanabhan 2000 RADAR [e7ac47fb]
- **Citation:** Bahl, P., & Padmanabhan, V. N. (2000). RADAR: An In-Building RF-based User Location and Tracking System. IEEE INFOCOM 2000.
- **Method:** WiFi RSSI nearest-neighbor(s) in signal space (NNSS) — classical kNN fingerprint matching.
- **Modalities:** WiFi RSSI (signal strength) only.
- **Headline (quoted):** "Table 1 The 25th, 50th, and 75th percentile values of the error distance. [...] Empirical 1.92 2.94 4.69" -> median 2.94 m. [e7ac47fb]
- **diff_vs_ours:** Non-learned, single-scan, single-modality, no time/async, no robustness mechanism. Pure baseline. (Note: on UJIIndoorLoc RADAR scores ~9.21 m MAE per Nguyen 2024 Table 3 [d4577fc7].)
- **key_quote:** "We term our general technique nearest neighbor(s) in signal space (NNSS)." [e7ac47fb]

### BENCHMARK — Youssef & Agrawala 2005 Horus [a7a2adb1]
- **Citation:** Youssef, M., & Agrawala, A. (2005). The Horus WLAN Location Determination System. MobiSys 2005.
- **Method:** Probabilistic RSSI fingerprinting with location-clustering to cut computation.
- **Modalities:** WiFi RSSI only.
- **Headline (quoted):** "It has an error of less than 0.6 meter on the average ..." [a7a2adb1] (NOTE: this 0.6 m is in the Horus source's own controlled testbed; not comparable to UJIIndoorLoc scale.)
- **diff_vs_ours:** Probabilistic, single-modality, no temporal/async modeling, no learned representation, no robustness/cross-session protocol.
- **key_quote:** "It has an error of less than 0.6 meter on the average and its computational requirements are more than an order of magnitude better than other WLAN location determination systems." [a7a2adb1]

### BENCHMARK — Wang et al. 2015 DeepFi [4d28b1dc]
- **Citation:** Wang, X., Gao, L., Mao, S., & Pandey, S. (2015). DeepFi: Deep Learning for Indoor Fingerprinting Using Channel State Information. IEEE WCNC 2015.
- **Method:** Deep-learning fingerprinting on WiFi CSI; off-line greedy layer-wise weight training, on-line RBF probabilistic estimation.
- **Modalities:** WiFi CSI.
- **Headline (quoted):** "We can see that in the living room experiment, the mean distance error is about 0.95 meter for DeepFi with a single AP." (computer lab ~1.8 m). [4d28b1dc]
- **diff_vs_ours:** CSI not RSSI, single modality, small controlled rooms, no temporal/async handling, no missing-modality robustness, no cross-session.
- **key_quote:** "In this paper, we present a novel deep learning based indoor fingerprinting system using Channel State Information (CSI), which is termed DeepFi." [4d28b1dc]

### BENCHMARK / competitor — Song et al. 2019 CNNLoc [8739f6cf]
- **Citation:** Song, X., et al. (2019). CNNLoc: A CNN-based indoor localization system with WiFi fingerprints (SAE + 1D-CNN); preliminary version UIC 2019.
- **Method:** Stacked Auto-Encoder (feature extraction from sparse RSS) + 1D-CNN positioning; multi-building/multi-floor.
- **Modalities:** WiFi RSSI.
- **Datasets:** UJIIndoorLoc, Tampere, and own UTSIndoorLoc.
- **Headline (quoted):** "compared with 11.78m on UJIIndoorLoc dataset." (building 100% / floor 95% success). [8739f6cf] (Nguyen 2024 reproduces 11.78 m UJIIndoorLoc for CNNLoc in Table 3 [d4577fc7].)
- **diff_vs_ours:** CNN (no attention), single modality, fixed-length RSS vector, no temporal/async, static imputation only, no cross-session generalization protocol.
- **key_quote:** "In this paper, we propose CNNLoc, a Convolutional Neural Network (CNN) based indoor localization system with WiFi fingerprints for multi-building and multi-floor localization." [8739f6cf]

### COMPETITOR — Tiku et al. 2022 ANVIL [c114e66e]
- **Citation:** Tiku, S., Gufran, D., & Pasricha, S. (2022). Multi-Head Attention Neural Network for Smartphone Invariant Indoor Localization. (Colorado State Univ.)
- **Method:** ANVIL — multi-head attention, calibration-free, deployable on phones; targets device heterogeneity.
- **Modalities:** WiFi RSSI only.
- **Time handling:** None — static offline/online fingerprint matching; no Delta-t.
- **Robustness:** Missing-AP handling via -100 dB imputation + an "AP Dropout" training augmentation (feature-level dropout, NOT modality-level / not stale-sensor).
- **Headline (quoted):** "up to 35% accuracy improvement compared to state-of-the-art indoor localization techniques." [c114e66e] (relative; absolute metres UNSUPPORTED in our quotes).
- **XSESS:** Cross-DEVICE / device-heterogeneity generalization (not cross-day/session).
- **diff_vs_ours:** Attention yes, but single modality, no continuous-time, no temporal fusion, AP-dropout != modality/instant dropout for stale sensors; cross-device not chronological cross-session.
- **key_quote:** "We propose a multi-head attention neural network-based indoor localization framework that is resilient to device heterogeneity." [c114e66e]

### COMPETITOR — Zhang et al. 2022 TIPS [07285e43]
- **Citation:** Zhang, Z., Du, H., Choi, S., & Cho, S. H. (2022). TIPS: Transformer Based Indoor Positioning System Using Both CSI and DoA of WiFi Signal. IEEE Access.
- **Method:** GPT-style decoder transformer with masked self-attention; routes treated as "sentences", positions as "words"; embeds DoA on top of CSI.
- **Modalities:** WiFi CSI + DoA (direction of arrival). Both WiFi-derived; no IMU.
- **Time handling:** Discrete sequence ("words" in a route), NO real-valued time gaps / async.
- **Robustness:** No modality-dropout; separate CSI-only/DoA-only/CSI+DoA models, jointly trained models degrade when tested on a single modality/SNR.
- **Headline (quoted):** "the proposed TIPS can reduce the positioning error down to 20 cm." [07285e43] (ray-tracing simulation + RF testbed.)
- **XSESS:** No cross-session/cross-day; robustness studied across SNR levels only.
- **diff_vs_ours:** Transformer yes, but both modalities are WiFi-derived (no inertial), discrete-time autoregressive (not continuous-time set), no missing-modality dropout, no cross-session.
- **key_quote:** "In order to integrate the two heterogeneous fingerprints, DoA and CSI, we adopt the transformer neural network ..." [07285e43]

### COMPETITOR — Zhang et al. 2023 AAResCNN [52b5f512]
- **Citation:** Zhang et al. (2023). Attention-Augmented Residual CNN (AAResCNN) for CSI-based fingerprinting + universal IMU-aided tracking (plug-and-play).
- **Method:** AAResCNN (attention-augmented residual blocks, CNN hybrid) for CSI positioning; separate universal tracking net (deep trajectory prior) that fuses IMU via plug-and-play optimization.
- **Modalities:** WiFi CSI (positioning) + IMU (tracking, fused later, decoupled). One of the few WiFi+IMU papers in group.
- **Time handling:** Discrete T time-step sequences; no continuous-time / real-valued gaps.
- **Robustness:** No inference-time missing-modality dropout; modular plug-and-play separates IMU from CSI.
- **Headline (quoted):** "the proposed AAResCNN improves the positioning accuracy of Chin20 by about 8%-48% in different cases." [52b5f512] (relative; absolute mm-scale tracking in controlled testbeds.)
- **XSESS:** Cross-ENVIRONMENT tracking generality (tracking net trained on trajectory data alone, "compatible with arbitrary environment and IMUs with arbitrary precision"); not chronological cross-session WiFi.
- **diff_vs_ours:** Does fuse WiFi+IMU, BUT (a) CSI not RSSI, (b) CNN+model-based PnP not a single transformer, (c) two SEPARATE stages (positioning then tracking) = branches not a unified block, (d) discrete time not continuous-time, (e) no modality/instant dropout.
- **key_quote:** "To improve the positioning performance, we develop a novel attention-augmented residual convolutional neural network (AAResCNN) for CSI-based fingerprinting indoor positioning." [52b5f512] ; "The proposed tracking system is compatible with arbitrary environment and IMUs with arbitrary precision." [52b5f512]

### COMPETITOR — Ott et al. 2024 Radio Foundation Models (TF-PT) [5aa39fa5]
- **Citation:** Ott, J., Pirkl, J., Stahlke, M., Feigl, T., & Mutschler, C. (2024). Radio Foundation Models: Pre-training Transformers for 5G-based Indoor Localization. Fraunhofer IIS.
- **Method:** Self-supervised pretraining of a general transformer on 5G Channel Impulse Responses; pretext = randomly mask & discard input, reconstruct; then fine-tune ("RadioGPT" direction).
- **Modalities:** 5G CIR (channel impulse responses); NOT WiFi RSSI, no IMU.
- **Time handling:** 1D sequences of fixed-length vectors (N_an * l_s); attention over time-steps, but no real-valued Delta-t / async — discrete alignment.
- **Robustness:** Masking-based pretext gives reconstruction of missing input (related to missing-data robustness).
- **Headline (quoted):** "from N=500 with a CE90 = 0.955 m to N=100k with a CE90 = 0.398 m ... on the Industrial dataset." [5aa39fa5]
- **XSESS:** Cross-SITE evaluation (TF-C-PT = pretrain on one site, fine-tune on other).
- **diff_vs_ours:** Transformer + cross-site + missing-data pretext, BUT 5G CIR single modality (no IMU/WiFi-RSSI fusion), discrete sequences (no continuous-time Delta-t), masking pretext != train-time modality/instant dropout for stale sensors at inference.
- **key_quote:** "In this paper we propose a self-supervised learning framework that pre-trains a general transformer (TF) neural network on 5G channel measurements ..." [5aa39fa5]

### COMPETITOR — Bhatia et al. 2025 Locaris [c4f1526c]
- **Citation:** Bhatia, N. S., Kocheta, P., Elliott, R., Kuttivelil, H. S., & Obraczka, K. (2025). Indoor Localization using Compact, Telemetry-Agnostic, Transfer-Learning Enabled Decoder-Only Transformer (Locaris). UC Santa Cruz.
- **Method:** Decoder-only LLM; each AP measurement = a token; ingests raw telemetry without preprocessing/padding/imputation; fine-tune across datasets; few-shot.
- **Modalities:** WiFi FTM (fine-time measurement) + RSSI. No IMU.
- **Time handling:** Token-per-reading, NOT continuous-time; no real-valued Delta-t. (Variable-length, schema-free, but not async time gaps.)
- **Robustness:** Native missing-modality handling — "provide whichever modality is available (FTM-only or RSSI-only) without requiring placeholders." This is the CLOSEST in-group to modality-dropout robustness.
- **Headline (quoted):** "reaching 0.88 meter error with only 3% of the target data, while the best baseline remains at 1.88m, a 53% performance gap." [c4f1526c]
- **XSESS:** True CROSS-ENVIRONMENT transfer protocol: "trained on two source environments and then provided a fraction (1-100%) of the training data from a held out target environment." [c4f1526c] (cross-building, not chronological cross-day.)
- **diff_vs_ours:** Transformer + flexible-modality + cross-environment, BUT (a) both modalities are WiFi (no IMU inertial), (b) token-per-reading != learned continuous-time Delta-t encoding, (c) decoder-only autoregressive, not permutation-invariant set over (modality,time), (d) no explicit instant/stale-time dropout.
- **key_quote:** "in the ablation settings, we simply provide whichever modality is available (FTM-only or RSSI-only) without requiring placeholders." [c4f1526c]

### COMPETITOR — Abdullah et al. 2025 RIS-assisted Transformer [1e099b7b]
- **Citation:** Abdullah, O., et al. (2025). Transformer-based RIS-assisted multimodal indoor localization for 6G IoT (CSI + RSS + geometric priors + RIS phase).
- **Method:** Multi-head self-attention transformer; 8 tokens (5 CSI subbands, 1 RSS, 1 RIS phase, 1 [CLS]); CLS regresses 3D position. UNIFIED single block over all modality tokens.
- **Modalities:** CSI + RSS + geometric priors + RIS phase (all RF; no IMU).
- **Time handling:** Synchronous discrete token sequence, NO real-valued time / async.
- **Robustness:** No dynamic inference dropout; ablation drops modalities to measure degradation (RSS->0.48, CSI->0.57, RIS->0.76).
- **Headline (quoted):** "The full Transformer model ... achieves the lowest average localization error of 0.31 meters." [1e099b7b] (3D simulation; Rayleigh fading.)
- **XSESS:** No cross-session/cross-day/cross-env protocol; robustness over SNR 0-30 dB and #RIS elements only.
- **diff_vs_ours:** Has a UNIFIED multimodal transformer block with [CLS] readout (architecturally similar to our fusion), BUT (a) all RF modalities (no inertial), (b) simulation only, (c) no continuous-time Delta-t, (d) static modality ablation != train-time modality/instant dropout, (e) no real-world cross-session.
- **key_quote:** "The input sequence consists of eight tokens: five derived from CSI, one from RSS, one from RIS phase settings, and a special [CLS] token." [1e099b7b]

### COMPETITOR — Nguyen et al. 2024 All-embracing Transformers (eAaT+) [d4577fc7]  *** in-domain key competitor: uses Anchor2Vec == our WiFiNet ***
- **Citation:** Nguyen, S. M., et al. (2024). All-embracing Transformers (AaTs) for RSS fingerprint indoor localization. Pervasive and Mobile Computing 100, 101912.
- **Method:** Transformer encoder over RSS with an "Anchor2Vec" tokenization layer (N anchors -> k+1 tokens, each d attributes); covariance+variance sub-constraints (multi-task, Adaptive Random Loss Weighting) to prevent representation collapse; eAaT+ reorders encoder-block layers for gradient flow. Params: k=64 tokens, d=128 attributes.
- **Modalities:** WiFi RSS only.
- **Time handling:** Static per-fingerprint vectors via Anchor2Vec; NO temporal/async.
- **Robustness:** Static imputation (missing anchors filled with 100 dB). No modality/instant dropout.
- **Datasets:** UJIIndoorLoc, UTS, Tampere.
- **Headline (quoted, MAE):** UJIIndoorLoc eAaT+ MAE = "8.16" m; UTS eAaT+ = 6.78 m; Tampere eAaT+ = 8.14 m. Table also lists RADAR 9.21 m, Weighted-KNN 9.33 m, CNNLoc 11.78 m on UJIIndoorLoc. [d4577fc7]
- **XSESS:** Environmental-heterogeneity robustness (Tampere unseen-test split: 697 train / 3951 unseen test); UJIIndoorLoc inherits the 4-month split. Not framed as chronological cross-session.
- **diff_vs_ours:** DIRECT WiFi-encoder relative — our WiFiNet was literally renamed from Anchor2Vec. BUT (a) RSS single modality (no IMU), (b) no continuous-time / no temporal fusion, (c) static imputation not modality/instant dropout, (d) no async multi-rate. Our novelty over this line is putting an Anchor2Vec-style WiFi token INTO a single continuous-time set-transformer alongside IMU tokens with async Delta-t + dropout robustness.
- **key_quote:** "we present all-embracing Transformers (AaTs) that are capable of deftly manipulating attention mechanism for Received Signal Strength (RSS) fingerprints." [d4577fc7] ; "we interpret/consolidate an arbitrary sequence of raw anchors into a sequence of k = 64 tokens constituted by d = 128 distinct attributes each." [d4577fc7]

### COMPETITOR — Nasir et al. 2024 HyTra [be87a391]
- **Citation:** Nasir, M., Esguerra, K., Faye, I., Tang, T. B., Yahya, M., Tumian, A., & Ho, E. T. W. (2024). HyTra: Hyperclass Transformer for WiFi Fingerprinting-based Indoor Localization. Transactions on Energy Systems and Engineering Applications, 5(1):542.
- **Method:** Encoder-only transformer; WAPs as learnable embeddings; "sentence" = fixed-order WAP RSS. HyTra-HF adds hierarchical coupling (building->floor->room) via attention-filtered values.
- **Modalities:** WiFi RSS only.
- **Time handling:** Fixed-order discrete sequence; no temporal/async/Delta-t.
- **Robustness:** Static imputation (+100 dBm -> -110 dBm). No modality dropout.
- **Headline (quoted):** "HyTra-HF outperforms existing deep learning solutions by obtaining 96.7% accuracy for the floor classification task on the UJIIndoorloc dataset." [be87a391] (classification accuracy, not metric MAE.)
- **XSESS:** Cross-TIME via UJIIndoorLoc 4-month split: "training and testing ... were generated four months apart to ensure data independence." [be87a391]
- **diff_vs_ours:** Transformer + cross-time split, BUT single modality RSS, floor classification focus, no continuous-time, no temporal fusion, no modality/instant dropout, no IMU.
- **key_quote:** "We propose the hyper-class Transformer (HyTra), an encoder-only Transformer neural network which learns the relative positions of wireless access points (WAPs) through multiple learnable embeddings." [be87a391]

### COMPETITOR — Turgut & Kakisim 2024 (explainable hybrid CNN-LSTM) [0d98906d]
- **Citation:** Turgut, Z., & Kakisim, A. G. (2024). An explainable hybrid deep learning architecture for WiFi-based indoor localization in IoT. Future Generation Computer Systems 151, 196-213.
- **Method:** Sparse Autoencoder + Particle Filter preprocessing; CNN and LSTM applied simultaneously (deep feature fusion); LIME/SHAP for explainability. Localization framed as classification.
- **Modalities:** WiFi RSSI only.
- **Time handling:** Discrete LSTM time-steps; particle filter cleans values but NO continuous-time/async.
- **Robustness:** Particle filter regulates zero/stale RSSI from prior high values (stale-value cleaning, not modality dropout).
- **Headline (quoted, accuracy):** "98.52 in the HALIC dataset, 98.42 in the RFKON dataset, and 95.33 in the UJIIndoorLoc dataset." [0d98906d] (classification accuracy %.)
- **XSESS:** Train-ratio subset generalization; no explicit cross-day/device/env transfer protocol.
- **diff_vs_ours:** No attention/transformer (CNN-LSTM), single modality, classification not regression, no continuous-time, no modality dropout, no cross-session.
- **key_quote:** "the method applies effective filtering and dimension scaling on the data ... using particle filter and sparse autoencoder." [0d98906d]

### COMPETITOR — Zhou et al. 2025 (conformal prediction for IP) [07baf76d]
- **Citation:** Zhou, Z., Peng, H., & Long, H. (2025). Conformal Prediction for Indoor Positioning with Correctness Coverage Guarantees. (Southwest Univ. / CQUPT.)
- **Method:** Conformal prediction (CP) over deep models: converts model uncertainty into a non-conformity score, builds prediction sets with statistical coverage; conformal risk control for path navigation (FDR/FNR); conformal p-value framework.
- **Modalities:** WiFi RSSI.
- **Time handling:** None (no temporal/async).
- **Robustness:** CP flags high-error samples; no modality dropout.
- **Headline (quoted):** "approximately 100% on the training dataset and 85% on the testing dataset." [07baf76d] (classification accuracy of the base model; CP coverage approaches target.)
- **XSESS:** Evaluated on UJIIndoorLoc; conformal risk control to bound errors under environmental dynamics; not a transfer protocol.
- **diff_vs_ours:** Relates to OUR uncertainty module (ConformalPosition split-conformal at alpha=0.1). Both rely on EXCHANGEABILITY — same caveat we document. CP here applied to classification base models (MobileNet/VGG/ResNet/EfficientNet), not to a continuous-time WiFi+IMU set-transformer regression. No fusion, no async, no IMU.
- **key_quote:** "CP transforms the uncertainty of the model into a non-conformity score, constructs prediction sets to ensure correctness coverage, and provides statistical guarantees." [07baf76d] ; "CP offers model-agnostic and distribution-free guarantees under the mild assumption of data exchangeability." [07baf76d]

### CONTEXT / SURVEYS (one-line capsules)
- **Feng et al. 2022** [072953c0] — "A survey of deep learning approaches for WiFi-based indoor positioning" (J. Information and Telecommunication 6(2):163-216). Key note: "the CSI is more stable than the RSS on a timescale but has strong specificity over space." [072953c0]
- **He & Chan 2016** [66be4066] — "Wi-Fi Fingerprint-Based Indoor Positioning: Recent Advances and Comparisons" (IEEE Comm. Surveys & Tutorials 18(1)). Key note: "As Wi-Fi signals may change due to environmental change ... another costly site survey may be needed to keep the fingerprints in the database up-to-date." [66be4066] Also reviews fusing inertial sensors with WiFi.
- **Martin-Frechina et al. 2025** [d3df2e7e] — "From Fingerprinting to Advanced Machine Learning: A Systematic Review of Wi-Fi and BLE-Based Indoor Positioning Systems" (PRISMA, 2020-2024). Open challenges: "scalability in large dynamic environments, high calibration costs, device heterogeneity, and the absence of standardized open datasets." Calls for "lightweight, adaptive DL models" and DL-based multimetric fusion. [d3df2e7e]

---

## B. COMPETITOR RUBRIC (gap matrix)

MODS = modalities fused | ATT = attention/transformer | CT = continuous-time/async real-valued Delta-t w/o resample or ODE | ROB = explicit missing/stale-modality robustness | XSESS = cross-session/day/device/env real-world eval | UNIFIED = single fusion block vs branches

| bibkey | MODS | ATT | CT | ROB | XSESS | UNIFIED |
|---|---|---|---|---|---|---|
| aristorenas2025set | WiFi RSSI only | yes | no (single scan, no time) | partial (missing-AP via set, no modality dropout) | no chronological; cross-building only | unified (set over APs, single modality) |
| tiku2022anvil | WiFi RSSI only | yes (multi-head attn) | no (static match) | partial (AP-dropout augmentation, -100dB impute) | yes: cross-device | unified (single modality) |
| zhang2022tips | WiFi CSI + DoA | yes (GPT decoder) | no (discrete "words") | no | no (SNR robustness only) | hybrid (separate models per modality combo) |
| zhang2023aarescnn | WiFi CSI + IMU | partial (attn-augmented CNN) | no (discrete T steps) | no (inference); modular PnP | yes: cross-environment tracking | branches (positioning net + separate tracking net) |
| ott2024radiofm | 5G CIR only | yes (transformer) | no (fixed-len seq) | partial (mask-reconstruct pretext) | yes: cross-site (TF-C-PT) | unified (single modality) |
| bhatia2025locaris | WiFi FTM + RSSI | yes (decoder-only LLM) | no (token-per-reading, not Delta-t) | yes (any-modality-available, no placeholders) | yes: cross-environment few-shot | unified (token stream) |
| abdullah2025ris | CSI + RSS + RIS + geom | yes (multi-head transformer) | no (synchronous tokens) | no (static ablation only) | no (sim; SNR sweep) | unified ([CLS]+8 tokens) |
| nguyen2024aat | WiFi RSS only | yes (transformer + Anchor2Vec) | no (static fingerprint) | no (100dB impute) | partial: env-heterogeneity / unseen split | unified (single modality) |
| nasir2024hytra | WiFi RSS only | yes (encoder-only transformer) | no (fixed-order seq) | no (static impute) | yes: cross-time 4-month UJI split | unified (single modality) |
| turgut2024xai | WiFi RSSI only | no (CNN-LSTM hybrid) | no (discrete LSTM) | partial (particle filter cleans stale RSSI) | no (train-ratio subsets only) | hybrid (CNN+LSTM feature fusion) |
| zhou2025conformal | WiFi RSSI only | no (CNN backbones + CP) | no | no | no (UJI; risk control under dynamics) | n/a (CP wrapper on classifier) |

Justify quotes (source_id):
- aristorenas2025set: "each RSSI set is processed individually (batch size = 1), avoiding the need for explicit padding or masking." [f6417660]
- tiku2022anvil: "resilient to device heterogeneity" + "AP that ... is not observed ... assumed to be -100dB" [c114e66e]
- zhang2022tips: "we adopt the transformer neural network ... Each route is then considered a sentence ... position ... a word" [07285e43]
- zhang2023aarescnn: "AAResCNN for CSI-based fingerprinting" + IMU "compatible with arbitrary environment and IMUs" [52b5f512]
- ott2024radiofm: "pre-trains a general transformer (TF) ... pretext task to randomly mask and discard input information" + cross-site "TF-C-PT" [5aa39fa5]
- bhatia2025locaris: "whichever modality is available (FTM-only or RSSI-only) without requiring placeholders" + "true cross-environment transfer" [c4f1526c]
- abdullah2025ris: "eight tokens: five derived from CSI, one from RSS, one from RIS phase settings, and a special [CLS] token" [1e099b7b]
- nguyen2024aat: "all-embracing Transformers (AaTs) ... for Received Signal Strength (RSS) fingerprints" + "missing anchors were filled with 100 dB" [d4577fc7]
- nasir2024hytra: "encoder-only Transformer ... learns the relative positions of wireless access points" + "four months apart to ensure data independence" [be87a391]
- turgut2024xai: "Convolutional Neural Network (CNN) and Long-Short-Term Memory (LSTM) simultaneously" + particle filter [0d98906d]
- zhou2025conformal: "CP ... constructs prediction sets to ensure correctness coverage" + "guarantees under the mild assumption of data exchangeability" [07baf76d]

---

## C. ATOMIC CLAIMS

1. The de-facto WiFi fingerprint benchmark contains 21,049 RSSI samples over 520 WAPs in 3 buildings, with validation taken 4 months after training to enforce independence — supports (iii) cross-session motivation and our use of UJIIndoorLoc. [46b80222] Quote: "Dataset independence has been assured by taking Validation (or testing) samples 4 months after Training ones."
2. The canonical UJIIndoorLoc 1NN baseline is 7.9 m positioning error — the number our WiFi half is compared against. [46b80222] Quote: "the error is, in average, 7.9 meters when the fingerprint has been located in the correct building and floor."
3. UJIIndoorLoc is WiFi-RSSI only with no inertial data — so the standard WiFi benchmark cannot itself demonstrate WiFi+IMU async fusion; supports our (i)+(ii) gap. [46b80222] Quote: "this information represents the 98% of the data given in each record (520 vector positions out of 529) as a 520-element vector of integer values. These values represent the RSSI levels."
4. The closest permutation-invariant set-transformer over RSSI is single-modality and processes each scan independently with no temporal/cross-time fusion — supports (i) and (ii). [f6417660] Quote: "each RSSI set is processed individually (batch size = 1), avoiding the need for explicit padding or masking."
5. That same set-transformer is restricted to WiFi RSSI and lists multimodal (BLE/WiFi/other) fusion only as future work — supports (ii) multimodal unified gap. [f6417660] Quote: "In this work, we evaluate the performance of permutation-invariant neural architectures ... purely from abundant Wi-Fi access points."
6. The in-domain WiFi transformer line (All-embracing Transformers) uses an Anchor2Vec tokenizer (k=64 tokens, d=128) on static RSS vectors with no temporal modeling and 100 dB imputation for missing anchors — directly relates to our WiFiNet encoder but lacks continuous-time and IMU. [d4577fc7] Quote: "we interpret/consolidate an arbitrary sequence of raw anchors into a sequence of k = 64 tokens constituted by d = 128 distinct attributes each." ; "Such missing anchors were filled with 100 dB for all these databases."
7. eAaT+ reports 8.16 m MAE on UJIIndoorLoc (vs RADAR 9.21 m, Weighted-KNN 9.33 m, CNNLoc 11.78 m) — the in-domain WiFi-transformer numbers we situate against. [d4577fc7] Quote: "eAaT+ ... 8.16" (Table 3, Mean absolute error (m), UJIIndoorLoc).
8. WiFi transformer competitors that DO use attention treat time as discrete fixed-step sequences ("sentence"/"word"), never real-valued Delta-t — supports (i) continuous-time novelty. [07285e43] Quote: "Each route is then considered a sentence, whereas the position along the route is treated as a word in terms of natural language processing." Also [be87a391]: "conceptualizing a fixed order sequence of WAP measurements as a sentence."
9. The few WiFi+IMU papers in the group keep modalities in SEPARATE stages/branches (CSI positioning net then IMU plug-and-play tracking), not one unified fusion block — supports (ii) unified-block novelty. [52b5f512] Quote: "we adopt the idea of PnP to incorporate the IMU measurements into the tracking system without retraining."
10. The closest in-group missing-modality robustness is "provide whichever modality is available without placeholders" (Locaris) — but it fuses two WiFi modalities (FTM+RSSI), not WiFi+IMU, and uses token-per-reading not continuous-time. Supports (iii) being adjacent but not matched. [c4f1526c] Quote: "in the ablation settings, we simply provide whichever modality is available (FTM-only or RSSI-only) without requiring placeholders."
11. Several competitors only do STATIC modality ablation (drop a modality and remeasure) rather than train-time stochastic modality/instant dropout for runtime robustness — supports (iii). [1e099b7b] Quote: "Systematically removing individual modalities leads to notable degradations: excluding RSS increases the error to 0.48 meters, while omitting CSI and RIS phase ... results in errors of 0.57 meters and 0.76 meters."
12. Surveys confirm WiFi fingerprints drift with environment/time, forcing recalibration — the core problem our cross-session robustness targets. [66be4066] Quote: "As Wi-Fi signals may change due to environmental change ... another costly site survey may be needed to keep the fingerprints in the database up-to-date."
13. The 2025 systematic review names DL-based multimetric fusion and lightweight adaptive models as open frontiers — positions our unified WiFi+IMU set-transformer as on-trend and the gap as real. [d3df2e7e] Quote: "Research into hybrid and multimetric systems that combine RSSI, CSI, RTT, and AoA is a promising frontier. The challenge lies in designing sophisticated fusion algorithms ... that can intelligently weight the contribution of each metric."
14. Conformal prediction for indoor positioning exists but is applied as a wrapper on classification backbones and relies on exchangeability — same guarantee/caveat as our ConformalPosition module; no async fusion. [07baf76d] Quote: "CP offers model-agnostic and distribution-free guarantees under the mild assumption of data exchangeability."
15. GROUNDED GAP: across the entire WiFi notebook, no single paper combines continuous-time Delta-t encoding + a unified permutation-invariant set-transformer over (modality,time) + modality/instant-dropout robustness with cross-session eval; in fact none satisfies even two of these three on WiFi+IMU. Supports the conjunction-of-(i)+(ii)+(iii) claim. [f6417660] (closest is Aristorenas, single-modality/single-time). Quote: "We propose a permutation-invariant neural architecture for indoor localization using RSSI scans from Wi-Fi access points." (sole satisfied criterion is partial perm-invariance, single modality, no time, no dropout.)

---

## D. GROUP GAP SYNTHESIS

The WiFi-fingerprinting group spans classical RSSI matching (RADAR kNN, Horus probabilistic), CSI deep models (DeepFi), CNN baselines on the canonical UJIIndoorLoc benchmark (CNNLoc, 7.9 m 1NN reference), and a fast-growing wave of attention/transformer methods (Tiku ANVIL, Zhang TIPS, Ott Radio Foundation Models, Bhatia Locaris, Abdullah RIS, Nguyen All-embracing/Anchor2Vec, Nasir HyTra) plus a permutation-invariant Set-Transformer (Aristorenas) and a conformal-prediction uncertainty paper (Zhou). The clear trend is toward transformers and self-attention over WiFi tokens for device-/environment-invariance and few-shot cross-environment transfer, with surveys (Feng, He & Chan, Martin-Frechina) flagging temporal/environmental signal drift and DL-based multimetric fusion as the open frontier. However, the entire group treats time as discrete fixed-step sequences or static single scans: not one paper uses a learned continuous-time encoding of real-valued elapsed Delta-t for asynchronous, multi-rate observations without resampling or an ODE (gap vs our (i)). The single permutation-invariant set-transformer that exists (Aristorenas) is WiFi-RSSI-only over a single scan and is even out-performed by a plain LSTM, while multimodal transformers either stay within RF modalities (CSI+DoA, CSI+RSS+RIS, FTM+RSSI) or push IMU into a separate plug-and-play tracking stage — none performs cross-modal AND cross-time fusion in one unified block over (modality,time) tokens that includes inertial data (gap vs our (ii)). Robustness in this group is mostly static imputation or one-off modality-ablation; only Locaris approaches train-time modality flexibility, and the cross-session evidence is fragmented (UJIIndoorLoc's 4-month split, cross-device in ANVIL, cross-site in Ott, cross-environment in Bhatia) with no paper pairing stochastic modality/instant-dropout robustness against missing/STALE sensors with real-world cross-session generalization on WiFi+IMU (gap vs our (iii)). A direct NotebookLM check confirms no source satisfies even two of our three criteria simultaneously on WiFi+IMU, establishing that our contribution is precisely the CONJUNCTION of continuous-time Delta-t encoding, a single unified permutation-invariant set-transformer, and async/stale-modality robustness applied to WiFi+IMU localization.
