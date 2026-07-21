# Related-Work Reference Master List — ICINCO 2026

**Paper:** "Async-Robust Multi-Modal Indoor Localization via a Continuous-Time
Set-Transformer" — a continuous-time set-transformer fusing **asynchronous WiFi
RSSI + IMU** for indoor (x, y) localization.

**Three contributions this list is organised against:**
- (i) learned **continuous-time Δt encoding** added per token (no resampling, no ODE solver),
- (ii) a **single permutation-invariant self-attention block** doing cross-modal AND cross-time fusion (no per-modality branches),
- (iii) **async robustness via modality-/instant-dropout** (graceful degradation under missing/stale sensors), evaluated cross-session.

**Source reports consolidated** (all in `paper-workspace/deep-search/`):
1. Architecture foundations (set + attention) — `[ARCH]`
2. Continuous-time / irregular sampling / time encoding — `[CT]`
3. WiFi localization benchmarks & baselines we use — `[WIFI]`
4. Learned inertial navigation — `[IMU]`
5. Closest competitors / novelty defense (HIGHEST PRIORITY) — `[COMP]`
6. Recent transformer / deep indoor-localization frontier (recency sweep) — `[REC]`
Plus venue-local cross-reference: `paper-workspace/icinco-2024-relevant.md` — `[ICINCO]`.

**STATUS tags:** `already-in-zotero` · `new-verified` · `needs-verification` · `conflict` · `likely-unreliable`.
"Cross-agent agreement" (the same ref verified in ≥2 reports) is noted as a confidence signal.

---

## ① WiFi fingerprinting

| Key | Authors, Year | Title | Venue / Publisher | DOI / arXiv | Status | Reports | Relation to our work |
|---|---|---|---|---|---|---|---|
| `torressospedra2014ujiindoorloc` | Torres-Sospedra, Montoliu, Martínez-Usó, Avariento, Arnau, Benedito-Bordonau, Huerta, 2014 | UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems | IPIN 2014, IEEE, pp. 261–270 | 10.1109/IPIN.2014.7275492 | new-verified | WIFI | Primary public WiFi-fingerprinting benchmark; our WiFi baselines are evaluated on it. |
| `bahl2000radar` | Bahl, Padmanabhan, 2000 | RADAR: An In-Building RF-Based User Location and Tracking System | IEEE INFOCOM 2000, vol. 2, pp. 775–784 | 10.1109/INFCOM.2000.832252 | new-verified | WIFI | Seminal RSSI-kNN fingerprinting; conceptual ancestor of our WiFi baseline (use as the kNN method citation for the wlan_localization repo). |
| `song2019cnnloc` | Song, Fan, Xiang, Ye, Liu, Wang, He, Yang, Fang, 2019 | A Novel Convolutional Neural Network Based Indoor Localization Framework with WiFi Fingerprinting (CNNLoc, journal) | IEEE Access, vol. 7, pp. 110698–110709 | 10.1109/ACCESS.2019.2933921 | conflict (two papers share "CNNLoc"; this is the recommended canonical) | WIFI, CT | Deep WiFi-only fingerprinting baseline (SAE + 1D-CNN); fixed-length 520-d RSS vectors vs our time-stamped tokens. |
| `song2019cnnlocconf` | Song, Fan, He, Xiang, Ye, Huang, Fang, Chen, Qin, Wang, 2019 | CNNLoc: Deep-Learning Based Indoor Localization with WiFi Fingerprinting (conference, bears literal name) | IEEE SmartWorld/UIC/ATC/SCALCOM/IOP/SCI 2019, pp. 589–595 | 10.1109/SmartWorld-UIC-ATC-SCALCOM-IOP-SCI.2019.00139 | conflict (alt of above; cite only if invoking literal "CNNLoc" name) | WIFI, CT | Same SAE + 1D-CNN method; conference version. |
| `naribole_wlanloc` (repo, not a paper) | Naribole (GitHub) | wlan_localization: A Machine Learning Approach to WLAN Fingerprinting based Localization | github.com/sharan-naribole/wlan_localization | — | needs-verification (cite as software; NO method paper — attribute method to RADAR) | WIFI | Open-source cascaded Random-Forest + weighted-kNN baseline on UJIIndoorLoc; cite repo URL only. |
| `youssef2005horus` | Youssef, Agrawala, 2005 | The Horus WLAN Location Determination System | ACM MobiSys 2005, pp. 205–218 | 10.1145/1067170.1067193 | new-verified | WIFI | Classic probabilistic (vs deterministic kNN) WiFi fingerprinting baseline. |
| `wang2015deepfi` | Wang, Gao, Mao, Pandey, 2015 | DeepFi: Deep Learning for Indoor Fingerprinting Using Channel State Information | IEEE WCNC 2015, pp. 1666–1671 | 10.1109/WCNC.2015.7127718 | new-verified | WIFI | Early deep-learning WiFi (CSI) fingerprinting; learned features for localization. |
| `he2016wifisurvey` | He, Chan, 2016 | Wi-Fi Fingerprint-Based Indoor Positioning: Recent Advances and Comparisons | IEEE Communications Surveys & Tutorials, vol. 18(1), pp. 466–490 | 10.1109/COMST.2015.2464084 | new-verified | WIFI | Landscape survey motivating sensor fusion. |
| `nasir2024hytra` | Nasir, Esguerra, Faye, Tang, Yahya, Tumian, Ho, 2024 | HyTra: Hyperclass Transformer for WiFi Fingerprinting-based Indoor Localization | Trans. Energy Systems and Engineering Applications (TESEA), vol. 5(1), pp. 1–24 | 10.32397/tesea.vol5.n1.542 | needs-verification (small Scopus/DOAJ journal; preprint precursor exists; 96.7% figure is HyTra-HF not base) | WIFI | Self-attention over WiFi RSS, but WiFi-only, no Δt, no IMU. |
| `nguyen2024allembracing` | Nguyen, Le, Havinga, 2024 | Seeing the world from its words: All-embracing Transformers for fingerprint-based indoor localization | Pervasive and Mobile Computing (Elsevier), vol. 100, art. 101912 | 10.1016/j.pmcj.2024.101912 | new-verified | REC | Attention over WiFi RSS; WiFi-only, no continuous time, no async fusion. (Same group as iMoT — double-blind caution.) |
| `aristorenas2025permutation` | Aristorenas, 2025 | Permutation-Invariant Transformer Neural Architectures for Set-Based Indoor Localization Using Learned RSSI Embeddings | arXiv preprint (Stanford), single author | arXiv:2506.00656 | needs-verification (PREPRINT ONLY, no peer-reviewed venue) | ARCH, REC | Closest WiFi-domain analogue: Set Transformer over (BSSID,RSSI) sets — but WiFi-only, no IMU, no Δt, no modality dropout; reports LSTM beating the Set Transformer. |

**Already in Zotero (WiFi group — do NOT list as new):** Feng 2022 (DL-for-WiFi survey), Martin-Frechina 2025 (Wi-Fi/BLE review), Turgut 2024, Zhang 2023 (CSI attention-ResCNN), Tiku 2022 (multi-head attention), Ai 2022 (DRVAT), Zhang 2022 (TIPS), Ott 2024 (Radio Foundation Models), Bhatia 2025 (decoder-only transformer), SwinULoc 2026, Abdullah 2025 (RIS transformer), "DL indoor positioning + uncertainty", "Conformal Prediction for Indoor Positioning".

---

## ② Inertial / IMU

| Key | Authors, Year | Title | Venue / Publisher | DOI / arXiv | Status | Reports | Relation to our work |
|---|---|---|---|---|---|---|---|
| `chen2018ionet` | Chen, (Chris Xiaoxuan) Lu, Markham, Trigoni, 2018 | IONet: Learning to Cure the Curse of Drift in Inertial Odometry | AAAI-18, vol. 32(1), pp. 6468–6476 | 10.1609/aaai.v32i1.12102 (arXiv:1802.02209) | new-verified (3-report agreement) | ARCH, COMP, CT, IMU, REC | Seminal deep inertial odometry on fixed-rate windows; our IMU stream descends from this but fuses async WiFi. |
| `yan2020ronin` | Herath, Yan, Furukawa, 2020 | RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods | IEEE ICRA 2020, pp. 3146–3152 | 10.1109/ICRA40945.2020.9196860 (arXiv:1905.12853) | already-in-zotero / conflict (author order + arXiv-vs-ICRA — canonical = ICRA 2020, IEEE order Herath,Yan,Furukawa) | ARCH, COMP, IMU, REC, WIFI | De-facto IMU baseline (~4.63M params vs our ~0.05M); we differ by async WiFi+IMU continuous-time fusion. |
| `yan2018ridi` | Yan, Shan, Furukawa, 2018 | RIDI: Robust IMU Double Integration | ECCV 2018, LNCS 11217, Springer | 10.1007/978-3-030-01261-8_38 (arXiv:1712.09004) | new-verified | WIFI, IMU | Early regress-velocity IMU paradigm; IMU-only fixed windows. (Page range conflict: 621–636 vs 641–656 — verify.) |
| `liu2020tlio` | Liu, Caruso, Ilg, Dong, Mourikis, Daniilidis, Kumar, Engel, 2020 | TLIO: Tight Learned Inertial Odometry | IEEE RA-L, vol. 5(4), pp. 5653–5660 | 10.1109/LRA.2020.3007421 (arXiv:2007.01867) | new-verified (corrected author: Eddy Ilg, not "Eric") | WIFI, IMU | Learned displacement + EKF; we fuse end-to-end with attention rather than feed an EKF. |
| `sun2021idol` | Sun, Melamed, Kitani, 2021 | IDOL: Inertial Deep Orientation-Estimation and Localization | AAAI 2021, vol. 35(7), pp. 6128–6137 | 10.1609/aaai.v35i7.16763 (arXiv:2102.04024) | new-verified | IMU | RNN+EKF two-stage IMU-only; no WiFi. |
| `chen2021rninvio` | Chen, Wang, Xu, Xie, Bao, Zhang, 2021 | RNIN-VIO: Robust Neural Inertial Navigation Aided Visual-Inertial Odometry in Challenging Scenes | IEEE ISMAR 2021, pp. 275–283 | 10.1109/ISMAR52148.2021.00044 | new-verified | IMU | EKF fusing neural inertial net with VIO; visual (not WiFi) fusion. |
| `rao2022ctin` | Rao, Kazemi, Ding, Shila, Tucker, Wang, 2022 | CTIN: Robust Contextual Transformer Network for Inertial Navigation | AAAI 2022, vol. 36(5), pp. 5413–5421 | 10.1609/aaai.v36i5.20479 (arXiv:2112.02143) | new-verified (2-report agreement) | IMU, REC | Transformer over IMU windows, IMU-only, NOT continuous-time; closest transformer-inertial predecessor. |
| `herath2022niloc` | Herath, Caruso, Liu, Chen, Furukawa, 2022 | Neural Inertial Localization | IEEE/CVF CVPR 2022, pp. 6604–6613 | arXiv:2203.15851 | new-verified | IMU | Transformer+TCN inertial *localization*; IMU-only, discrete-time. |
| `zeinali2024imunet` | Zeinali, Zandizari, Chang, 2024 | IMUNet: Efficient Regression Architecture for Inertial IMU Navigation and Positioning | IEEE TIM, vol. 73, art. 2513413 | IEEE doc 10480886 (arXiv:2208.00068) | new-verified | IMU | Edge-efficient IMU net; efficiency story analogous to our small encoder; no WiFi. |
| `nguyen2025imot` | Nguyen, (Tran,) Le, Havinga, 2025 | iMoT: Inertial Motion Transformer for Inertial Navigation | AAAI 2025, vol. 39(6), pp. 6209–6217 | 10.1609/aaai.v39i6.32664 (arXiv:2412.12190) | new-verified (3-report agreement) | COMP, IMU, REC | Cross-modal transformer w/ Adaptive Positional Encoding + query "motion particles"; IMU-only (acc+gyro), APE ≠ real Δt, no WiFi/modality dropout. Closest conceptual neighbor. |
| `jayanth2025eqnio` | Jayanth, Xu, Wang, Chatzipantazis, Daniilidis, Gehrig, 2025 | EqNIO: Subequivariant Neural Inertial Odometry | ICLR 2025 (OpenReview C8jXEugWkq) | arXiv:2408.06321 | new-verified | IMU | Current IMU-only SOTA via equivariance; orthogonal to our multimodal task (cite as IMU-only SOTA). |
| `brotchie2023riot` | Brotchie, Li, Greentree, Kealy, 2023 | RIOT: Recursive Inertial Odometry Transformer for Localisation from Low-Cost IMU Measurements | Sensors (MDPI), vol. 23(6), art. 3217 | 10.3390/s23063217 | new-verified | REC | Self-attention IMU odometry; IMU-only, no WiFi, no continuous time. |
| `chen2018oxiod` | Chen, Zhao, Lu, Wang, Markham, Trigoni, 2018 | OxIOD: The Dataset for Deep Inertial Odometry | arXiv preprint (CoRR) | arXiv:1809.07491 | needs-verification (PREPRINT; peer-reviewed companion = arXiv:2001.04061, L-IONet) | IMU | IMU-odometry benchmark dataset. |
| `zheng2024neurit` | Zheng, Ji, Pan, Zhang, Wu, 2024 | NeurIT: Pushing the Limit of Neural Inertial Tracking for Indoor Robotic IoT | arXiv preprint | arXiv:2404.08939 | likely-unreliable for citation (PREPRINT only; 48.21% gain is vs UniTS not RoNIN) | IMU | TF-BRT (RNN+Transformer) inertial tracking; magnetometer reliance, no WiFi. Cite only as preprint if needed. |

**Already in Zotero (Inertial group):** RoNIN (Yan/Herath/Furukawa) [also above], Cohen & Klein 2024 (inertial-DL survey).

---

## ③ Multimodal fusion (for localization)

| Key | Authors, Year | Title | Venue / Publisher | DOI / arXiv | Status | Reports | Relation to our work |
|---|---|---|---|---|---|---|---|
| `yu2022multimodal` | Yu, Wang, Koike-Akino, Orlik, 2022 | Multi-Modal Recurrent Fusion for Indoor Localization | IEEE ICASSP 2022 | arXiv:2203.00510 (Xplore doc 9746071) | already-in-zotero / conflict (held title "A Multi-Modal Recurrent Fusion-based Indoor Localization" is WRONG — correct it) | COMP | Per-modality recurrent branches w/ uncertainty (WiFi+IMU+UWB); NOT a single block, not permutation-invariant. |
| `antsfeld2021deep` | Antsfeld, Chidlovskii, Sansano-Sansano, 2020/2021 | Deep Smartphone Sensors-WiFi Fusion for Indoor Positioning and Tracking | IPIN 2021, IEEE, pp. 1–8 (preprint arXiv:2011.10799, 2020) | arXiv:2011.10799 | already-in-zotero / needs-verification (cite IPIN 2021 published version, not the bare arXiv) | COMP, WIFI | WiFi+IMU async fusion via Kalman filter; two separate models, no attention, no Δt embedding. |
| `zhou2024wioekf` | Zhou, Wang, Gravina, Sun, 2024 | WIO-EKF: Extended Kalman Filtering-Based Wi-Fi and Inertial Odometry Fusion Method for Indoor Localization | IEEE IoT Journal, vol. 11(13), pp. 23592–23603 | 10.1109/JIOT.2024.3386889 | already-in-zotero (2-report agreement) | COMP, REC | Per-modality deep branches (CDAELoc + DbDIO) glued by EKF; no attention, no Δt, no modality dropout. First-author spelled "Peng Zhou" (one report wrote "Ping" — verify). |
| `yang2025wimu` | Yang, Xu, Song, Li, 2025 | WiMU: Real-time Indoor Localization via Wi-Fi/IMU Fusion with Minimal Site Survey | ACM IMWUT, vol. 9(4), art. 233, pp. 1–25 | 10.1145/3770667 | already-in-zotero (2-report agreement) | COMP, REC | Most recent WiFi+IMU competitor; system pipeline, not single attention block; no Δt, no modality dropout. |
| `zhang2021lstmfusion` | Zhang, Jia, Chen, Deng, Wang, Aghvami, 2021 | Indoor Localization Fusing WiFi With Smartphone Inertial Sensors Using LSTM Networks | IEEE IoT Journal, vol. 8(17), pp. 13608–13623 | 10.1109/JIOT.2021.3067515 | new-verified (issue no. 17 inferred — verify) | COMP | WiFi+PDR fused by single LSTM; fixed-timestep resampling, not permutation-invariant. |
| `wei2021mmloc` | Wei, Wei, Radu, 2021 | Sensor-Fusion for Smartphone Location Tracking Using Hybrid Multimodal Deep Neural Networks (MM-Loc) | Sensors (MDPI), vol. 21(22), art. 7488 | 10.3390/s21227488 | new-verified (author list reconstructed — verify Wei, Wei, Radu) | CT | WiFi+IMU at different rates via two per-modality branches + late concat — the multi-branch design we avoid. |
| `hua2023smartfps` | Hua, Zhuang, Yang, 2023 | SmartFPS: Neural network based wireless-inertial fusion positioning system | Frontiers in Neurorobotics, vol. 17, art. 1121623 | 10.3389/fnbot.2023.1121623 | new-verified | REC | Deep WiFi+inertial fusion w/ transfer learning for cross-session; not attention/set, no Δt, no async dropout. |
| `herath2021fusiondhl` | Herath, Irandoust, Chen, Qian, Kim, Furukawa, 2021 | Fusion-DHL: WiFi, IMU, and Floorplan Fusion for Dense History of Locations in Indoor Environments | IEEE ICRA 2021, pp. 5677–5683 | 10.1109/ICRA48506.2021.9561115 (arXiv:2105.08837) | new-verified | IMU(REC) | Closest exact-modality prior art (WiFi+IMU); optimization+CNN relying on floorplans, not end-to-end attention, not continuous-time. |
| `chen2015wifipdrkalman` | Chen, Zou, Jiang, Zhu, Soh, Xie, 2015 | Fusion of WiFi, Smartphone Sensors and Landmarks Using the Kalman Filter for Indoor Localization | Sensors (MDPI), vol. 15(1), pp. 715–732 | 10.3390/s150100715 | new-verified | WIFI | Canonical classical WiFi+PDR Kalman fusion; contrast to our learned fusion. |
| `lajoie2024peoplex` | Lajoie, Baghi, Herath, Hogan, Liu, Dudek, 2024 | PEOPLEx: PEdestrian Opportunistic Positioning LEveraging IMU, UWB, BLE and WiFi | IEEE ICC 2024 (Xplore doc 10622566) | arXiv:2311.18182 | needs-verification (DOI suffix/pages unconfirmed) | REC | Factor-graph opportunistic graceful degradation; classical optimization, not learned modality dropout. Closest localization analogue for contribution (iii). |
| `neverova2016moddrop` | Neverova, Wolf, Taylor, Nebout, 2016 | ModDrop: Adaptive Multi-Modal Gesture Recognition | IEEE TPAMI, vol. 38(8), pp. 1692–1706 | 10.1109/TPAMI.2015.2461544 (arXiv:1501.00102) | new-verified | REC | Canonical modality-dropout method (gesture, not localization); method precedent for contribution (iii). |

**Already in Zotero (Multimodal group):** Yu 2022 [above], Antsfeld 2020 [above], WIO-EKF 2024 (Zhou) [above], Liu 2025 (survey), Wang 2024 (robust multimodal multi-scale), WiMU 2025 (Yang) [above], Geneva 2018 (async multi-sensor), Silva 2023 (dataset), Abdalla 2026 (dataset), Lukasik 2024 (review), Wang & Ahmad 2025 (AMR review).

---

## ④ Attention / transformer / set

| Key | Authors, Year | Title | Venue / Publisher | DOI / arXiv | Status | Reports | Relation to our work |
|---|---|---|---|---|---|---|---|
| `vaswani2017attention` | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin, 2017 | Attention Is All You Need | NeurIPS/NIPS 2017, Curran, pp. 5998–6008 | arXiv:1706.03762 (**no DOI**) | new-verified (3-report agreement) | ARCH, COMP, WIFI | PILLAR. Our single self-attention fusion block descends directly from multi-head attention. |
| `lee2019set` | Lee, Lee, Kim, Kosiorek, Choi, Teh, 2019 | Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks | ICML 2019, PMLR vol. 97, pp. 3744–3753 | arXiv:1810.00825 (**no DOI** per DBLP) | new-verified (5-report agreement) | ARCH, COMP, CT, IMU, REC, WIFI | PILLAR. Architectural backbone: permutation invariance over the (modality,time) observation set. |
| `zaheer2017deepsets` | Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov, Smola, 2017 | Deep Sets | NeurIPS/NIPS 2017, Curran, pp. 3391–3401 | arXiv:1703.06114 (**no Crossref DOI**) | new-verified (2-report agreement) | ARCH, WIFI | Theory of permutation-invariant set functions; justifies treating async observations as an unordered set. |
| `jaegle2021perceiver` | Jaegle, Gimeno, Brock, Zisserman, Vinyals, Carreira, 2021 | Perceiver: General Perception with Iterative Attention | ICML 2021, PMLR vol. 139, pp. 4651–4664 | arXiv:2103.03206 (**no DOI**) | new-verified | ARCH | Single attention stack fusing many modalities w/o per-modality branches; lacks Δt + async dropout. |
| `jaegle2022perceiverio` | Jaegle, Borgeaud, Alayrac, Doersch, et al., 2022 | Perceiver IO: A General Architecture for Structured Inputs & Outputs | ICLR 2022 | arXiv:2107.14795 (**no DOI**) | new-verified (note: distinct from Perceiver above; one report listed only IO) | WIFI | General multimodal attention; no continuous-time Δt for async streams. |
| `kaygusuz2022aftvo` | Kaygusuz, Mendez, Bowden, 2022 | AFT-VO: Asynchronous Fusion Transformers for Multi-View Visual Odometry Estimation | IEEE/RSJ IROS 2022 | arXiv:2206.12946 | already-in-zotero | COMP | Async transformer w/ continuous-timestamp encoding (Discretiser) — but multi-camera VO, per-source branches; NOT WiFi. |
| `cohen2024akit` | Cohen, Klein, 2024 | A-KIT: Adaptive Kalman-Informed Transformer | arXiv preprint (v1 Jan 2024, v2 Mar 2025) | arXiv:2401.09987 | already-in-zotero / needs-verification (PREPRINT only — do not assign a venue) | COMP | Set-transformer that tunes EKF noise (INS/DVL, AUV); set-transformer is EKF-auxiliary, not end-to-end fusion. |
| `diazguerra2023pirnn` | Diaz-Guerra, Politis, Miguel, Beltran, Virtanen, 2023 | Permutation Invariant Recurrent Neural Networks for Sound Source Tracking | Forum Acusticum 2023 (EAA) | 10.61782/fa.2023.1132 (arXiv:2306.08510) | new-verified (disambiguation: NOT the TASLP icosahedral-CNN paper) | ARCH | Permutation invariance over tracked sources; tracking analogue, single-input. Optional. |

**Already in Zotero (Attention/set group):** AFT-VO 2022 (Kaygusuz) [above], A-KIT 2024 (Cohen & Klein) [above], EffLoc 2024 (Xiao), Lin & Evans 2025 (place recognition).

---

## ⑤ Continuous-time / async / irregularly-sampled

| Key | Authors, Year | Title | Venue / Publisher | DOI / arXiv | Status | Reports | Relation to our work |
|---|---|---|---|---|---|---|---|
| `shukla2021mtan` | Shukla, Marlin, 2021 | Multi-Time Attention Networks for Irregularly Sampled Time Series (mTAN) | ICLR 2021 (OpenReview 4c0J6lwQ4_) | arXiv:2101.10318 (**no DOI**) | new-verified (5-report agreement) | ARCH, COMP, CT, REC, WIFI | PILLAR. Learned continuous-time embedding + time-attention; but interpolates to fixed reference points, single-modality. |
| `chen2018neuralode` | Chen, Rubanova, Bettencourt, Duvenaud, 2018 | Neural Ordinary Differential Equations | NeurIPS 2018, Curran, pp. ~6572–6583 (Best Paper) | arXiv:1806.07366 (**no DOI**) | new-verified (2-report agreement) / conflict (page range: 6571–6583 vs 6572–6583) | CT, COMP, WIFI | PILLAR. Continuous-time via ODE solver; we use no solver, Δt is additive. |
| `rubanova2019latentode` | Rubanova, Chen, Duvenaud, 2019 | Latent ODEs for Irregularly-Sampled Time Series | NeurIPS 2019, Curran | arXiv:1907.03907 (**no DOI**) | new-verified (2-report agreement; pages not consistently reported) | CT, WIFI | PILLAR. ODE-RNN/Latent-ODE for irregular Δt; we discard recurrence + ODE solves. |
| `kazemi2019time2vec` | Kazemi, Goel, Eghbali, Ramanan, Sahota, Thakur, Wu, Smyth, Poupart, Brubaker, 2019 | Time2Vec: Learning a Vector Representation of Time | arXiv preprint (NO peer-reviewed venue) | arXiv:1907.05321 | needs-verification (PREPRINT ONLY — most likely metadata point a reviewer scrutinises) | CT, WIFI | PILLAR / direct ancestor of our learned-sinusoid Δt encoding. Cite as @misc. |
| `kidger2020neuralcde` | Kidger, Morrill, Foster, Lyons, 2020 | Neural Controlled Differential Equations for Irregular Time Series | NeurIPS 2020 (Spotlight) | arXiv:2005.08926 (**no DOI**) | new-verified (2-report agreement) | ARCH, COMP, CT | Continuous control path + CDE solver for irregular series; we use neither interpolation nor solver. |
| `horn2020seft` | Horn, Moor, Bock, Rieck, Borgwardt, 2020 | Set Functions for Time Series (SeFT) | ICML 2020, PMLR vol. 119, pp. 4353–4363 | arXiv:1909.12064 (**no DOI**) | new-verified (2-report agreement) | ARCH, CT | Very close: observations-as-time-stamped-set, no resampling, explicitly async; classification not WiFi/IMU regression. |
| `chen2023contiformer` | Chen, Ren, Wang, Fang, Sun, Li, 2023 | ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling | NeurIPS 2023 | arXiv:2402.10635 | new-verified | REC | Neural-ODE + attention (CT-MHA) for irregular series; generic, not localization/cross-modal. |
| `tipirneni2022strats` | Tipirneni, Reddy, 2022 | Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series (STraTS) | ACM TKDD, vol. 16(6), art. 105 | 10.1145/3516367 (arXiv:2107.14293) | new-verified | CT | Tokenizes observations as (time,var,value) triplets w/ continuous value embedding; keeps separate embeddings, clinical. |
| `zhang2022raindrop` | Zhang, Zeman, Tsiligkaridis, Zitnik, 2022 | Graph-Guided Network for Irregularly Sampled Multivariate Time Series (Raindrop) | ICLR 2022 (OpenReview Kwm8I7dU-l5) | arXiv:2110.05357 (**no DOI**) | new-verified | CT | Leave-sensor-out robustness via learned sensor graph (vs our modality/instant dropout + single block). |
| `debrouwer2019gruodebayes` | De Brouwer, Simm, Arany, Moreau, 2019 | GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series | NeurIPS 2019 | arXiv:1905.12374 (**no DOI**) | new-verified | CT | Continuous-time GRU + Bayesian update; recurrence + ODE, which we avoid. |
| `shukla2019ipnets` | Shukla, Marlin, 2019 | Interpolation-Prediction Networks for Irregularly Sampled Time Series (IP-Nets) | ICLR 2019 (OpenReview r1efr3C9Ym) | arXiv:1909.07782 (**no DOI**) | new-verified | CT | Predecessor to mTAN; fixed-kernel interpolation onto reference points (we use no resampling). |
| `che2018grud` | Che, Purushotham, Cho, Sontag, Liu, 2018 | Recurrent Neural Networks for Multivariate Time Series with Missing Values (GRU-D) | Scientific Reports, vol. 8, art. 6085 | 10.1038/s41598-018-24271-9 (arXiv:1606.01865) | already-in-zotero (2-report agreement) | CT, WIFI | Time-decay gates + Δt/missingness masks inside an RNN; we are recurrence-free with explicit Δt + dropout. |

**Already in Zotero (Continuous-time/async group):** Che 2018 (GRU-D) [above], Shou 2024 (Dynamic Graph Neural ODE).

**Already in Zotero (Classical filtering):** Feng 2023 (Kalman+NN review), DNN-EKF UWB 2024 (Eang).

---

## Venue-local — ICINCO 2024 (SciTePress; ISBN 978-989-758-717-7; ISSN 2184-2809)

All `not-in-zotero`; must be added manually if cited. None are WiFi/IMU SOTA baselines — use for community positioning and classical-vs-learned-fusion narrative. (Full 26-paper list in `icinco-2024-relevant.md`; the in-scope:paper-1 subset is below.)

| Suggested key | Authors, 2024 | Short title | Vol/pp | Scope |
|---|---|---|---|---|
| `rafique2024lcm` | Rafique, Patti, Palesi, La Delfa | Characteristics-Based Least Common Multiple clustering for indoor positioning | 1 / 301–308 | in-scope:paper-1 |
| `grumeza2024maps` | Grumeza, Lazăr, Drămnesc, Kusper, Papadopoulos, Fachantidis, Lefkos | A Case Study in Building 2D Maps with Robots | 2 / 228–235 | in-scope:paper-1 |
| `vaghi2024uncertainty` | Vaghi, Ballardini, Fontana, Sorrenti | Uncertainty-Aware DNN for Multi-Modal Camera Localization | 2 / 80–90 | in-scope:paper-1 |
| `lourenco2024pallets` | Lourenço, Arsénio, Garrote, Nunes | Multimodal 6D Detection of Industrial Pallets (multi-head self-attention fusion) | 2 / 345–352 | in-scope:paper-1 |
| `borges2024mot` | Borges, Garrote, Nunes | A Modular Multimodal Multi-Object Tracking-by-Detection Approach (Kalman) | 2 / 336–344 | in-scope:paper-1 |
| `novak2024uavusv` | Novák, Báča, Procházka, Saska | Towards UAV-USV Collaboration... (multi-rate async IMU/GPS/vision KF) | 1 / 545–554 | in-scope:paper-1 |
| `rama2024lanechange` | Rama, Bajcinca | Edge-Featured Graph Attention Network for Lane Change Prediction (attn+RNN) | 2 / 282–289 | in-scope:paper-1 |
| `bazzi2024robomorph` | Bazzi, Shahid, Agia, Alora, Forgione, Piga, Braghin, Pavone, Roveda | RoboMorph: In-Context Meta-Learning for Robot Dynamics (transformer) | 2 / 149–156 | in-scope:paper-1 |
| `ahmed2024node` | Ahmed, Lee, Park | NODE and Contraction Methods for Dynamics Learning (Neural ODE) | 2 / 205–211 | in-scope:paper-1 |
| `mohammadi2024lstm` | Mohammadi, Ortiz-Arroyo, Stokholm-Bjerregaard, Durdevic | Multi-Step Simulation Improvement for Time Series (LSTM + exogenous) | 1 / 651–659 | in-scope:paper-1 |
| `alfaro2024triplet` | Alfaro, Cabrera, Jiménez, Reinoso, Payá | Triplet Neural Networks for Visual Localization of Mobile Robots (kNN retrieval) | 2 / 125–132 | in-scope:paper-1 |

**Top venue-local picks (per report):** #17 Triplet NN (learned embedding + kNN, mirrors WiFi-Net), #6 pallets + #9 lane-change (attention as fusion operator), #8 UAV-USV (multi-rate async), #7 MOT (classical Kalman contrast), #5 uncertainty cam-loc; #11 RoboMorph + #12 NODE + #13 LSTM (architecture framing); #1 LCM + #2 2D maps (venue indoor-loc context).

---

## Datasets / benchmarks (cross-cut)

| Key | Name | Modality | DOI / arXiv | Status |
|---|---|---|---|---|
| `torressospedra2014ujiindoorloc` | UJIIndoorLoc | WiFi RSSI fingerprint | 10.1109/IPIN.2014.7275492 | new-verified |
| `yan2020ronin` | RoNIN benchmark (42.7 h, 100 subjects) | IMU | 10.1109/ICRA40945.2020.9196860 | already-in-zotero |
| `yan2018ridi` | RIDI dataset | IMU | 10.1007/978-3-030-01261-8_38 | new-verified |
| `chen2018oxiod` | OxIOD (158 seq, ~42.5 km) | IMU | arXiv:1809.07491 | needs-verification (preprint) |
| `herath2022niloc` | NILoc benchmark (53 h) | IMU | arXiv:2203.15851 | new-verified |
| `sun2021idol` | IDOL dataset (20+ h) | IMU | 10.1609/aaai.v35i7.16763 | new-verified |

---

# COMPETITOR TABLE (from "Closest competitors" report, ranked by closeness)

Criteria — Y/N each:
- **(i) Attn/transformer fusion**
- **(ii) Continuous-time OR async/multi-rate without resampling**
- **(iii) Missing/stale-modality robustness**
- **(iv) Cross-session / cross-day real-world eval**
- **Block** = single unified block (U) vs per-modality branches (B) vs auxiliary (Aux)

| Rank | Work | (i) | (ii) | (iii) | (iv) | Block | How it differs from ours (one line) |
|---|---|---|---|---|---|---|---|
| 1 | **A-KIT** (Cohen & Klein 2024, arXiv) | Y | Partial (perm-inv handling of async arrivals; EKF propagates time) | N | Y (AUV) | Aux (set-transformer only tunes EKF noise) | INS/DVL underwater; set-transformer is EKF-auxiliary, no real-valued Δt embedding, no modality/instant dropout. |
| 2 | **AFT-VO** (Kaygusuz 2022, IROS) | Y | Y (Discretiser encodes real-valued timestamps, multi-rate) | Partial (uncertainty weighting) | Y (nuScenes/KITTI) | B (per-camera MDN branches feed fusion) | Multi-camera visual odometry, not WiFi/IMU; fuses precomputed per-source poses; robustness via uncertainty not dropout. |
| 3 | **WiMU** (Yang 2025, IMWUT) | N | Partial (targets low WiFi rate at system level) | N | Y (real-time deployment) | B (fingerprinting + IMU system) | Survey-light WiFi/IMU system, not a single attention block; no Δt embedding, no modality dropout. |
| 4 | **WIO-EKF** (Zhou 2024, IoT-J) | N | N (EKF propagation) | N | Y (multi-dataset incl. self-collected) | B (two deep branches + EKF) | Per-modality branches glued by hand-built EKF; no attention, no Δt, no dropout. |
| 5 | **iMoT** (Nguyen 2025, AAAI) | Y | Partial (Adaptive Positional Encoding for temporal discrepancy) | N | Y (inertial benchmarks) | U-ish cross-modal but IMU-only (acc+gyro) | IMU-only two channels; APE ≠ real Δt for 1 Hz vs 30 Hz; no WiFi, no modality dropout. |
| 6 | **Zhang et al.** (2021, IoT-J) | N | N (fixed sliding windows, resampled) | N | N | B-ish single LSTM, not perm-invariant | Recurrent fixed-timestep WiFi+PDR requiring resampling; no Δt, no set block, no dropout. |
| 7 | **Yu et al.** (2022, ICASSP) | N | N (recurrent fixed-timestep) | Partial (per-modality uncertainty) | N | B (per-modality RNN streams) | Per-modality recurrent streams w/ uncertainty; not perm-invariant, no real Δt, no modality dropout. |
| 8 | **mTAN** (Shukla 2021, ICLR) | Y | Y (genuine continuous-time embedding, no resampling) | N | N | Interpolates to fixed reference points | General time-series; interpolates rather than fuses modalities as a set; no dropout, no localization eval. |
| 9 | **Neural CDE** (Kidger 2020, NeurIPS) | N | Y (continuous control path, no resampling) | N | N | ODE/CDE (imputes between obs) | ODE-based not attention/set; no cross-modal WiFi/IMU fusion, no dropout. |
| 10 | **Antsfeld** (2020/21, IPIN) | N | Y (low-rate WiFi corrects high-rate PDR via KF) | N | Y (IPIN'19 challenge) | B (two models + Kalman filter) | Two separate models glued by KF; no attention, no learned Δt, no dropout. |
| 11 | **RoNIN** (2020, ICRA) | N | N (fixed-rate windows) | N | Y (unseen-subject) | none (IMU-only) | IMU-only velocity regression; no WiFi, no fusion, no Δt, no attention, no dropout. |
| 12 | **IONet** (2018, AAAI) | N | N (fixed windows) | N | N (generalizes to non-periodic motion) | none (IMU-only) | IMU-only recurrent windowed regression; no fusion, no Δt, no set block, no dropout. |

**At-a-glance neighbors vs ours** (from report):

| Property | A-KIT | AFT-VO | iMoT | **Ours** |
|---|---|---|---|---|
| Attention/transformer (i) | Y (set-tf) | Y | Y | Y (set-tf) |
| Real-valued continuous-time Δt (ii) | N (EKF) | Y (Discretiser) | Partial (APE) | **Y** |
| Single unified fusion block | N (EKF-aux) | N (per-source) | Cross-modal but IMU-only | **Y** |
| Modality/instant dropout (iii) | N | N (uncertainty) | N | **Y** |
| Domain | INS/DVL (AUV) | Multi-camera VO | IMU-only | **WiFi RSSI + IMU** |
| Cross-session real-world (iv) | Y | Y | Y | **Y (target)** |

**Novelty conclusion (cross-agent unanimous):** No published work holds all three contributions together for WiFi+IMU localization. The conjunction (continuous-time Δt + single permutation-invariant cross-modal/cross-time block + modality/instant dropout) is defensibly novel; frame novelty as the conjunction, not any single property.

---

# FLAGGED FOR MANUAL VERIFICATION

### Preprint-only (cite as @misc / arXiv; flag in double-blind text)
- **Time2Vec** (`kazemi2019time2vec`, arXiv:1907.05321) — NO peer-reviewed venue, ever. Highest-risk metadata point. (CT, WIFI reports agree.)
- **Aristorenas 2025** (`aristorenas2025permutation`, arXiv:2506.00656) — single-author, Stanford, no peer-reviewed version found. (ARCH, REC agree.)
- **A-KIT** (`cohen2024akit`, arXiv:2401.09987) — preprint only; do NOT assign a journal/conference venue. (Already in Zotero.)
- **OxIOD** (`chen2018oxiod`, arXiv:1809.07491) — preprint; prefer peer-reviewed companion arXiv:2001.04061 (L-IONet) where a published cite is required.
- **NeurIT** (`zheng2024neurit`, arXiv:2404.08939) — preprint; its "48.21%" gain is vs UniTS, NOT RoNIN. Treat as `likely-unreliable` for headline claims.

### Metadata conflicts (pick the canonical one)
- **RoNIN — arXiv 2019 vs ICRA 2020.** CANONICAL = **ICRA 2020** (10.1109/ICRA40945.2020.9196860, pp. 3146–3152). Author order: published IEEE/project-site order is **Herath, Yan, Furukawa**; arXiv/Semantic Scholar lists **Yan, Herath, Furukawa**. Cite the IEEE order.
- **CNNLoc — two distinct real papers.** Journal (IEEE Access 2019, 10.1109/ACCESS.2019.2933921, pp. 110698–110709) vs conference (IEEE SmartWorld 2019, pp. 589–595) — different author orderings. Recommend the IEEE Access journal as canonical; cite the conference only if invoking the literal "CNNLoc" name.
- **Neural ODE pages** — 6571–6583 vs 6572–6583 across reports; venue (NeurIPS 2018, Best Paper) firm. Verify pagination at camera-ready.
- **Latent ODE pages** — not consistently reported; arXiv 1907.03907 + NeurIPS 2019 firm; verify pages.
- **RIDI pages** — 621–636 vs 641–656 in the Springer LNCS 11217 volume; verify.
- **Vaswani pages** — 5998–6008 (NeurIPS proceedings) vs 6000–6010 (ACM/Curran reprint); use NeurIPS range.
- **WIO-EKF first author** — "Peng Zhou" ([COMP]) vs "Ping Zhou" ([REC]). Verify (likely Peng Zhou).
- **Zhang et al. 2021 issue no.** — no. 17 inferred from page range; double-check.
- **MM-Loc author list** — Wei, Wei, Radu reconstructed from MDPI metadata (early extracts showed only Radu); verify on CrossRef.
- **iMoT author list** — [IMU] report lists 4 authors (Nguyen, Tran, Le, Havinga); [COMP]/[REC] list 3 (Nguyen, Le, Havinga). Verify the full author list on AAAI/arXiv.

### DOI / venue needs confirmation on publisher page
- **Masrur 2025** (Transformer 5G/6G NLOS localization) — DOI 10.1109/ICCWorkshops67674.2025.11162366 verified, but **proceedings page numbers not located**; extended preprint arXiv:2501.07774. NOT added to the master tables above (borderline relevance) — include only if a multi-sensor-attention localization cite is wanted.
- **PEOPLEx** (`lajoie2024peoplex`) — IEEE Xplore doc 10622566 + arXiv:2311.18182 verified; exact DOI suffix and pages NOT independently confirmed.
- **HyTra** (`nasir2024hytra`) — small Scopus/DOAJ journal; non-peer-reviewed Preprints.org precursor exists; 96.7% UJIIndoorLoc figure is the HyTra-HF variant, not base HyTra.
- **wlan_localization repo** — no formal method paper exists; cite the repository URL and attribute the kNN method to RADAR. Do NOT fabricate a method citation.

### Author-name corrections (apply before refs.bib)
- TLIO third author = **Eddy Ilg** (not "Eric Ilg").
- IONet second author = **Chris Xiaoxuan Lu** (task wrote "Xiaoxuan Lu"); "Niki/Agathoniki Trigoni" = same person.
- Neural ODE first author = **Ricky T. Q. Chen** (= "Tian Qi Chen" in DBLP).
- Diacritics: Łukasz Kaiser, João Carreira, İsmail Güvenç, Barnabás Póczos.
- Held title fix: Yu et al. 2022 = "**Multi-Modal Recurrent Fusion for Indoor Localization**" (NOT "A Multi-Modal Recurrent Fusion-based...").

### Excluded as too weak / unverifiable (do NOT cite without independent verification)
- **MetaGraphLoc** (arXiv:2411.17781) — preprint, GNN/meta-learning (NOT transformer); do not present as peer-reviewed or attention-based, no fabricated DOI.
- **DamLoc** (Wang et al., Future Generation Computer Systems 2024) — magnetic+BLE; DOI/pages unverified.
- **EKF Wi-Fi/LiDAR/IMU** (arXiv:2509.23118) — preprint, classical EKF; low relevance.

### Smell-test (possible fabrication risk — none confirmed fabricated, but verify)
- The circulated **"CTIN reduces ATE ~46% / RTE ~32% over RoNIN"** claim is **contradicted by CTIN's own figures** (21.78% seen / 3.97% avg ATE over RoNIN-ResNet). Do NOT propagate the inflated figure.
- All other references were cross-checked against primary sources by the agents; no entry appears fabricated. The two single-author / no-venue items (Time2Vec, Aristorenas) are real but preprint-only.
