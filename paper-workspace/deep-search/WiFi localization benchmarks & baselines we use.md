# Verified Canonical Citations for an ICINCO 2026 Related Work Section: Continuous-Time Set-Transformer for Asynchronous WiFi/IMU Indoor Localization

## TL;DR
- All four priority experiment datasets/baselines are confirmed real publications with verified metadata: UJIIndoorLoc (Torres-Sospedra et al., IPIN 2014, DOI 10.1109/IPIN.2014.7275492), RADAR (Bahl & Padmanabhan, INFOCOM 2000, DOI 10.1109/INFCOM.2000.832252), and CNNLoc — which resolves to TWO distinct Song et al. papers (a 2019 IEEE Access journal article and a 2019 IEEE SmartWorld conference paper); the sharan-naribole GitHub repository cites NO formal kNN method paper, only the UJIIndoorLoc dataset.
- The remaining ~28 Related Work candidates across all five requested groups are verified with DOIs/arXiv IDs; canonical published versions are given where they exist, with preprint-only items flagged (notably Time2Vec, the Antsfeld WiFi+IMU fusion paper).
- One important correction the section must reflect: the GitHub repo's "underlying method" is a cascaded Random-Forest + weighted-kNN pipeline that cites only UJIIndoorLoc — there is no separate algorithm paper to cite, so RADAR (the canonical RSSI-kNN reference) should carry that citation role.

## Key Findings

**Priority must-find verification status:**
- (a) UJIIndoorLoc — VERIFIED. Exact pages 261–270, IEEE, 2014, DOI 10.1109/IPIN.2014.7275492. Conference held in Busan, Korea, 27–30 October 2014 (confirmed via the PMC citation index: "Proceedings of the 2014 International Conference on Indoor Positioning and Indoor Navigation (IPIN); Busan, Korea. 27–30 October 2014; pp. 261–270").
- (b) CNNLoc — VERIFIED but AMBIGUOUS: two real papers share the "CNNLoc" identity. The journal version (IEEE Access 2019) and the conference version (IEEE SmartWorld 2019) have different author orderings and DOIs. Both reported below.
- (c) sharan-naribole/wlan_localization — INSPECTED. README cites only the UJIIndoorLoc dataset (BibTeX for Torres-Sospedra 2014). No formal kNN/RSSI method paper is referenced. Method is a cascaded Random Forest + weighted kNN (k=3) pipeline. Stated explicitly below.
- (d) RADAR — VERIFIED. Vol. 2, pages 775–784, IEEE, 2000, DOI 10.1109/INFCOM.2000.832252.

## Details

### GROUP — PRIORITY EXPERIMENT DATASETS / BASELINES

**(a) [DATASET] UJIIndoorLoc**
- Authors: Joaquín Torres-Sospedra, Raúl Montoliu, Adolfo Martínez-Usó, Joan P. Avariento, Tomás J. Arnau, Mauri Benedito-Bordonau, Joaquín Huerta
- Title: "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems"
- Venue: 2014 International Conference on Indoor Positioning and Indoor Navigation (IPIN), IEEE, 2014, pp. 261–270 (Busan, Korea, 27–30 October 2014)
- DOI: 10.1109/IPIN.2014.7275492
- Method: Introduces the first large-scale, publicly available multi-building/multi-floor WLAN RSSI fingerprint database, covering three buildings of Universitat Jaume I with 520 wireless access points over more than 108,703 m². It contains 21,049 records total (19,937 training + 1,111 validation, captured roughly 4 months apart); per the UCI Machine Learning Repository (dataset 310), "Data were collected by more than 20 users using 25 different models of mobile devices." Provides standard train/validation splits used as a benchmark.
- Relation: This is the primary public benchmark dataset for WiFi fingerprinting and the dataset the paper's WiFi baselines are evaluated on. Group: WiFi fingerprinting.

**(b) [BASELINE] CNNLoc — canonical reference is ambiguous; both real papers reported.**

Version 1 (journal — recommended canonical):
- Authors: Xudong Song, Xiaochen Fan, Chaocan Xiang, Qianwen Ye, Leyu Liu, Zumin Wang, Xiangjian He, Ning Yang, Gengfa Fang
- Title: "A Novel Convolutional Neural Network Based Indoor Localization Framework with WiFi Fingerprinting"
- Venue: IEEE Access, vol. 7, 2019, pp. 110698–110709
- DOI: 10.1109/ACCESS.2019.2933921
- Method: Combines a Stacked Auto-Encoder (SAE) for feature extraction from sparse RSS data with a one-dimensional CNN for building/floor classification and coordinate regression. Evaluated on UJIIndoorLoc and Tampere datasets; introduces a new UTSIndoorLoc dataset.

Version 2 (conference — bears the literal "CNNLoc" title):
- Authors: Xudong Song, Xiaochen Fan, Xiangjian He, Chaocan Xiang, Qianwen Ye, Xiang Huang, Gengfa Fang, Liming Luke Chen, Jing Qin, Zumin Wang
- Title: "CNNLoc: Deep-Learning Based Indoor Localization with WiFi Fingerprinting"
- Venue: 2019 IEEE SmartWorld, Ubiquitous Intelligence & Computing, Advanced & Trusted Computing, Scalable Computing & Communications, Cloud & Big Data Computing, Internet of People and Smart City Innovation (SmartWorld/SCALCOM/UIC/ATC/CBDCom/IOP/SCI), IEEE, 2019, pp. 589–595
- DOI: 10.1109/SmartWorld-UIC-ATC-SCALCOM-IOP-SCI.2019.00139
- Method: Same SAE + 1D-CNN approach; this conference paper is where the "CNNLoc" name is explicitly used.
- Relation/which to cite: The name "CNNLoc" literally appears in the conference paper, but the IEEE Access journal article is the more complete and more cited canonical version. If the paper cites "CNNLoc" as a deep-learning WiFi baseline, cite the IEEE Access 2019 article and optionally the SmartWorld paper for the name. Group: WiFi fingerprinting.

**(c) [BASELINE] kNN/RSSI fingerprinting implemented by sharan-naribole/wlan_localization**
- Repository: https://github.com/sharan-naribole/wlan_localization ("A Machine Learning Approach to WLAN Fingerprinting based Localization")
- Finding: The repository README documents a cascaded pipeline — Stage 1 building classification via Random Forest (n_estimators = 100), Stage 2 per-building floor classification via weighted kNN (k = 3, Manhattan distance, distance-weighted), Stage 3 per-building-floor position regression via weighted distance-weighted kNN (k = 3). Per the README, the system "Achieves competitive accuracy (2.6-8.2m RMSE) on the benchmark UJIIndoorLoc dataset," and the config file confirms the RandomForest/KNN/WeightedKNN stage settings above.
- Citation finding: The ONLY formal citation in the repository is the UJIIndoorLoc dataset (Torres-Sospedra et al., 2014, given as BibTeX in the README's "Citation" section). The repo does NOT cite any specific kNN/RSSI method paper, course, or source for its algorithm. Explicit statement: no formal method citation exists for this repository's kNN approach.
- Recommendation: For the underlying kNN/RSSI fingerprinting method, cite the canonical RADAR paper (item d), and cite the repository itself only as the open-source implementation. Group: WiFi fingerprinting.

**(d) [BASELINE] RADAR**
- Authors: Paramvir Bahl, Venkata N. Padmanabhan
- Title: "RADAR: An In-Building RF-Based User Location and Tracking System"
- Venue: Proceedings IEEE INFOCOM 2000, Conference on Computer Communications, Nineteenth Annual Joint Conference of the IEEE Computer and Communications Societies (Cat. No.00CH37064), IEEE, 2000, vol. 2, pp. 775–784 (Tel Aviv, Israel, 26–30 March 2000)
- DOI: 10.1109/INFCOM.2000.832252
- Method: The seminal RF/WiFi RSSI fingerprinting system; records signal strength at multiple base stations and uses nearest-neighbor matching in signal space (kNN) plus signal-propagation modeling to estimate user location. It won the inaugural ACM SIGMOBILE Test-of-Time Paper Award in 2016 (per Microsoft Research).
- Relation: The classic RSSI-kNN fingerprinting reference and the conceptual ancestor of the paper's WiFi baselines. Group: WiFi fingerprinting.

### GROUP 1 — WiFi FINGERPRINTING (beyond the baselines)

**Horus**
- Authors: Moustafa Youssef, Ashok Agrawala
- Title: "The Horus WLAN Location Determination System"
- Venue: MobiSys '05: Proceedings of the 3rd International Conference on Mobile Systems, Applications, and Services, ACM, 2005, pp. 205–218
- DOI: 10.1145/1067170.1067193
- Method: A probabilistic WLAN fingerprinting system modeling the RSSI distribution at each location, using probability distributions and location clustering to achieve high accuracy at low computational cost.
- Relation: Classic probabilistic (vs. deterministic kNN) WiFi fingerprinting baseline. Group: WiFi fingerprinting.

**DeepFi**
- Authors: Xuyu Wang, Lingjun Gao, Shiwen Mao, Santosh Pandey
- Title: "DeepFi: Deep Learning for Indoor Fingerprinting Using Channel State Information"
- Venue: 2015 IEEE Wireless Communications and Networking Conference (WCNC), IEEE, 2015, pp. 1666–1671
- DOI: 10.1109/WCNC.2015.7127718
- Method: Uses a deep network (restricted Boltzmann machine with greedy layer-wise training) to learn weights as fingerprints from CSI amplitude, with a probabilistic RBF data-fusion step for online localization.
- Relation: Early deep-learning WiFi fingerprinting (CSI rather than RSSI), illustrating learned feature extraction for localization. Group: WiFi fingerprinting.

**WiFi fingerprinting survey**
- Authors: Suining He, S.-H. Gary Chan
- Title: "Wi-Fi Fingerprint-Based Indoor Positioning: Recent Advances and Comparisons"
- Venue: IEEE Communications Surveys & Tutorials, vol. 18, no. 1, 2016, pp. 466–490
- DOI: 10.1109/COMST.2015.2464084
- Method: A comprehensive survey of WiFi fingerprinting localization covering advanced techniques (temporal/spatial patterns, user collaboration, motion sensors) and efficient deployment.
- Relation: Survey establishing the WiFi fingerprinting landscape and motivating sensor fusion. Group: WiFi fingerprinting.

**Transformer-for-WiFi-localization (HyTra)**
- Authors: Muneeb Nasir, Kiara Esguerra, Ibrahima Faye, Tong Boon Tang, Mazlaini Yahya, Afidalina Tumian, Eric Tatt Wei Ho
- Title: "HyTra: Hyperclass Transformer for WiFi Fingerprinting-based Indoor Localization"
- Venue: Transactions on Energy Systems and Engineering Applications (TESEA), Universidad Tecnológica de Bolívar, vol. 5, no. 1, 2024, pp. 1–24
- DOI: 10.32397/tesea.vol5.n1.542
- Method: An encoder-only Transformer using learnable embeddings and self-attention over WiFi RSS fingerprints. The hierarchical variant HyTra-HF (not the base HyTra) reports 96.7% floor-classification accuracy on UJIIndoorLoc ("HyTra-HF outperforms existing deep learning solutions by obtaining 96.7% accuracy for the floor classification task on the UJIIndoorLoc dataset").
- Relation: Directly relevant prior art applying self-attention/Transformer to WiFi RSSI fingerprinting, but uses standard token embeddings without continuous-time Δt encoding and does not fuse asynchronous IMU. Group: WiFi fingerprinting / attention. (Caveat: published in a smaller Scopus/DOAJ-indexed journal; an earlier non-peer-reviewed Preprints.org version exists.)

### GROUP 2 — INERTIAL / IMU LOCALIZATION

**IONet**
- Authors: Changhao Chen, Xiaoxuan Lu, Andrew Markham, Niki Trigoni
- Title: "IONet: Learning to Cure the Curse of Drift in Inertial Odometry"
- Venue: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), AAAI Press, 2018, pp. 6468–6476
- arXiv: 1802.02209 (canonical version is the AAAI proceedings paper)
- Method: Segments IMU data into independent windows and uses deep LSTMs to regress polar displacement/heading changes, breaking the continuous double-integration that causes drift.
- Relation: Foundational deep inertial odometry; the paper's IMU stream modeling builds on this line of work. Group: inertial/IMU.

**RoNIN**
- Authors: Sachini Herath, Hang Yan, Yasutaka Furukawa
- Title: "RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods"
- Venue: 2020 IEEE International Conference on Robotics and Automation (ICRA), IEEE, 2020, pp. 3146–3152
- DOI: 10.1109/ICRA40945.2020.9196860; arXiv: 1905.12853
- Method: Provides the largest inertial navigation database — more than 42.7 hours of IMU and ground-truth 3D motion data from 100 human subjects (per the RoNIN project page) — and ResNet/LSTM/TCN neural architectures that regress 2D velocity robust to device orientation/placement.
- Relation: State-of-the-art neural inertial navigation; relevant baseline for the IMU modality. Group: inertial/IMU.

**RIDI**
- Authors: Hang Yan, Qi Shan, Yasutaka Furukawa
- Title: "RIDI: Robust IMU Double Integration"
- Venue: Computer Vision – ECCV 2018, Lecture Notes in Computer Science vol. 11217, Springer, 2018, pp. 641–656
- DOI: 10.1007/978-3-030-01261-8_38; arXiv: 1712.09004
- Method: Regresses a velocity vector from histories of linear acceleration and angular velocity, then corrects raw accelerations so that double integration yields low-drift trajectories.
- Relation: Early data-driven IMU navigation establishing the regress-velocity paradigm. Group: inertial/IMU.

**TLIO**
- Authors: Wenxin Liu, David Caruso, Eddy Ilg, Jing Dong, Anastasios I. Mourikis, Kostas Daniilidis, Vijay Kumar, Jakob Engel
- Title: "TLIO: Tight Learned Inertial Odometry"
- Venue: IEEE Robotics and Automation Letters (RA-L), vol. 5, no. 4, 2020, pp. 5653–5660
- DOI: 10.1109/LRA.2020.3007421; arXiv: 2007.01867
- Method: A tightly-coupled Extended Kalman Filter that fuses a ResNet-based network regressing 3D displacement and its uncertainty with IMU kinematics to estimate pose, velocity, and sensor biases.
- Relation: Combines learned inertial regression with probabilistic filtering — relevant to graceful degradation and uncertainty handling. Group: inertial/IMU.

### GROUP 3 — MULTIMODAL FUSION FOR LOCALIZATION

**Deep smartphone WiFi+IMU fusion**
- Authors: Leonid Antsfeld, Boris Chidlovskii, Emilio Sansano-Sansano
- Title: "Deep Smartphone Sensors-WiFi Fusion for Indoor Positioning and Tracking"
- Venue: arXiv preprint, 2020
- arXiv: 2011.10799
- Method: A deep PDR model produces high-rate relative position from inertial sensors; a Kalman filter corrects PDR drift using WiFi absolute-position predictions, followed by a map-free walkable-path projection. Tested on the IPIN'19 challenge dataset.
- Relation: Directly comparable WiFi+IMU fusion, but uses a classical Kalman-filter pipeline rather than a single attention-based fusion model — the contrast the paper draws. Group: multimodal fusion. (Caveat: preprint only; no peer-reviewed venue located.)

**WiFi + PDR + landmarks Kalman fusion**
- Authors: Zhenghua Chen, Han Zou, Hao Jiang, Qingchang Zhu, Yeng Chai Soh, Lihua Xie
- Title: "Fusion of WiFi, Smartphone Sensors and Landmarks Using the Kalman Filter for Indoor Localization"
- Venue: Sensors, MDPI, vol. 15, no. 1, 2015, pp. 715–732
- DOI: 10.3390/s150100715
- Method: Fuses WiFi fingerprinting, PDR (from smartphone inertial sensors), and indoor landmarks via Kalman filtering, where PDR smooths WiFi jumps and WiFi corrects PDR drift.
- Relation: Canonical classical WiFi+IMU(PDR) fusion baseline; contrasts with the paper's learned permutation-invariant fusion. Group: multimodal fusion.

### GROUP 4 — ATTENTION / TRANSFORMER / SET ARCHITECTURES

**Transformer**
- Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- Title: "Attention Is All You Need"
- Venue: Advances in Neural Information Processing Systems 30 (NeurIPS 2017), Curran Associates, 2017, pp. 5998–6008
- arXiv: 1706.03762
- Method: Introduces the Transformer, an architecture based solely on multi-head self-attention, dispensing with recurrence and convolution.
- Relation: The foundational self-attention architecture underlying the paper's set-transformer fusion block. Group: attention/transformer/set.

**Set Transformer**
- Authors: Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh
- Title: "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
- Venue: Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR vol. 97, 2019, pp. 3744–3753
- arXiv: 1810.00825
- Method: Attention-based modules (SAB, ISAB, PMA) for permutation-invariant set-input functions, reducing self-attention cost from quadratic to linear via inducing points.
- Relation: The architectural template for the paper's permutation-invariant-over-modalities fusion. Group: attention/transformer/set.

**Deep Sets**
- Authors: Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, Alexander J. Smola
- Title: "Deep Sets"
- Venue: Advances in Neural Information Processing Systems 30 (NeurIPS 2017), Curran Associates, 2017, pp. 3391–3401
- arXiv: 1703.06114
- Method: Characterizes permutation-invariant/equivariant functions on sets (the sum-decomposition form ρ(Σ φ(x))) and proposes corresponding neural architectures.
- Relation: Theoretical foundation for permutation-invariant treatment of the modality/observation set. Group: attention/transformer/set.

**Perceiver IO**
- Authors: Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira
- Title: "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
- Venue: International Conference on Learning Representations (ICLR) 2022
- arXiv: 2107.14795
- Method: Augments the Perceiver with a flexible querying mechanism, using cross-attention to a latent array to handle arbitrary multimodal inputs and structured outputs while scaling linearly with input/output size.
- Relation: Closely related general-purpose attention architecture for heterogeneous multimodal inputs, but without explicit continuous-time Δt encoding for asynchronous streams. Group: attention/transformer/set.

### GROUP 5 — CONTINUOUS-TIME / ASYNCHRONOUS / IRREGULARLY-SAMPLED MODELS

**Neural ODEs**
- Authors: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
- Title: "Neural Ordinary Differential Equations"
- Venue: Advances in Neural Information Processing Systems 31 (NeurIPS 2018), Curran Associates, 2018, pp. 6571–6583 (NeurIPS 2018 best-paper award)
- arXiv: 1806.07366
- Method: Parameterizes the derivative of the hidden state with a neural network solved by a black-box ODE solver, giving continuous-depth/continuous-time models with constant memory cost.
- Relation: Foundational continuous-time modeling approach motivating Δt-aware handling of irregular sampling. Group: continuous-time/async.

**Latent ODEs**
- Authors: Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud
- Title: "Latent ODEs for Irregularly-Sampled Time Series"
- Venue: Advances in Neural Information Processing Systems 32 (NeurIPS 2019), Curran Associates, 2019
- arXiv: 1907.03907
- Method: Generalizes RNNs to continuous-time hidden dynamics (ODE-RNN) and uses it as the recognition network of a Latent ODE, naturally handling arbitrary time gaps between observations and optionally modeling observation times with a Poisson process.
- Relation: Direct prior art on continuous-time modeling of irregular/asynchronous sampling without resampling. Group: continuous-time/async.

**GRU-D**
- Authors: Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu
- Title: "Recurrent Neural Networks for Multivariate Time Series with Missing Values"
- Venue: Scientific Reports, vol. 8, article 6085, 2018
- DOI: 10.1038/s41598-018-24271-9; arXiv: 1606.01865
- Method: GRU-D incorporates masking and time-interval (Δt) representations with learned exponential decay to handle informative missingness in multivariate time series.
- Relation: Key precedent for using Δt and missingness masks for graceful degradation under missing/stale sensors. Group: continuous-time/async.

**mTAND (Multi-Time Attention Networks)**
- Authors: Satya Narayan Shukla, Benjamin M. Marlin
- Title: "Multi-Time Attention Networks for Irregularly Sampled Time Series"
- Venue: International Conference on Learning Representations (ICLR) 2021
- arXiv: 2101.10318
- Method: Learns a continuous-time embedding and a time-attention mechanism (reference points as queries, observed times as keys) to produce fixed-length representations of irregularly sampled multivariate series.
- Relation: Closest methodological analog — continuous-time embeddings + attention for irregular sampling; the paper extends this idea to cross-modal fusion in a single block. Group: continuous-time/async.

**Time2Vec**
- Authors: Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker
- Title: "Time2Vec: Learning a Vector Representation of Time"
- Venue: arXiv preprint, 2019 (no peer-reviewed venue)
- arXiv: 1907.05321
- Method: A model-agnostic vector embedding of time combining a linear term and periodic (sinusoidal) terms with learnable frequencies and phases, droppable into existing architectures.
- Relation: Directly relevant continuous-time encoding mechanism for elapsed-time Δt features. Group: continuous-time/async. (Caveat: preprint only — flag as non-peer-reviewed.)

## Recommendations
1. **Baselines:** Cite UJIIndoorLoc, RADAR, and CNNLoc (IEEE Access 2019 as the canonical version; add the SmartWorld 2019 paper if the literal "CNNLoc" name is invoked) as the experiment datasets/baselines.
2. **GitHub kNN baseline:** Do NOT invent a method citation. Describe it as an open-source cascaded Random-Forest + weighted-kNN implementation (cite the repository URL) and attribute the underlying RSSI-kNN method to RADAR. Benchmark to change this stance: only if the repository adds an explicit method citation in a future commit.
3. **Architecture narrative:** Anchor on Transformer + Set Transformer + Deep Sets (and Perceiver IO as a multimodal-attention contrast), making explicit that none use a continuous-time Δt encoding.
4. **Continuous-time narrative:** Anchor on Neural ODEs, Latent ODEs, mTAND, GRU-D, and Time2Vec (flag Time2Vec as a preprint). Use mTAND and GRU-D as the closest competitors to contrast against the paper's single self-attention block performing simultaneous cross-modal and cross-time fusion.
5. **Positioning of the contribution:** Frame contribution (i) against Time2Vec/mTAND/GRU-D (continuous-time Δt encoding), contribution (ii) against Set Transformer/Deep Sets/Perceiver IO (permutation-invariant single-block fusion), and contribution (iii) against GRU-D/TLIO and the Kalman-filter fusion papers (graceful degradation under missing/stale sensors).

## Caveats
- **CNNLoc maps to two genuine papers** with different author orderings and DOIs; choose deliberately and, if precision matters, cite the IEEE Access journal article as canonical.
- **Preprint/weaker-venue items, flagged:** Time2Vec (arXiv only, no peer-reviewed venue); the Antsfeld et al. WiFi+IMU fusion paper (arXiv only); HyTra (smaller Scopus/DOAJ journal, with an earlier non-peer-reviewed Preprints.org version). The 96.7% UJIIndoorLoc figure for HyTra is specifically for the hierarchical HyTra-HF variant, not the base model.
- **Latent ODEs** NeurIPS 2019 proceedings page numbers were not consistently reported across sources; the arXiv ID (1907.03907) and venue are firm. IONet's canonical record is the AAAI-18 proceedings (pp. 6468–6476), not the arXiv preprint.
- **arXiv duplicates:** IONet, RoNIN, RIDI, TLIO, Deep Sets, Set Transformer, Neural ODEs, mTAND, and Perceiver IO all have arXiv preprints; the canonical published venues are reported above and should be preferred in the bibliography.
- All DOIs and arXiv IDs above were verified against publisher pages, DBLP, official proceedings (NeurIPS/PMLR/AAAI/ACM/IEEE), or the authors' own pages during research. No DOIs, venues, or author lists were fabricated; any item that could not be fully verified is flagged in this section.