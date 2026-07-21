# Zotero completion + 5-notebook organization map

Purpose: (1) the **named papers to ADD to Zotero** to complete the corpus, and
(2) the **per-group grouping** = the 5 NotebookLM notebooks to create and upload
into. `✓` = already in Zotero (don't re-add). `+` = add. Identifiers given so
you can find each fast. Metadata conflicts/preprint flags are handled later at
the `refs.bib` stage (see `related-work-references.md`), not now.

NOTE: the Zotero MCP is read-only, so I cannot create collections or move items.
Use this file as the map: add the `+` items, then upload each group's papers
(✓ and +) into its notebook.

---

## Notebook 1 — WiFi fingerprinting
✓ Have: Feng 2022 (DL-for-WiFi survey), Martín-Frechina 2025 (Wi-Fi/BLE review),
Turgut 2024, Zhang 2023 (CSI attention-ResCNN), Tiku 2022 (multi-head attention),
Ai 2022 (DRVAT), Zhang 2022 (TIPS), Ott 2024 (Radio Foundation Models),
Bhatia 2025 (decoder-only transformer), SwinULoc 2026, Abdullah 2025 (RIS
transformer), "DL indoor positioning + uncertainty", "Conformal Prediction for
Indoor Positioning".

+ ADD:
- UJIIndoorLoc — Torres-Sospedra et al. 2014 — DOI 10.1109/IPIN.2014.7275492
- RADAR — Bahl & Padmanabhan 2000 — DOI 10.1109/INFCOM.2000.832252
- CNNLoc (journal) — Song et al. 2019, IEEE Access — DOI 10.1109/ACCESS.2019.2933921
- Horus — Youssef & Agrawala 2005 — DOI 10.1145/1067170.1067193
- DeepFi — Wang et al. 2015 — DOI 10.1109/WCNC.2015.7127718
- Wi-Fi fingerprint survey — He & Chan 2016 — DOI 10.1109/COMST.2015.2464084
- All-embracing Transformers — Nguyen et al. 2024, PMC — DOI 10.1016/j.pmcj.2024.101912
- Aristorenas 2025 (perm-invariant set RSSI) — arXiv:2506.00656 (preprint)
- HyTra — Nasir et al. 2024 (small journal; optional) — DOI 10.32397/tesea.vol5.n1.542
- wlan_localization — GitHub repo (software cite, no paper): github.com/sharan-naribole/wlan_localization

## Notebook 2 — Inertial / IMU
✓ Have: RoNIN (Herath/Yan/Furukawa), Cohen & Klein 2024 (inertial-DL survey).

+ ADD:
- IONet — Chen et al. 2018, AAAI — DOI 10.1609/aaai.v32i1.12102 (arXiv:1802.02209)
- RIDI — Yan et al. 2018, ECCV — DOI 10.1007/978-3-030-01261-8_38 (arXiv:1712.09004)
- TLIO — Liu et al. 2020, RA-L — DOI 10.1109/LRA.2020.3007421 (arXiv:2007.01867)
- IDOL — Sun et al. 2021, AAAI — DOI 10.1609/aaai.v35i7.16763 (arXiv:2102.04024)
- RNIN-VIO — Chen et al. 2021, ISMAR — DOI 10.1109/ISMAR52148.2021.00044
- CTIN — Rao et al. 2022, AAAI — DOI 10.1609/aaai.v36i5.20479 (arXiv:2112.02143)
- NILoc (Neural Inertial Localization) — Herath et al. 2022, CVPR — arXiv:2203.15851
- IMUNet — Zeinali et al. 2024, IEEE TIM — arXiv:2208.00068
- iMoT — Nguyen et al. 2025, AAAI — DOI 10.1609/aaai.v39i6.32664 (arXiv:2412.12190)  ← CLOSEST competitor, must-have
- EqNIO — Jayanth et al. 2025, ICLR — arXiv:2408.06321
- RIOT — Brotchie et al. 2023, Sensors — DOI 10.3390/s23063217
- OxIOD (dataset) — Chen et al. 2018 — arXiv:1809.07491 (preprint)
- NeurIT — Zheng et al. 2024 (preprint, weak; optional) — arXiv:2404.08939

## Notebook 3 — Multimodal fusion (localization)
✓ Have: Yu 2022 (Multi-Modal Recurrent Fusion), Antsfeld 2020, WIO-EKF 2024 (Zhou),
Liu 2025 (survey), Wang 2024 (robust multi-scale), WiMU 2025 (Yang), Geneva 2018
(async), Silva 2023 (dataset), Abdalla 2026 (dataset), Łukasik 2024 (review),
Wang & Ahmad 2025 (AMR review).

+ ADD:
- LSTM WiFi+PDR fusion — Zhang et al. 2021, IEEE IoT-J — DOI 10.1109/JIOT.2021.3067515
- MM-Loc — Wei et al. 2021, Sensors — DOI 10.3390/s21227488
- SmartFPS — Hua et al. 2023, Front. Neurorobot. — DOI 10.3389/fnbot.2023.1121623
- Fusion-DHL — Herath et al. 2021, ICRA — DOI 10.1109/ICRA48506.2021.9561115 (arXiv:2105.08837)
- WiFi+PDR+landmarks Kalman — Chen et al. 2015, Sensors — DOI 10.3390/s150100715
- PEOPLEx — Lajoie et al. 2024, ICC — arXiv:2311.18182 (DOI suffix to confirm)
- ModDrop (modality-dropout precedent) — Neverova et al. 2016, TPAMI — DOI 10.1109/TPAMI.2015.2461544

## Notebook 4 — Attention / transformer / set
✓ Have: AFT-VO 2022 (Kaygusuz), A-KIT 2024 (Cohen & Klein), EffLoc 2024 (Xiao),
Lin & Evans 2025 (place recognition).

+ ADD:
- Attention Is All You Need — Vaswani et al. 2017, NeurIPS — arXiv:1706.03762  ← PILLAR
- Set Transformer — Lee et al. 2019, ICML — arXiv:1810.00825  ← PILLAR
- Deep Sets — Zaheer et al. 2017, NeurIPS — arXiv:1703.06114
- Perceiver — Jaegle et al. 2021, ICML — arXiv:2103.03206
- Perceiver IO — Jaegle et al. 2022, ICLR — arXiv:2107.14795
- PI-RNN sound-source tracking — Diaz-Guerra et al. 2023 (optional) — DOI 10.61782/fa.2023.1132

## Notebook 5 — Continuous-time / async / irregularly-sampled
✓ Have: GRU-D (Che 2018), Shou 2024 (Dynamic Graph Neural ODE).

+ ADD:
- mTAN — Shukla & Marlin 2021, ICLR — arXiv:2101.10318  ← PILLAR (our Δt encoding ancestor)
- Neural ODE — Chen et al. 2018, NeurIPS — arXiv:1806.07366  ← PILLAR
- Latent ODE — Rubanova et al. 2019, NeurIPS — arXiv:1907.03907  ← PILLAR
- Time2Vec — Kazemi et al. 2019 — arXiv:1907.05321 (preprint)  ← PILLAR
- Neural CDE — Kidger et al. 2020, NeurIPS — arXiv:2005.08926
- SeFT (Set Functions for Time Series) — Horn et al. 2020, ICML — arXiv:1909.12064
- ContiFormer — Chen et al. 2023, NeurIPS — arXiv:2402.10635
- STraTS — Tipirneni & Reddy 2022, ACM TKDD — DOI 10.1145/3516367
- Raindrop — Zhang et al. 2022, ICLR — arXiv:2110.05357
- GRU-ODE-Bayes — De Brouwer et al. 2019, NeurIPS — arXiv:1905.12374
- IP-Nets — Shukla & Marlin 2019, ICLR — arXiv:1909.07782

(Classical filtering, already in Zotero: Feng 2023 (Kalman+NN review),
DNN-EKF UWB 2024 (Eang) — keep with Notebook 3 if you want the classical contrast there.)

---

## Optional — venue-local ICINCO 2024 (community positioning only, NOT SOTA)
Not in Zotero; add only if you want venue citations. Map to notebooks by theme:
- N1 (WiFi/loc): Rafique 2024 (LCM clustering), Grumeza 2024 (2D maps)
- N3 (fusion): Vaghi 2024 (uncertainty cam-loc), Lourenço 2024 (pallets, attn fusion),
  Borges 2024 (MOT, Kalman), Novák 2024 (UAV-USV multi-rate async)
- N4 (attention): Rama 2024 (graph-attn lane change), Bazzi 2024 (RoboMorph transformer)
- N5 (continuous-time): Ahmed 2024 (NODE), Mohammadi 2024 (LSTM multi-step)
- N1/analogue: Alfaro 2024 (triplet NN + kNN retrieval)
(Full details + author diacritics in `icinco-2024-relevant.md`.)

---

## To-add count
- N1 WiFi: ~9 (+repo)   N2 IMU: ~13   N3 fusion: ~7   N4 attention/set: ~6   N5 continuous-time: ~11
- Plus 11 optional venue-local ICINCO papers.
Must-haves (do not skip): the 6 pillars (Vaswani, Lee/Set-Transformer, mTAN,
Neural ODE, Latent ODE, Time2Vec), the benchmarks (UJIIndoorLoc, CNNLoc, RADAR),
and the closest competitor (iMoT).
