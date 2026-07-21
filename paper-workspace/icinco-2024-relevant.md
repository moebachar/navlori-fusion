# ICINCO 2024 — Relevant Papers for Citation

Curated candidate list for the ICINCO 2026 submission
("Async-Robust Multi-Modal Indoor Localization via a Continuous-Time
Set-Transformer", WiFi + IMU). Built by iterative querying of the
"ICINCO 2024 proceedings" NotebookLM notebook (2 sources: Volume 1 =
`de3ed65e…`, Volume 2 = `b82c0d66…`) and cross-checked against the
Zotero library.

**ISBN/ISSN for all entries:** ISBN 978-989-758-717-7; ISSN 2184-2809.
21st International Conference on Informatics in Control, Automation and
Robotics (ICINCO 2024), SciTePress.

## How to read the tags

- **Scope tag** (from `scope.md`): the conference paper is WiFi+IMU,
  one set-transformer, indoor localization. Camera/odometry are
  deferred to the journal (paper-2).
  - `[in-scope:paper-1]` — citable in the conference paper's Introduction
    or Related Work (indoor localization; multimodal/attention fusion;
    async/multi-rate or continuous-time sequence models; learned
    localization via embedding+nearest-neighbour).
  - `[hold-for:paper-2]` — camera / visual-odometry specific; belongs to
    the journal's 4-modality + camera story.
  - `[context-only]` — tangential (navigation control, industrial
    inspection, USV/UAV control, simulation); usable as broad
    Introduction motivation, not as core related work.
- **Zotero status:** every candidate was probed in Zotero by title,
  author surname, and topic. **None of the ICINCO 2024 proceedings papers
  are in the library** (the library holds journal/preprint SOTA such as
  CSI-fingerprinting, Latent ODEs, SwinULoc — not these conference
  papers). All entries are therefore `[not-in-zotero]`. A Zotero search
  for "ICINCO" returned zero items. See the caveat at the end.

> ⚠️ **Realism note.** ICINCO is a control/automation/robotics venue, not
> a dedicated indoor-positioning conference (e.g. IPIN). There is **no
> WiFi-RSSI fingerprinting paper and no inertial-/IMU dead-reckoning
> paper** in ICINCO 2024 that could serve as a direct SOTA baseline for
> our WiFi+IMU pipeline. The relevant papers below are best used to (a)
> situate indoor localization and multimodal/attention fusion *within the
> ICINCO community* (helps the "Relevance" reviewer score), and (b)
> contrast classical filtering vs. learned/attention fusion. The genuine
> WiFi/IMU SOTA baselines (wlan_localization, RoNIN, UJIIndoorLoc) come
> from the repo/Zotero, not from these proceedings.

---

## A. Indoor localization / positioning (closest to our topic)

### 1. Characteristics-Based Least Common Multiple: A Novel Clustering Algorithm to Optimize Indoor Positioning
- **Authors:** Hamaad Rafique, Davide Patti, Maurizio Palesi, Gaetano Carmelo La Delfa
- **Volume / pages:** Vol. 1, pp. 301–308
- **Topic tags:** indoor-localization, magnetic-field, machine-learning, clustering, fingerprinting
- **Relevance:** Proposes an LCM-based clustering algorithm to organise
  magnetic-field fingerprints for indoor positioning, explicitly framing
  WiFi/RFID/BLE/MFS as the alternative-to-GPS sensor families. Useful as
  a venue-local indoor-localization citation and as a foil for "classical
  clustering of fingerprints" vs. our learned embedding approach.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 2. A Case Study in Building 2D Maps with Robots
- **Authors:** Theodor-Radu Grumeza, Thomas-Andrei Lazăr, Isabela Drămnesc, Gabor Kusper, Konstantinos Papadopoulos, Nikolaos Fachantidis, Ioannis Lefkos
- **Volume / pages:** Vol. 2, pp. 228–235
- **Topic tags:** indoor-localization, WiFi-RSSI, RFID, UWB, indoor-mapping, LiDAR, SLAM
- **Relevance:** Its State-of-the-Art section surveys indoor localization
  by WiFi signal-strength heat-maps, RFID/beacon triangulation, and UWB
  distance estimation, then builds 2D LiDAR maps on limited hardware.
  Good single cite for the WiFi/RFID/UWB indoor-localization taxonomy in
  our Related Work, and it references PDR/Kalman filtering for positioning.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 3. Dynamic Position Estimation and Flocking Control in Multi-Robot Systems
- **Authors:** Jonatan Alvarez, Assia Belbachir
- **Volume / pages:** Vol. 1, pp. 269–276
- **Topic tags:** position-estimation, multi-robot, GPS-denied
- **Relevance:** Addresses dynamic position estimation for GPS-denied
  agents (the defining constraint of indoor environments) within a
  multi-robot flocking controller. Tangential to single-agent indoor
  localization but reinforces the GPS-denied motivation.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

### 4. Drone Technology for Efficient Warehouse Product Localization
- **Authors:** Assia Belbachir, Antonio M. Ortiz, Erik T. Hauge, Ahmed Nabil Belbachir, Giusy Bonanno, Emanuele Ciccia, Giorgio Felline
- **Volume / pages:** Vol. 2, pp. 357–364
- **Topic tags:** indoor-localization, UWB, RFID, vision, warehouse
- **Relevance:** Drone-camera relative-positioning system whose State of
  the Art reviews RFID, GNSS, and impulse-radio UWB Time-of-Arrival
  indoor-localization systems. Application is camera/relative-positioning
  (not our setting), so it serves as positioning-context rather than a
  core method cite.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

---

## B. Multimodal / multi-sensor fusion (our core method family)

### 5. Uncertainty-Aware DNN for Multi-Modal Camera Localization
- **Authors:** M. Vaghi, A. L. Ballardini, S. Fontana, D. G. Sorrenti
- **Volume / pages:** Vol. 2, pp. 80–90
- **Topic tags:** sensor-fusion, visual-localization, deep-learning, uncertainty, pose-regression, multimodal
- **Relevance:** Fuses RGB images with 3D LiDAR maps in a DNN
  (CMRNet-based) for 6-DoF localization and adds epistemic-uncertainty
  estimation (Deep Evidential Regression, MC-Dropout, Deep Ensembles).
  Strong Related-Work cite for *learned multimodal localization with
  uncertainty* — adjacent to our conformal/uncertainty discussion and to
  the multimodal-fusion theme.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 6. Multimodal 6D Detection of Industrial Pallets, in Real and Virtual Environments, with Applications in Industrial AMRs
- **Authors:** José Lourenço, Gonçalo Arsénio, Luís Garrote, Urbano Nunes
- **Volume / pages:** Vol. 2, pp. 345–352
- **Topic tags:** sensor-fusion, attention, multimodal, deep-learning, 6D-pose, RGB-D
- **Relevance:** Extends DenseFusion with **multi-head self-attention** to
  fuse RGB and depth features for 6D pose in AMRs. Directly supports our
  claim that attention is an effective multimodal-fusion operator — a
  venue-local example of cross-attention fusion of heterogeneous sensors.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 7. A Modular Multimodal Multi-Object Tracking-by-Detection Approach, with Applications in Outdoor and Indoor Environments
- **Authors:** Eduardo Borges, Luís Garrote, Urbano J. Nunes
- **Volume / pages:** Vol. 2, pp. 336–344
- **Topic tags:** sensor-fusion, multimodal, Kalman-filtering, tracking, deep-learning, indoor
- **Relevance:** Tracking-by-detection that fuses LiDAR 3D point clouds
  with RGB (PointPillars + YOLOv8) and associates detections with a
  constant-velocity **Kalman filter**. Useful as the classical-filter
  contrast in Related Work (Kalman fusion of modalities) and as an
  indoor/outdoor multimodal example.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 8. Towards UAV-USV Collaboration in Harsh Maritime Conditions Including Large Waves
- **Authors:** Filip Novák, Tomáš Báča, Ondřej Procházka, Martin Saska
- **Volume / pages:** Vol. 1, pp. 545–554
- **Topic tags:** sensor-fusion, IMU, Kalman-filtering, async, multi-rate, state-estimation
- **Relevance:** Implements an explicit **multi-rate** state estimator
  that fuses 100 Hz IMU, 10 Hz GPS, and 30/50 Hz visual detections with a
  Linear Kalman Filter. One of the few ICINCO 2024 papers that confronts
  *asynchronous, multi-rate* sensor streams head-on — a venue-local cite
  for the async-fusion motivation our paper is built on.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 9. Multi-Modal Deep Learning Architecture Based on Edge-Featured Graph Attention Network for Lane Change Prediction
- **Authors:** Petrit Rama, Naim Bajcinca
- **Volume / pages:** Vol. 2, pp. 282–289
- **Topic tags:** attention, GNN, RNN, multimodal, temporal-fusion, deep-learning
- **Relevance:** Combines an edge-featured **graph attention network** for
  spatial interaction with **RNN** modules over a time-window for
  temporal sequence modelling on multimodal inputs (cameras + vehicle
  state + lane features). Supports our "attention for cross-modal fusion +
  sequence model for cross-time" framing with a concrete attention+temporal
  example.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 10. LiDAR-Based Object Recognition for Robotic Inspection of Power Lines
- **Authors:** José Mário Nishihara de Albuquerque, Ronnier Frates Rohrich
- **Volume / pages:** Vol. 2, pp. 197–204
- **Topic tags:** multimodal, LiDAR, inspection
- **Relevance:** A multimodal predictive-inspection robot fusing LiDAR,
  acoustic, spectral, ToF, thermal, and depth sensing. Only loosely
  related (no localization target); usable as a breadth example of
  multimodal sensing at the venue.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

---

## C. Attention / transformer / sequence & continuous-time models (our architecture family)

### 11. RoboMorph: In-Context Meta-Learning for Robot Dynamics Modeling
- **Authors:** Manuel Bianchi Bazzi, Asad Ali Shahid, Christopher Agia, John Alora, Marco Forgione, Dario Piga, Francesco Braghin, Marco Pavone, Loris Roveda
- **Volume / pages:** Vol. 2, pp. 149–156
- **Topic tags:** transformer, attention, sequence-model, robotics, meta-learning
- **Relevance:** A **transformer encoder-decoder with multi-head and
  causal multi-head attention** that learns a meta-dynamical model of a
  robot arm in-context (predicting torques and end-effector poses).
  Best venue-local evidence that transformer/attention architectures are
  being adopted for robot sequence modelling — supports our architecture
  choice in Related Work.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 12. NODE and Contraction Methods for Dynamics Learning from Human Expert Demonstrations
- **Authors:** Tufail Ahmed, Sangmoon Lee, Ju H. Park
- **Volume / pages:** Vol. 2, pp. 205–211
- **Topic tags:** neural-ODE, continuous-time, sequence-model, robotics
- **Relevance:** Uses **Neural ODEs** to model continuous-time nonlinear
  trajectories from demonstrations. Resonates directly with our
  "continuous-time" angle: a venue-local point of contrast — NODE models
  continuous dynamics via an ODE solver, whereas we inject continuous Δt
  through a time-encoding on each token.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 13. Multi-Step Simulation Improvement for Time Series Using Exogenous State Variables
- **Authors:** Esmaeel Mohammadi, Daniel Ortiz-Arroyo, Mikkel Stokholm-Bjerregaard, Petar Durdevic
- **Volume / pages:** Vol. 1, pp. 651–659
- **Topic tags:** LSTM, time-series, sequence-model, multi-step-forecasting
- **Relevance:** **LSTM** sequence model that injects exogenous variables
  at each step to curb compounding multi-step error, and cites Neural-ODE
  smoothing of irregular series. Relevant to our temporal-fusion
  discussion and the journal's LSTM-aggregator note (sequence modelling of
  multi-rate state).
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 14. Automated Detection of Defects on Metal Surfaces Using Vision Transformers
- **Authors:** Toqa Alaa, Mostafa Kotb, Arwa Zakaria, Mariam Diab, Walid Gomaa
- **Volume / pages:** Vol. 2, pp. 36–45
- **Topic tags:** transformer, attention, ViT, deep-learning
- **Relevance:** A **Vision Transformer** (self-attention) + CNN/MLP hybrid
  for defect classification. Off-topic application, but a citable data
  point if we want to note the breadth of transformer adoption at ICINCO.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

### 15. Two-Stage Fault Detection and Control Approach for DFIG-Based Wind Energy Conversion System
- **Authors:** Daison Stallon, Ichrak Eben Zaid, Yolanda Vidal
- **Volume / pages:** Vol. 1, pp. 208–216
- **Topic tags:** cross-attention, attention, time-series, CNN
- **Relevance:** Applies a **multi-head cross-attention** mechanism over
  sequential time-series for fault diagnosis. Domain is wind energy, but
  it is a venue-local example of cross-attention on time-series signals.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

### 16. Domain-Decoupled Physics-informed Neural Networks with Closed-Form Gradients for Fast Model Learning of Dynamical Systems
- **Authors:** Henrik Krauss, Tim-Lukas Habich, Max Bartholdt, Thomas Seel, Moritz Schappler
- **Volume / pages:** Vol. 1, pp. 55–66
- **Topic tags:** continuous-time, PINN, dynamics, deep-learning
- **Relevance:** Decouples the **time domain** from a feedforward network
  to learn continuous-time state-space dynamics with closed-form
  gradients. Tangential but thematically aligned with explicit-time
  neural modelling (our continuous-time token).
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

---

## D. Visual / camera localization & odometry (mostly journal / paper-2)

### 17. Triplet Neural Networks for the Visual Localization of Mobile Robots
- **Authors:** Marcos Alfaro, Juan José Cabrera, Luis Miguel Jiménez, Óscar Reinoso, Luis Payá
- **Volume / pages:** Vol. 2, pp. 125–132
- **Topic tags:** visual-localization, CNN, metric-learning, triplet-loss, place-recognition, indoor, nearest-neighbour
- **Relevance:** Trains a triplet CNN on indoor panoramic images to learn
  descriptors, then localizes by **nearest-neighbour retrieval** in
  descriptor space — methodologically parallel to our learned-embedding +
  kNN positioning (WiFi-Net). Even though camera is a journal modality,
  the *method* (metric-learned embedding + NN localization) is squarely in
  scope for Related Work.
- **Scope:** `[in-scope:paper-1]`  ·  **Zotero:** `[not-in-zotero]`

### 18. Enhancing Visual Odometry Estimation Performance Using Image Enhancement Models
- **Authors:** Hajira Saleem, Reza Malekian, Hussan Munir
- **Volume / pages:** Vol. 1, pp. 293–300
- **Topic tags:** visual-odometry, deep-learning, camera, low-light
- **Relevance:** Improves visual odometry in low light with GAN/Retinex-style
  image enhancement front-ends. Camera/VO specific → belongs to the
  journal's camera story.
- **Scope:** `[hold-for:paper-2]`  ·  **Zotero:** `[not-in-zotero]`

### 19. Uncertainty Hypervolume in Point Feature-Based Visual Odometry
- **Authors:** InJun Mun, Sukhan Lee
- **Volume / pages:** Vol. 2, pp. 290–299
- **Topic tags:** visual-odometry, uncertainty, feature-selection
- **Relevance:** Derives an "uncertainty hypervolume" linking feature-point
  selection to visual-odometry uncertainty. Camera/VO specific →
  journal/paper-2 (could pair with our DPVO motion encoder discussion).
- **Scope:** `[hold-for:paper-2]`  ·  **Zotero:** `[not-in-zotero]`

### 20. A Vision Based System for Assisting Blind People at Indoor and Outdoor Exploration
- **Authors:** Raluca Didona Brehar, Sand Elena-Andreea
- **Volume / pages:** Vol. 2, pp. 54–65
- **Topic tags:** visual-localization, indoor-navigation, assistive, object-detection
- **Relevance:** Vision + ultrasonic assistive navigation for visually
  impaired users in indoor/outdoor spaces. Indoor-navigation context only.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

---

## E. Filtering / state estimation (classical-baseline contrast)

### 21. BVE + EKF: A Viewpoint Estimator for the Estimation of the Object's Position in the 3D Task Space Using Extended Kalman Filters
- **Authors:** Sandro Costa Magalhães, António Paulo Moreira, Filipe Neves dos Santos, Jorge Dias
- **Volume / pages:** Vol. 2, pp. 157–165
- **Topic tags:** Kalman-filtering, EKF, pose-estimation, vision
- **Relevance:** Uses an **EKF** to track and correct 3D object position
  from monocular observations (≈32 mm error). A clean classical-filter
  reference contrasting recursive Bayesian estimation with our learned
  temporal smoothing.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

### 22. Data-Driven Intrusion Detection in Vehicles: Integrating Unscented Kalman Filter (UKF) with Machine Learning
- **Authors:** Shuhao Bian, Milad Farsi, Nasser L. Azad, Chris Hobbs
- **Volume / pages:** Vol. 1, pp. 714–723
- **Topic tags:** Kalman-filtering, UKF, machine-learning, state-estimation
- **Relevance:** Couples a **UKF** for nonlinear state estimation with ML
  for attack detection. Off-topic application; usable only as a
  filter-plus-learning example.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

---

## F. Robot navigation / simulation context (Introduction motivation only)

### 23. Autonomous Forklift Navigation Inside a Cluttered Logistics Factory
- **Authors:** Eric Lucet, Antoine Lucazeau, Jason Chemin
- **Volume / pages:** Vol. 2, pp. 327–335
- **Topic tags:** robot-navigation, industrial, MPC, indoor
- **Relevance:** Full MPC-based navigation stack for an autonomous forklift
  in a real printing factory. Industrial indoor-robot motivation for the
  Introduction (why robust indoor localization matters in logistics).
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

### 24. Using Shapley Additive Explanations to Explain a Deep Reinforcement Learning Agent Controlling a Turtlebot3 for Autonomous Navigation
- **Authors:** Sindre Benjamin Remman, Anastasios M. Lekkas
- **Volume / pages:** Vol. 1, pp. 334–340
- **Topic tags:** deep-RL, robot-navigation, explainability, indoor, Conv1d
- **Relevance:** DRL navigation agent whose actor uses a **1D CNN** over
  LiDAR sequences; adds SHAP explainability. Context for learned indoor
  navigation; the Conv1d-over-sequence detail loosely echoes our CNN
  encoders.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

### 25. Modeling Sunlight in Gazebo for Vision-Based Applications Under Varying Light Conditions
- **Authors:** Ramir Sultanov, Ramil Safin, Edgar A. Martínez-García, Evgeni Magid
- **Volume / pages:** Vol. 1, pp. 519–526
- **Topic tags:** simulation, vision, sim-to-real
- **Relevance:** Models realistic lighting in a Gazebo simulator for
  vision tasks. Relevant as a citation defending *simulated sensor data*
  (our Webots pipeline) — supports the "controlled lab simulation" framing
  flagged in the scope risk register.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

### 26. Miniature Autonomous Vehicle Environment for Sim-to-Real Transfer in Reinforcement Learning
- **Authors:** Stephan Pareigis, Daniel Riege, Tim Tiedemann
- **Volume / pages:** Vol. 1, pp. 309–318
- **Topic tags:** simulation, sim-to-real, reinforcement-learning
- **Relevance:** Builds a miniature environment for sim-to-real RL transfer.
  Context for the sim-vs-real-validity question (Webots sim → real MSILN)
  that our paper must defend.
- **Scope:** `[context-only]`  ·  **Zotero:** `[not-in-zotero]`

---

## Summary table

| # | Short title | Vol/pp | Scope | Zotero |
|---|---|---|---|---|
| 1 | LCM clustering indoor positioning | 1 / 301–308 | in-scope:paper-1 | not-in-zotero |
| 2 | Building 2D maps with robots | 2 / 228–235 | in-scope:paper-1 | not-in-zotero |
| 3 | Dynamic position est. + flocking | 1 / 269–276 | context-only | not-in-zotero |
| 4 | Drone warehouse localization | 2 / 357–364 | context-only | not-in-zotero |
| 5 | Uncertainty-aware multimodal cam loc | 2 / 80–90 | in-scope:paper-1 | not-in-zotero |
| 6 | Multimodal 6D pallets (attn fusion) | 2 / 345–352 | in-scope:paper-1 | not-in-zotero |
| 7 | Modular multimodal MOT (Kalman) | 2 / 336–344 | in-scope:paper-1 | not-in-zotero |
| 8 | UAV-USV multi-rate async fusion | 1 / 545–554 | in-scope:paper-1 | not-in-zotero |
| 9 | Graph-attention lane change (attn+RNN) | 2 / 282–289 | in-scope:paper-1 | not-in-zotero |
| 10 | LiDAR multimodal power-line inspect | 2 / 197–204 | context-only | not-in-zotero |
| 11 | RoboMorph transformer dynamics | 2 / 149–156 | in-scope:paper-1 | not-in-zotero |
| 12 | NODE continuous-time dynamics | 2 / 205–211 | in-scope:paper-1 | not-in-zotero |
| 13 | Multi-step LSTM time series | 1 / 651–659 | in-scope:paper-1 | not-in-zotero |
| 14 | Vision Transformer defect detection | 2 / 36–45 | context-only | not-in-zotero |
| 15 | DFIG multihead cross-attn time-series | 1 / 208–216 | context-only | not-in-zotero |
| 16 | Domain-decoupled PINN continuous-time | 1 / 55–66 | context-only | not-in-zotero |
| 17 | Triplet NN visual localization (kNN) | 2 / 125–132 | in-scope:paper-1 | not-in-zotero |
| 18 | Visual odometry image enhancement | 1 / 293–300 | hold-for:paper-2 | not-in-zotero |
| 19 | Uncertainty hypervolume VO | 2 / 290–299 | hold-for:paper-2 | not-in-zotero |
| 20 | Vision system assisting blind people | 2 / 54–65 | context-only | not-in-zotero |
| 21 | BVE + EKF viewpoint estimator | 2 / 157–165 | context-only | not-in-zotero |
| 22 | UKF + ML intrusion detection | 1 / 714–723 | context-only | not-in-zotero |
| 23 | Autonomous forklift navigation | 2 / 327–335 | context-only | not-in-zotero |
| 24 | SHAP DRL Turtlebot3 navigation | 1 / 334–340 | context-only | not-in-zotero |
| 25 | Modeling sunlight in Gazebo | 1 / 519–526 | context-only | not-in-zotero |
| 26 | Miniature AV sim-to-real RL | 1 / 309–318 | context-only | not-in-zotero |

**Counts:** 26 papers — 11 `in-scope:paper-1`, 2 `hold-for:paper-2`,
13 `context-only`. 0 in Zotero.

## Recommended priority for the conference Related Work (§2)

Highest value (cite first): **#17 Triplet NN** (learned embedding + kNN
localization, mirrors our WiFi encoder), **#6 Multimodal 6D pallets** and
**#9 Graph-attention lane change** (attention as a multimodal/temporal
fusion operator), **#8 UAV-USV** (multi-rate asynchronous fusion), **#7
Modular MOT** (classical Kalman multimodal contrast), **#5 Uncertainty
multimodal cam loc** (learned multimodal localization). Architecture
framing: **#11 RoboMorph** (transformer in robotics), **#12 NODE** and
**#13 LSTM multi-step** (continuous-time / temporal sequence contrast).
Indoor-localization venue context: **#1 LCM** and **#2 2D maps**.

## Caveats

1. **No direct SOTA baseline in these proceedings.** ICINCO 2024 contains
   no WiFi-RSSI fingerprinting or IMU dead-reckoning paper that competes
   with our pipeline. The actual quantitative baselines (wlan_localization,
   RoNIN, UJIIndoorLoc) are from the repo/Zotero, not here. Use these
   ICINCO papers for *positioning within the community* and for the
   classical-vs-learned-fusion narrative, not for the results tables.
2. **Author names taken verbatim from the proceedings front-matter** as
   surfaced by NotebookLM; double-check spelling/diacritics against the
   PDF before they enter `refs.bib` (e.g. Drămnesc, Martínez-García,
   Báča, Kővári).
3. **Zotero coverage is zero for these papers.** All 26 are
   `not-in-zotero`; any chosen for citation must be added to the library /
   `refs.bib` manually (BibTeX key pattern `firstauthor2024word`). A
   library-wide "ICINCO" search returned no items, so there is no risk of
   silent duplicates.
4. **A few entries are borderline scope calls** (#5 camera-but-multimodal,
   #13 LSTM-but-process-control, #15 cross-attention-but-wind-energy). They
   are tagged by their *methodological* relevance to our fusion/attention/
   temporal themes; downgrade to `context-only` if §2 space is tight.
