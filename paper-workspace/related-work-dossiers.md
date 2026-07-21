# Related Work — Paper Dossiers (Phase 4)

**Purpose.** One compact, accurate capsule per candidate paper, so the writer characterises each
work correctly and cites the right one. Each capsule: `bibkey` · role · citation · method ·
modalities · time-handling · robustness · datasets · *grounded* headline · limitation ·
diff-vs-ours · `source_id`. Full quotes: `related-work-raw/<group>.md`. Metadata fixes: see
`related-work-evidence.md` §Integrity flags (F4).

Roles: **pillar** (we build on it) · **benchmark** (dataset/baseline we use) · **competitor**
(positioned against) · **context** (survey/background). ⚑ = must-cite precedent (do not claim as ours).

---

## Group W — WiFi fingerprinting

**`bahl2000radar`** · benchmark · Bahl & Padmanabhan, RADAR, INFOCOM 2000. Classical RSSI
nearest-neighbour-in-signal-space (kNN). WiFi RSSI only; single-scan; no time/robustness.
Grounded: median 2.94 m (own testbed); ~9.21 m MAE on UJI per others. *Diff:* non-learned baseline. · e7ac47fb

**`youssef2005horus`** · benchmark · Youssef & Agrawala, Horus, MobiSys 2005. Probabilistic RSSI
fingerprinting + clustering. WiFi RSSI only. Grounded: "<0.6 m on average" (own small testbed —
not UJI-scale; do not compare to our numbers). · a7a2adb1

**`wang2015deepfi`** · benchmark · Wang et al., DeepFi, WCNC 2015. Deep CSI fingerprinting (greedy
pretrain + RBF). WiFi CSI; controlled rooms. Grounded: ~0.95 m (living room, 1 AP). *Diff:* CSI not
RSSI, no attention/time/fusion. · 4d28b1dc

**`torressospedra2014ujiindoorloc`** · benchmark (THE WiFi dataset) · Torres-Sospedra et al.,
UJIIndoorLoc, IPIN 2014. 21,049 RSSI samples, 520 WAPs, 3 buildings; **validation 4 months after
training** (cross-session). WiFi RSSI only. Grounded: 1NN = **7.9 m**, 89.92% success. *Diff:* the
WiFi half of our setup + the cross-session motivation; no IMU/async. · 46b80222

**`song2019cnnloc`** · benchmark/competitor · Song et al., CNNLoc, IEEE Access/UIC 2019. SAE + 1D-CNN,
multi-building/floor. WiFi RSSI. Grounded: **11.78 m on UJI**. *Diff:* CNN (no attention), single
modality, static. · 8739f6cf

**`tiku2022anvil`** · competitor · Tiku et al., ANVIL, 2022. Multi-head attention, device-invariant,
on-device. WiFi RSSI. AP-dropout augmentation + −100 dB impute. Grounded: "up to 35% improvement"
(⚑ relative). XSESS = cross-**device**. *Diff:* attention yes, but single modality, no Δt, AP-dropout
≠ modality dropout. · c114e66e

**`zhang2022tips`** · competitor · Zhang et al., TIPS, IEEE Access 2022. GPT-style decoder transformer;
routes="sentences". WiFi **CSI + DoA** (both RF). Grounded: "down to 20 cm" (sim+testbed). *Diff:*
transformer but no IMU, discrete autoregressive time, no modality dropout, no cross-session. · 07285e43

**`zhang2023aarescnn`** · competitor · Zhang et al., 2023. Attention-augmented residual CNN for CSI +
**separate** universal IMU tracking (plug-and-play). WiFi CSI + IMU (decoupled stages). Grounded:
"8%–48%" (⚑ relative). *Diff:* fuses WiFi+IMU but in two branches, discrete time, no dropout. · 52b5f512

**`ott2024radiofm`** · competitor · Ott et al., Radio Foundation Models, 2024. Self-supervised
pretrained transformer on 5G CIR (mask-reconstruct pretext). 5G CIR only. Grounded: CE90 0.398 m
@100k. XSESS = cross-**site**. *Diff:* single RF modality, discrete seq, mask-pretext ≠ inference
modality dropout. · 5aa39fa5

**`bhatia2025locaris`** · competitor · Bhatia et al., Locaris, 2025. Decoder-only LLM, token-per-AP,
schema-free. WiFi **FTM + RSSI** (no IMU). Native missing-modality ("whichever modality is available
… without placeholders"). Grounded: 0.88 m @3% target data. XSESS = cross-environment few-shot.
*Diff:* closest in-WiFi flexible-modality, but no IMU, token-per-reading ≠ Δt, decoder-only ≠
perm-invariant set. · c4f1526c

**`abdullah2025ris`** · competitor · Abdullah et al., 2025. Multi-head transformer, 8 tokens
(5 CSI+RSS+RIS+[CLS]), unified block. CSI+RSS+RIS+geom (all RF). Grounded: 0.31 m (sim only). *Diff:*
unified multimodal transformer (architecturally close!) but all RF (no inertial), simulation, no Δt,
static ablation, no cross-session. · 1e099b7b

**`nguyen2024aat`** · competitor (in-domain WiFi-encoder relative) · Nguyen et al., All-embracing
Transformers, Pervasive Mob. Comput. 2024. Transformer over RSS with **Anchor2Vec** tokeniser
(k=64 tokens, d=128) — *our WiFiNet was renamed from Anchor2Vec*. WiFi RSS only; 100 dB impute.
Grounded: **8.16 m on UJI** (vs RADAR 9.21, CNNLoc 11.78). *Diff:* same WiFi-encoder lineage but no
continuous-time, no temporal fusion, no IMU. · d4577fc7

**`nasir2024hytra`** · competitor · Nasir et al., HyTra, 2024. Encoder-only transformer, WAPs as
learnable embeddings; hierarchical building→floor→room. WiFi RSS only. Grounded: 96.7% floor-class
acc (UJI 4-month split). *Diff:* transformer + cross-time split, but single modality, classification,
no Δt/fusion/dropout. · be87a391

**`turgut2024xai`** · competitor · Turgut & Kakisim, FGCS 2024. SAE + particle filter + CNN-LSTM,
explainable (LIME/SHAP). WiFi RSSI; classification. Grounded: 95.33% acc on UJI. *Diff:* no attention,
single modality, no Δt/dropout/cross-session. · 0d98906d

**`aristorenas2025set`** · competitor (closest to pillar ii in WiFi) · Aristorenas, Stanford
**preprint** 2025. Set Transformer over a scan's unordered (BSSID, RSSI) set. WiFi RSSI only;
single scan (no time). Grounded: ST 3.82 m on E1 — **beaten by a plain LSTM (2.23 m)**. *Diff:*
closest architectural cousin to (ii) but single-modality, no temporal fusion, no dropout/cross-session;
multimodal = future work. · f6417660

**`zhou2025conformal`** · context (relates to our uncertainty module) · Zhou et al., 2025. Conformal
prediction wrapper on CNN classifiers; relies on exchangeability. WiFi RSSI. *Diff:* same
exchangeability caveat as our ConformalPosition; no fusion/async/IMU. · 07baf76d

**Surveys (context):** `feng2022survey` (072953c0) DL-for-WiFi; `hechan2016survey` (66be4066) — key:
fingerprints drift, recalibration costly; `martinfrechina2025review` (d3df2e7e) — open frontier =
DL-based multimetric fusion + lightweight adaptive models.

---

## Group I — Inertial / IMU

**`nguyen2025imot`** · competitor (**CLOSEST OVERALL**) · Nguyen et al., iMoT, AAAI **2025**.
Transformer enc-dec; encoder self-attn over accel/gyro tokens, decoder cross-attn with learnable
"query motion particles". **Inertial-only** (accel+gyro). Fixed 1 s windows; token dim per rate
(no Δt). No missing-modality mechanism. Grounded: RoNIN dynamic **unseen ATE 5.31 m**. *Diff:* closest
in spirit but no WiFi, fixed-rate, enc/dec branches, no dropout, cross-subject (not cross-session
WiFi). · 7b6e4a06

**`yan2019ronin`** · benchmark (our ResNet1D baseline) · Yan/Herath/Furukawa, RoNIN, arXiv 2019 /
ICRA 2020. ResNet-18-1D / LSTM / TCN regress 2D velocity → integrate. IMU-only (200×6). Grounded:
RoNIN **unseen ResNet ATE 5.14 / RTE 4.37**. *Diff:* our IMU baseline; no fusion/Δt/robustness/WiFi. · 9b151e20

**`rao2022ctin`** · competitor · Rao et al., CTIN, AAAI 2022. ResNet encoder + local/global self-attn,
transformer decoder fuses temporal. Inertial-only (6D). Sliding window m=200. Grounded: CTIN-dataset
ATE 1.28 m (⚑ relative gains elsewhere). *Diff:* hybrid ResNet+transformer (not one set block),
single modality, no Δt/dropout. · 88da0ef4

**`brotchie2023riot`** · competitor · Brotchie et al., RIOT, Sensors 2023. Self-attn enc-dec, recursive.
Inertial-only 9D (+magnetometer). Synced 100 Hz. Grounded: ATE 0.0865 m (OxIOD). *Diff:* transformer
but inertial-only, fixed-rate, no WiFi/fusion/dropout. · 13fe55b9

**`zheng2024neurit`** · competitor · Zheng et al., NeurIT, 2024 (venue not grounded). Time-Frequency
Block-recurrent Transformer. Inertial-only 9D (+mag). Fixed window. Grounded: ~1 m / 300 m;
"+48.21% vs best baseline" (⚑ relative). XSESS = cross-building (train A, test B/C). *Diff:* strong
transformer + cross-env but inertial-only, fixed-rate, no Δt/WiFi/dropout. · 01c12ae2

**`zeinali2022imunet`** · competitor · Zeinali et al., IMUNet, 2022. CNN (MobileResNet), edge-efficient.
Inertial-only (6×200). No attention. Grounded: own-dataset ATE 2.59 m; RoNIN 3.52 m. *Diff:* CNN, no
attention, orthogonal to all 3 contributions. · dcc6888f

**`herath2022niloc`** · competitor · Herath et al., NILoc, CVPR 2022. RoNIN-ResNet→velocity, two-branch
transformer→location likelihood; **scene-specific**. IMU-only. Distance-based resampling. *Notable:*
explicitly motivates "WiFi once in a few minutes + IMU in between" — our exact scenario, but it does
NOT fuse WiFi. Grounded: success-rate metrics (no ATE). *Diff:* per-scene branches, no Δt, no WiFi
fusion, no dropout. · 506570fb

**`chen2021rninvio`** · competitor (graceful-degradation analog) · Chen et al., RNIN-VIO, ISMAR 2021.
ResNet+LSTM inertial net + EKF tight-coupling of vision+IMU. **Vision+IMU**. Linear interpolation to
100 Hz. **Degrades to IMU-only** when visual drops (via EKF, not dropout). Grounded: IDOL ATE
2.71/3.62 etc. *Diff:* fuses 2 modalities + graceful, but EKF (not one attn block), fixed-rate,
vision≠WiFi, hand-engineered robustness. · 1b3ac22c

**`jayanth2024eqnio`** · competitor · Jayanth et al., EqNIO (ICLR-era preprint). O(2)-equivariant
canonicalisation wrapper around TLIO/RoNIN (MLP+conv, **no attention**). Inertial-only. Grounded:
cross-dataset gains (⚑ relative). *Diff:* equivariance generalisation route, not transformer/Δt/fusion. · c90706f3

**`chen2018ionet`** · pillar · Chen et al., IONet, AAAI 2018. First DNN inertial odometry (Bi-LSTM,
polar displacement). IMU-only. Grounded: "~2 m max error, 90% of time". *Diff:* foundational, LSTM
not attention, no fusion. · 4eb0c9d6

**`liu2020tlio`** · pillar · Liu et al., TLIO, RA-L 2020. ResNet displacement+uncertainty → stochastic-
cloning EKF. IMU-only. Grounded: "−27%/−33% yaw/pos drift vs RoNIN". *Diff:* network+EKF, single
modality, no Δt/attention. · 50b563d3

**`yan2018ridi`** (benchmark, ECCV 2018, e8c59e76) · **`chen2018oxiod`** (benchmark dataset,
arXiv 1809.07491, e20813a5) · **`sun2021idol`** (benchmark, AAAI 2021, 9a712c1a) — inertial datasets
used by competitors; IMU-only; cite as benchmark provenance if needed.

**`cohenklein2024survey`** · context · Cohen & Klein, *Inertial Navigation Meets Deep Learning*,
Results in Eng. 2024. ⚑ *Useful:* notes a Set-Transformer ("ST-BeamsNet") for sensor-outage recovery
in **marine DVL** — set-transformers-for-missing-modality exist OUTSIDE indoor localization, which
sharpens that nobody applied it to WiFi+IMU. · cd200533

---

## Group F — Multimodal fusion (localization) — most on-problem

**`yu2022multimodal`** · competitor · Yu et al., Multi-Modal Recurrent Fusion, MERL 2022. Multi-stream
LSTM + learned per-modality importance weights. WiFi(RSSI+CSI)+IMU+UWB. Discrete T steps. Grounded:
0.06 m mean (SPAWC2021, random split). *Diff:* per-modality LSTM branches, no Δt/perm-invariance, no
dropout, no cross-session. (⚑ not plain WiFi+IMU — adds CSI+UWB.) · 53d5d1d1

**`zhou2024wioekf`** · competitor (closest cross-session WiFi+IMU) · Zhou et al., WIO-EKF, IEEE 2024.
CDAELoc (WiFi) + DbDIO (IMU) fused by **EKF**. WiFi+IMU. Fixed 1 s windows. CDAE mask-noise robust.
Grounded: APE **2.53 m**; **cross-day (10 days apart)**. *Diff:* EKF + branches, no transformer/Δt,
AP-mask ≠ modality dropout. · de3074ce

**`yang2025wimu`** · competitor · Yang et al., WiMU, MobiSys '25 demo. GNN/VGAE WiFi + PDR fused by
**particle filter**. WiFi RSSI+IMU. Grounded: 4.6 m (campus). *Diff:* filter-based branches, no
attention/Δt/dropout/cross-session. · 31231955

**`hua2023smartfps`** · competitor · Hua et al., SmartFPS, Front. Neurorobot. 2023. LSTM(inertial)+
CNN(wireless)+**attention sub-layer**+LSTM decoder; GAN transfer. **Bluetooth+IMU**. Down-sampled 1 s
windows. Grounded: 0.506 m. *Diff:* attention inside branches (hybrid), fixed windows, BT not WiFi,
GAN transfer ≠ dropout. (⚑ BT+IMU.) · 6b420277

**`herath2021fusiondhl`** · competitor · Herath et al., Fusion-DHL, ICRA 2021. NLS optimization
(RoNIN trajectory vs sparse WiFi/FLP) + CNN floorplan refine. WiFi(FLP)+IMU+floorplan. Grounded:
RMSE ~5 m (vs ~12 m prior). XSESS = cross-building. *Diff:* optimization+CNN (not attention), needs
floorplan, staged, no Δt/dropout. · 583c1128

**`wei2021sensorfusion`** · competitor (closest WiFi+IMU robustness) · Wei et al., Sensors 2021.
LSTM(inertial)+DNN(WiFi) concat → FC. WiFi RSS+inertial. Linear interp, WiFi→100 ms. **Missing-WiFi
via NULL vector** (−100 dBm) → inertial. Grounded: **1.9 m median**. *Diff:* concat branches, no
attention, resampling, NULL-flag ≠ learned dropout, random split. · 75d15e66

**`zhang2021lstm`** · competitor · Zhang et al., IEEE IoT-J 2021. Single LSTM over displacement
features. WiFi RSS+PDR. Unify to 20 Hz, interpolate + moving-average (stale-RSS). Grounded: 0.42 m
best. XSESS = cross-user. *Diff:* LSTM not set-transformer, resampling, smoothing ≠ instant dropout,
cross-user not cross-session. · 5fcd4079

**`wang2024damloc`** · competitor · Wang et al., DamLoc, FGCS 2024. Multi-branch CNN + **attention** +
context modality. **Magnetic+BLE+context** (not WiFi/IMU). Piecewise interp. Grounded: 0.30–1.38 m.
*Diff:* attention inside branches, wrong modalities, interp, context-zeroing ≠ learned dropout, single
env. (⚑ Magnetic+BLE.) · 94456ab2

**`lajoie2023peoplex`** · competitor (closest async-without-resampling) · Lajoie et al., PEOPLEx,
2023. Nonlinear **factor-graph** optimization; IMU backbone + opportunistic UWB/BLE/WiFi. Ingests
async "as available"; IMU-only fallback. Grounded: RMSE **1.05 m** (vs RoNIN 2.88). *Diff:* classical
optimization (not learned attention/Δt), opportunistic = architectural not learned dropout, no
cross-session. · 4d20d630

**`chen2015kalman`** · competitor (classical) · Chen et al., Sensors 2015. Linear Kalman filter:
WiFi (obs) + PDR (state) + landmarks (drift reset). Grounded: ~1 m. *Diff:* classical KF, linear, no
async/learning/cross-session. · 1c454340

**`geneva2018async`** · competitor (async precedent) · Geneva et al., 2018. Factor-graph (iSAM2);
async measurements **aligned to fixed states by analytical interpolation/extrapolation**.
LIDAR+stereo+RTK GPS (driving, not indoor). Grounded: GPS-denied RMSE 0.71 m. *Diff:* the canonical
"async fusion" still resamples-to-states; classical, wrong domain, no dropout — **key contrast**: our
learned Δt needs no alignment. · 59d94997

**`neverova2014moddrop`** ⚑ · pillar/precedent (MUST CITE) · Neverova et al., ModDrop, TPAMI
(2014/2016). Random Bernoulli dropping of **whole modality channels** in fusion training → robust to
missing channels. Gesture recognition (RGB/depth/pose/audio). *Relation:* the explicit ancestor of
our modality-dropout; we extend with **per-instant (token) dropout** + WiFi+IMU + cross-session. · d8cc9acd

**`silva2023dataset`** · benchmark (dataset) · Silva et al., Data 2023. Industrial **WiFi (~0.6 Hz)
+ 2 IMU (20 Hz) + odometry (50 Hz)** trolley dataset — real multi-rate async, AMR analog of our
TIAGO++. Grounded: DR 8.25 m, WiFi-FP 2.19 m. · b0e54375

**`abdalla2025dataset`** · benchmark (dataset) · Abdalla et al., Data in Brief **2025**. WiFi RSS
(10 Hz)+inertial(~5 Hz)+CCTV. Resampled+NN-aligned (the op our Δt removes). Headline *not grounded*
(only "preliminary trials"). · 1bb05d0a

**Surveys (context):** `wangahmad2025survey` (51755f40) AI-for-AMR — does NOT treat async fusion as a
solved gap; `lukasik2024survey` (c3c7d669) image-based multimodal — multimodal beats unimodal, attention
emerging but no unified async set-transformer.

---

## Group A — Attention / transformer / set

**`vaswani2017transformer`** · pillar · Vaswani et al., NeurIPS 2017. Self-attention; **sinusoidal
positional encoding assumes equidistant positions** — the assumption we break for async Δt. · f7cf37ed

**`lee2019settransformer`** · pillar (our ii) · Lee et al., ICML 2019. Permutation-invariant
attention (ISAB + PMA); universal approximator of set functions. *We instantiate it over (modality,
time) tokens.* Set Transformer itself has no time notion / no async-fusion application. · a85a3ae2

**`zaheer2017deepsets`** · pillar · Zaheer et al., NeurIPS 2017. Theory: perm-invariant set functions
= ρ(Σφ(x)). Justifies the unordered-set view. Sum-pooling (no interactions) — we use attention. · 2e61db11

**`jaegle2021perceiver`** ⚑ · pillar/precedent · Jaegle et al., ICML 2021. Cross-attention into a
latent bottleneck; modality-agnostic. ⚑ Uses **"video dropout"** (whole-modality dropout, 30%) —
second modality-dropout precedent + cross-attention-readout precedent. Not async-Δt, not localization. · 44dce69c

**`jaegle2022perceiverio`** · pillar/context · Jaegle et al., ICLR 2022. Query-based structured-output
readout (parallels our cross-attention readout). General-purpose, not localization. · a66f7a26

**`kaygusuz2022aftvo`** · competitor (closest async+attention) · Kaygusuz et al., AFT-VO, 2022.
Per-camera MDN poses → transformer fuses; **time binned ("Discretiser")**, no resampling.
**Multi-view cameras only** (IMU = explicit future work). Grounded: nuScenes RPE 0.031 (day).
*Diff:* single-modality late-fusion of predictions, binned time ≠ real-valued Δt, no modality dropout,
outdoor VO. · 8b22a8f8

**`cohen2024akit`** · competitor · Cohen & Klein, A-KIT, 2024. Set-transformer **regresses EKF
process-noise covariance**; EKF does the fusion. IMU/INS+DVL (underwater). Fixed 1 s/100-sample
windows; positional encoding removed. Grounded: ">49.5% over EKF" (⚑ relative). *Diff:* transformer
≠ the fusion block, no Δt, no dropout, single-session, underwater. · fb531c11

**`xiao2024effloc`** · competitor · Xiao et al., EffLoc, 2024. Efficient ViT, single-image 6-DoF
relocalization. Single camera; no time/fusion. Grounded: 7.58 m (RobotCar LOOP1). XSESS = cross-day/
weather (single-modality). *Diff:* transformer-for-localization exists, but opposite of our problem. · 1d12937b

**`lin2025scmpr`** · competitor · Lin & Evans, SCM-PR, 2025. Cross-modal semantic attention; RGB→LiDAR
place recognition (Recall@1, not (x,y)). Grounded: 62.58% R@1 (KITTI). *Diff:* cross-modal only (no
cross-time), retrieval not regression, no Δt/dropout. · 5cd52555

**`diazguerra2023pirnn`** · competitor · Diaz-Guerra et al., PI-RNN, **Forum Acusticum 2023**.
Permutation-invariant RNN (internal multi-head attn) for sound-source tracking. Acoustic; perm-invariant
over **sources** not (modality,time). *Diff:* RNN-based, single-domain, no Δt/dropout. · 3b94d89f

---

## Group C — Continuous-time / async / irregularly-sampled (pillars for i)

**`shukla2021mtan`** ⚑ · pillar (our Δt ancestor) · Shukla & Marlin, mTAN, ICLR 2021. Learned **sin+linear
time embedding** as attention keys/queries; interpolates to a fixed reference grid. Clinical + HAR.
*Diff:* same sin+linear primitive but used for interpolation-to-grid, not per-token Δt; not localization. · cfe566d2

**`kazemi2019time2vec`** ⚑ · pillar (time primitive) · Kazemi et al., Time2Vec, arXiv 2019. Learnable
linear+sine vector embedding of scalar time; drop-in. Event/recsys/audio. *Diff:* absolute scalar time,
not elapsed Δt in a fusion set-transformer; not localization. · 6155e781

**`chen2018neuralode`** · pillar (ODE we avoid) · Chen et al., Neural ODE, NeurIPS 2018. Continuous-depth
via black-box ODE solver. Generic. *Diff:* we avoid the solver entirely. · 107ff554

**`rubanova2019latentode`** · pillar · Rubanova et al., Latent ODE / ODE-RNN, NeurIPS 2019. ODE between
obs + RNN update. Clinical/physics/HAR. *Diff:* sequential ODE+RNN; we use parallel set-attention, no ODE. · cbcd353a

**`kidger2020neuralcde`** · related · Kidger et al., Neural CDE, NeurIPS 2020. Data-controlled
differential equation for irregular series. Generic/clinical. *Diff:* CDE solver vs our learned Δt token. · 5e708532

**`debrouwer2019gruodebayes`** · related · De Brouwer et al., GRU-ODE-Bayes, NeurIPS 2019. Learnable
continuous-time filter (ODE + Bayes update). Clinical/climate. *Diff:* a learned filter; we don't filter
or solve ODEs. · 793ea3c3

**`chen2023contiformer`** · competitor (continuous-time transformer) · Chen et al., ContiFormer,
NeurIPS 2023. Neural-ODE dynamics **inside attention** (CT-MHA). Generic/clinical. ⚑ Grounded **cost**:
"substantial time and GPU memory overhead … ~4× slower at length 1000". *Diff:* the heavy ODE-in-attention
route; our Δt is the lightweight alternative; not localization. · d4fc0e36

**`horn2020seft`** · competitor (closest architecture, (i)+(ii)) · Horn et al., SeFT, ICML 2020.
Perm-invariant set function over **(time, value, modality) triplets** + trigonometric time encoding;
DeepSets aggregation. Clinical only. *Diff:* same triplet/set idea, but FIXED time encoding +
sum-decomposition, no missing-modality test, not localization. · 2c3d6b02

**`che2018grud`** · related · Che et al., GRU-D, Sci. Rep. 2018. GRU with trainable **decay** over time
gaps. Clinical. *Diff:* recurrent decay heuristic vs our Δt-in-attention. · 974ad9a2

**`shukla2019ipnets`** · related · Shukla & Marlin, IP-Nets, ICLR 2019. RBF **interpolation to a regular
grid** + GRU. Clinical/gesture. *Diff:* interpolates (the resampling we avoid). · 586bbf61

**`tipirneni2022strats`** · competitor (closest architecture, (i)+(ii)) · Tipirneni & Reddy, STraTS,
ACM TKDD 2022. Transformer over (time,variable,value) triplets; **Continuous Value Embedding** of time,
no discretization/ODE. Clinical only. *Diff:* architecturally very close, but absolute time/value not
elapsed Δt, no missing-modality test, not localization. · a7665dca

**`zhang2022raindrop`** · competitor (closest robustness, (i)+(iii)) · Zhang et al., Raindrop, ICLR
2022. Sensor-dependency **graph** message passing at irregular times. Clinical/HAR. ⚑ **Leave-sensors-
out** (up to 50%) + cross-group generalization. *Diff:* multi-stage graph (not one block), classification,
not localization. · a068da78

**`shou2024dgode`** · related · Shou et al., DGODE, 2024. Graph-Neural-ODE for multimodal emotion
(text/audio/video); tests modality subsets. *Diff:* ODE/graph, emotion not localization. · 92ad96ff

**`feng2023kfnnreview`** · context · Feng et al., 2023. Review of Kalman-filter + NN hybrid state
estimation. Classical contrast to attention-based temporal fusion. · eba958aa

**`eang2024dnnekf`** · context (the only localization-applied here) · Eang & Lee, Sensors 2024.
MLP-refined EKF for **UWB** localization; fixed-rate. Grounded: 68.06 mm. *Diff:* UWB (active beacons)
not WiFi RSSI, MLP+EKF not attention, no async/Δt/dropout/cross-session. · 369ed09c
