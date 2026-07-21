# Related Work — Evidence Matrix (Phase 4 backbone)

**Purpose.** This is the traceability spine for the Related Work section. Every load-bearing
sentence the writer puts in §2 must map to a claim ID below (E#). Each claim carries: the
fact, the supporting paper(s) by `bibkey`, a short verbatim quote, the `source_id`, and which
gap/contribution it supports. **Full quotes + per-paper capsules live in
`paper-workspace/related-work-raw/<group>.md`** — this file is the curated, de-duplicated index.

**Traceability chain:** prose sentence → claim ID (E#) → `bibkey` + quote + `source_id` →
raw file (`related-work-raw/<group>.md`) → NotebookLM source PDF.

**Grounding status.** All 5 group notebooks + 3 adversarial refuters completed. Every quote
below was returned by NotebookLM with a source citation (verified during the gather pass).
Numbers that could not be quoted are marked *not grounded* and must NOT be cited.

Groups: **W** = WiFi fingerprinting · **I** = Inertial/IMU · **F** = Multimodal fusion ·
**A** = Attention/transformer/set · **C** = Continuous-time/async.

---

## Gap 1 — Continuous-time Δt (our contribution i)

> *We encode each observation's real-valued elapsed Δt with a learned sinusoidal embedding added
> per token; no resampling, no ODE solver.*

| ID | Claim | bibkey · source_id | Quote (short) | Grp |
|----|-------|--------------------|---------------|-----|
| E1 | Standard transformer positional encoding assumes **equidistant** token positions — breaks for async multi-rate sensors. | `vaswani2017transformer` · f7cf37ed; restated by `kaygusuz2022aftvo` · 8b22a8f8 | "they assume the consecutive items to be equidistant in the time domain … this is not applicable … multiple asynchronous sources" | A |
| E2 | The closest async+attention competitor handles time by **binning/quantising** continuous timestamps, not real-valued Δt. | `kaygusuz2022aftvo` · 8b22a8f8 | "we propose to discretise the continuous time domain into bins" | A |
| E3 | The dominant WiFi+IMU fusion practice **resamples both streams onto a fixed grid / fixed windows**. | `wei2021sensorfusion` · 75d15e66; `zhang2021lstm` · 5fcd4079; `zhou2024wioekf` · de3074ce | "we adjust the WiFi scan rate … to every 100 ms"; "the unified sampling frequency is set to 20Hz"; "time window is 200 (i.e., 1 s)" | F |
| E4 | Even the canonical **asynchronous-fusion precedent** still aligns measurements to fixed states by interpolation/extrapolation. | `geneva2018async` · 59d94997 | "we interpolate between two sequential 3D pose measurements to a given state timestamp" | F |
| E5 | Even recent WiFi+inertial **datasets ship pre-resampled** with nearest-neighbour alignment — the op our Δt removes. | `abdalla2025dataset` · 1bb05d0a | "All IMU streams are resampled to a consistent 5 Hz grid … index-based nearest-neighbour alignment" | F |
| E6 | The **learned-time-embedding ancestors** of our Δt (sin+linear of time) exist but were used for interpolation-to-grid or absolute time, not per-token async fusion. | `shukla2021mtan` · cfe566d2; `kazemi2019time2vec` · 6155e781 | mTAN: "learn an embedding of continuous time values … attention"; Time2Vec: "t2v(τ)[i] = … F=sine" | C |
| E7 | The **continuous-time-transformer** route puts an **ODE solver inside attention** and documents heavy cost (~4× slower at length 1000). | `chen2023contiformer` · d4fc0e36 | "substantial time and GPU memory overhead … approximately four times slower as the input length extends to 1000" | C |
| E8 | The pure **ODE/CDE** family (Neural ODE, Latent ODE, Neural CDE, GRU-ODE-Bayes) models irregular time via a solver — the machinery we avoid. | `chen2018neuralode` · 107ff554; `rubanova2019latentode` · cbcd353a; `kidger2020neuralcde` · 5e708532; `debrouwer2019gruodebayes` · 793ea3c3 | "black-box differential equation solver"; "h′=ODESolve(f,h,(t,t))" | C |
| E9 | Inertial methods bake the **rate into the window/token dimension** (fixed 1 s windows; token dim 100@100 Hz / 200@200 Hz). | `nguyen2025imot` · 7b6e4a06; `yan2019ronin` · 9b151e20 | "the token dimension is set to 100 … 100 Hz and to 200 … 200 Hz"; "IMU data from frame i−200 to i as a 200×6 tensor" | I |

---

## Gap 2 — Single unified permutation-invariant set-transformer (our contribution ii)

> *One self-attention block performs cross-modal AND cross-time fusion over the set of
> (modality, time) tokens — no per-modality branches.*

| ID | Claim | bibkey · source_id | Quote (short) | Grp |
|----|-------|--------------------|---------------|-----|
| E10 | The **Set Transformer** is a permutation-invariant attention architecture and universal approximator of set functions — the pillar for (ii). | `lee2019settransformer` · a85a3ae2 | "Proposition 1. The Set Transformer is permutation invariant." | A |
| E11 | **Deep Sets** gives the theory that permutation-invariant set functions decompose as ρ(Σφ(x)) — justifies treating observations as an unordered set. | `zaheer2017deepsets` · 2e61db11 | "invariant to the permutation … iff it can be decomposed in the form ρ(Σ φ(x))" | A |
| E12 | The dominant fusion topology in localization is **per-modality branches** combined late (concat / EKF / particle filter / weighted hidden states), not one unified block. | `wei2021sensorfusion` · 75d15e66; `yu2022multimodal` · 53d5d1d1; `zhou2024wioekf` · de3074ce | "two parallel single-modality feature extractors … merge latent features"; "multi-stream recurrent fusion" | F,I |
| E13 | Where **attention appears in localization fusion, it is a sub-layer inside branches**, not the whole fusion. | `hua2023smartfps` · 6b420277; `wang2024damloc` · 94456ab2 | "(1) inertial encoder…LSTM; (2) wireless encoder…CNN; (3) attention layer; (4) fusion decoder…LSTM" | F |
| E14 | The few **WiFi+IMU papers keep modalities in separate stages/branches** (e.g. CSI positioning net then IMU plug-and-play tracking). | `zhang2023aarescnn` · 52b5f512 | "PnP to incorporate the IMU measurements into the tracking system without retraining" | W |
| E15 | The **set-transformer used in IMU navigation (A-KIT) only regresses EKF noise covariance** — the EKF, not the transformer, does the fusion. | `cohen2024akit` · fb531c11 | "Built upon a set-transformer network, A-KIT is designed for … regression of the process noise covariance matrix" | A |
| E16 | The closest **async fusion transformer (AFT-VO)** does **late fusion of per-camera pose predictions**, single modality. | `kaygusuz2022aftvo` · 8b22a8f8 | "a Mixture Density Network … for every camera … Then a … transformer-based fusion module … combines these asynchronous pose estimations" | A |
| E17 | The closest **perm-invariant set-transformer over RSSI (Aristorenas)** is **single-modality, single-scan**, no cross-time. | `aristorenas2025set` · f6417660 | "each RSSI set is processed individually (batch size = 1) … avoiding … padding or masking"; multimodal listed as future work | W |
| E18 | The **architectural ancestors that DO realize a single set/triplet transformer with continuous time (SeFT, STraTS)** are clinical, not localization. | `horn2020seft` · 2c3d6b02; `tipirneni2022strats` · a7665dca | SeFT: "classifying time series as classifying a set of observations"; STraTS: "set of observation triplets" | C |

---

## Gap 3 — Async robustness via modality + instant dropout, shown cross-session (our contribution iii)

> *Train-time modality-dropout (0.4) + instant-dropout (0.45) → graceful degradation under
> missing/stale sensors, validated by real-world cross-session generalization.*

| ID | Claim | bibkey · source_id | Quote (short) | Grp |
|----|-------|--------------------|---------------|-----|
| E19 | **ModDrop is the explicit ancestor** of our modality-dropout: random Bernoulli dropping of whole modality channels for missing-signal robustness. *(MUST CITE.)* | `neverova2014moddrop` · d8cc9acd | "random dropping of separate channels (dubbed ModDrop) … robustness … to missing signals in one or several channels" | F |
| E20 | **Perceiver also uses whole-modality dropout** ("video dropout") — a second modality-dropout precedent. *(MUST CITE.)* | `jaegle2021perceiver` · 44dce69c | "video dropout — entirely zeroing out the video stream during training … 30% probability" | A |
| E21 | Localization-side robustness is mostly **hand-set / filter-based** (NULL vector, moving-average, context-zeroing, opportunistic fallback), not learned per-token dropout. | `wei2021sensorfusion` · 75d15e66; `wang2024damloc` · 94456ab2 | "the WiFi input is a vector with all components value of 0 (−100 dBm)"; context "determined as (0,0)" | F |
| E22 | The only graceful-degradation-to-fewer-sensors example among inertial fusers is **EKF + outlier rejection (RNIN-VIO)**, not learned dropout. | `chen2021rninvio` · 1b3ac22c | "visual constraints can be removed at any time, and state estimation … only based on IMU measurements" | I |
| E23 | The one **leave-sensors-out robustness protocol** in the continuous-time family (Raindrop) uses a sensor-dependency **graph** for clinical/HAR classification — multi-stage, not a unified block, not localization. | `zhang2022raindrop` · a068da78 | "test whether RAINDROP can achieve good performance when a subset of sensors are completely missing" | C |
| E24 | Most WiFi+IMU fusion papers evaluate with **random or cross-user/device splits**, not cross-session. | `yu2022multimodal` · 53d5d1d1; `wei2021sensorfusion` · 75d15e66 | "80/10/10 splitting"; "65%, 25% and 10% for training, validation and testing" | F |
| E25 | The one **cross-day WiFi+IMU result** is EKF-based with per-modality branches (WIO-EKF, 10-day gap). | `zhou2024wioekf` · de3074ce | "time interval between data collection for the training and test sets is ten days" | F |
| E26 | The canonical WiFi benchmark already enforces a **cross-session split** (validation 4 months after training) — motivates (iii). | `torressospedra2014ujiindoorloc` · 46b80222 | "Validation … samples … 4 months after Training ones" | W |
| E27 | WiFi fingerprints **drift with environment/time**, forcing recalibration — the core problem (iii) targets. | `hechan2016survey` · 66be4066 | "Wi-Fi signals may change … another costly site survey may be needed" | W |
| E28 | The strongest **near-modality-flexibility in WiFi (Locaris)** provides whichever modality is available without placeholders — but fuses two WiFi modalities (FTM+RSSI), no IMU, no continuous time. | `bhatia2025locaris` · c4f1526c | "provide whichever modality is available (FTM-only or RSSI-only) without requiring placeholders" | W |

---

## Closest competitors — what each LACKS (the novelty pivot)

| ID | Competitor | Has | Lacks (vs our conjunction) | bibkey · source_id |
|----|-----------|-----|----------------------------|--------------------|
| E29 | **iMoT** (closest overall) | transformer + cross-modal attn between accel/gyro; cross-subject eval | **inertial-only** (no WiFi); fixed-rate windows (no Δt); enc-self/dec-cross **branches**; no modality dropout | `nguyen2025imot` · 7b6e4a06 |
| E30 | **AFT-VO** (closest async+attn) | single transformer fusing async sources; no resampling | **cameras-only** (IMU = future work); time **binned** not real-valued; no modality dropout; outdoor VO | `kaygusuz2022aftvo` · 8b22a8f8 |
| E31 | **SeFT / STraTS** (closest architecture, (i)+(ii)) | single perm-invariant set/triplet transformer + continuous time, no ODE | **clinical only**; no missing-modality test; no cross-session; not localization | `horn2020seft` · 2c3d6b02; `tipirneni2022strats` · a7665dca |
| E32 | **Raindrop** (closest robustness, (i)+(iii)) | leave-sensors-out + cross-group generalization + continuous time | **multi-stage graph** (not one block); clinical/HAR; not localization/(x,y) | `zhang2022raindrop` · a068da78 |
| E33 | **A-KIT** (set-transformer + 2 nav sensors) | set-transformer; IMU+DVL | transformer only tunes **EKF noise** (EKF fuses); fixed windows; underwater; no dropout | `cohen2024akit` · fb531c11 |
| E34 | **WIO-EKF** (closest cross-session WiFi+IMU) | WiFi+IMU; cross-day (10 d); DAE robustness | **EKF + branches**; fixed 1 s windows; no transformer; AP-mask ≠ modality dropout | `zhou2024wioekf` · de3074ce |
| E35 | **Wei 2021 / MM-Loc** (closest WiFi+IMU robustness) | WiFi+IMU; missing-WiFi null vector; imbalanced rates | **branches + interpolation**; no attention; random split (no cross-session) | `wei2021sensorfusion` · 75d15e66 |

---

## Novelty conjunction — adversarial verdict

| ID | Finding | Evidence |
|----|---------|----------|
| E36 | **All three adversarial refuters independently concluded the conjunction holds** — no single paper has (i)+(ii)+(iii) for WiFi+IMU localization. | `refute-fusion-domain.md`, `refute-transformer-domain.md`, `refute-continuoustime-applied.md` (all `conjunction_holds = TRUE`) |
| E37 | In the **fusion notebook, ZERO papers use any transformer/set-transformer and ZERO encode real-valued Δt** — (i) and (ii) are uncontested there. | `refute-fusion-domain.md` |
| E38 | A **cross-notebook sweep over all 5 corpora** surfaced only single- or two-pillar partial matches; none is a full counterexample. | `refute-continuoustime-applied.md`, `refute-transformer-domain.md` |
| E39 | **Residual reviewer risk:** each pillar exists individually (ModDrop/Perceiver = dropout; SeFT/STraTS = set+continuous-time; Raindrop = leave-sensor-out; AFT-VO = async transformer). Defense = the **conjunction + the specific mechanism + WiFi+IMU + cross-session**, and honest citation of each precedent. | all 3 refute files |

---

## Integrity flags (carry into writing + refs.bib)

- **F1 — Cite, don't claim:** modality-dropout origin = `neverova2014moddrop` (E19) and `jaegle2021perceiver` "video dropout" (E20); continuous-time-set precedent = `horn2020seft`/`tipirneni2022strats` (E18/E31); learned-time-embedding ancestor = `kazemi2019time2vec`/`shukla2021mtan` (E6). Our novelty is the **conjunction + per-instant (token) dropout + WiFi+IMU cross-session**, not the primitives.
- **F2 — Don't mislabel modalities:** `hua2023smartfps` = **Bluetooth**+IMU; `wang2024damloc` = **Magnetic+BLE**+context; `yu2022multimodal` adds **CSI+UWB**. Do not present these as plain "WiFi+IMU" baselines.
- **F3 — Relative-only numbers:** `cohen2024akit` (">49.5% over EKF"), `rao2022ctin`, `tiku2022anvil` ("up to 35%"), `zhang2023aarescnn` ("8%–48%") report **relative** gains with no quotable absolute MAE — never present as absolute error.
- **F4 — Metadata to fix at bib stage:** `nguyen2025imot` = AAAI **2025** (copyright 2025; arXiv 2024); `diazguerra2023pirnn` = Forum Acusticum **2023** (not 2024); `abdalla2025dataset` = Data in Brief **2025** (not 2026); `yan2019ronin` = arXiv 2019 / ICRA **2020**; `aristorenas2025set` = Stanford **preprint**; `kazemi2019time2vec`, `zhang2022raindrop` venue not grounded in notebook (use known canonical at bib stage).
- **F5 — Numbers we CAN cite (grounded):** UJIIndoorLoc 1NN = 7.9 m (E26/`torressospedra2014ujiindoorloc`); RoNIN-ResNet unseen ATE 5.14 m (`yan2019ronin`); iMoT RoNIN unseen ATE 5.31 m (`nguyen2025imot`); eAaT+ UJI MAE 8.16 m (`nguyen2024aat`); CNNLoc UJI 11.78 m (`song2019cnnloc`). Use only as positioning context, not as direct head-to-head with our numbers (different datasets/metrics).
