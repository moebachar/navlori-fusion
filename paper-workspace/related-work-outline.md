# Related Work — Narrative Blueprint / Draft Scaffold (Phase 4)

**Purpose.** The paragraph-by-paragraph skeleton for §2 "Related Work and motivation", following the
director's structure in `main.tex`: **groups of methods → gaps → motivation → added value**. Each
paragraph lists: topic sentence, the works to cite (`bibkey`), the gap it leaves, the transition, and
the evidence IDs (E#) from `related-work-evidence.md` that back it. This becomes the `draft.tex`
bullets / `sections/02_related_work.tex` scaffold.

**Budget:** ~1 page, ~5 paragraphs (scope.md §5). The 5-group corpus taxonomy is condensed into 4
SOTA paragraphs + 1 gap/motivation paragraph. Cite the load-bearing works only — the full set is in
`related-work-dossiers.md`; do not cite all 65.

**Self-cite / blinding:** single-blind (authors visible) — refer to our own method in the **third
person** ("we propose…" is fine in §2 only if consistent with the rest; otherwise "this work").
**Baselines of record** (scope.md): WiFi = wlan_localization; IMU = RoNIN ResNet1D; headline = MSILN
cross-session.

---

## ¶1 — WiFi fingerprinting (the absolute-anchor modality)

**Topic sentence.** WiFi RSSI fingerprinting is the canonical absolute-position modality indoors,
evolving from classical signal-space matching to deep and, recently, transformer-based encoders.

**Arc to cite (chronological → architectural):**
- Classical baselines: RADAR kNN `bahl2000radar`, Horus `youssef2005horus`; the canonical benchmark
  UJIIndoorLoc `torressospedra2014ujiindoorloc` (1NN ≈ 7.9 m; **4-month cross-session split** — E26).
- Deep: DeepFi `wang2015deepfi`, CNNLoc `song2019cnnloc` (UJI 11.78 m).
- Transformer wave: All-embracing Transformers `nguyen2024aat` (the **Anchor2Vec** tokeniser our
  WiFi-Net descends from; UJI 8.16 m), Locaris `bhatia2025locaris` (flexible-modality; E28),
  ANVIL `tiku2022anvil`, TIPS `zhang2022tips`, the perm-invariant Set-Transformer over RSSI
  `aristorenas2025set` (E17).
- Open frontier (survey): `martinfrechina2025review`, `hechan2016survey` (fingerprints drift → E27).

**Gap left.** All single-modality; treat a scan as a static vector or fixed-step sequence; the one
perm-invariant set-transformer is single-scan and beaten by a plain LSTM; **none encodes real-valued
Δt or fuses inertial data** (E17, E8/E9, E27).

**Transition.** "WiFi alone is sparse (~1 Hz) and drifts across sessions, motivating fusion with a
dense motion modality."

**Backed by:** E26, E27, E17, E28.

---

## ¶2 — Inertial / learned inertial navigation (the dense motion modality)

**Topic sentence.** Learned inertial navigation provides dense, high-rate motion estimates, advancing
from LSTM/CNN regressors to transformer architectures, but is inherently drift-bound without an
absolute reference.

**Arc to cite:**
- Foundations / benchmark: IONet `chen2018ionet`, TLIO `liu2020tlio`, **RoNIN `yan2019ronin`**
  (our IMU baseline-of-record; unseen ResNet ATE 5.14 m).
- Transformer wave: CTIN `rao2022ctin`, RIOT `brotchie2023riot`, NeurIT `zheng2024neurit`, and the
  **closest overall competitor iMoT `nguyen2025imot`** (cross-modal attention between accel/gyro;
  RoNIN unseen ATE 5.31 m — E29).
- Survey: `cohenklein2024survey` (note: set-transformers used for sensor-*outage* recovery in marine
  DVL — outside indoor localization).

**Gap left.** Inertial-only; fixed-rate windows or rate-baked-into-token-dim (E9); fusion, where it
exists (RNIN-VIO), is EKF-based graceful degradation, not learned (E22); generalization is
cross-subject, not cross-session WiFi (E29). **No absolute anchor.**

**Transition.** "Combining WiFi's absolute anchor with IMU's dense motion is the natural step —
multimodal fusion."

**Backed by:** E29, E9, E22.

---

## ¶3 — Multimodal WiFi+IMU fusion (the on-problem cluster)

**Topic sentence.** Fusing WiFi with inertial data is well studied, but predominantly through
per-modality branches merged by classical filters or late concatenation, with time handled by
resampling.

**Arc to cite:**
- Classical / filter fusion: Kalman `chen2015kalman`, particle-filter WiMU `yang2025wimu`,
  EKF WIO-EKF `zhou2024wioekf` (**cross-day 10 d**, APE 2.53 m — E25), factor-graph PEOPLEx
  `lajoie2023peoplex` (async-as-available — closest async-without-resampling, but classical), and the
  named asynchronous-fusion precedent Geneva `geneva2018async` (still interpolates to fixed states — E4).
- Deep fusion: Multi-Modal Recurrent Fusion `yu2022multimodal`, Fusion-DHL `herath2021fusiondhl`,
  LSTM fusion `zhang2021lstm`, and **MM-Loc `wei2021sensorfusion`** (NULL-vector for missing WiFi —
  closest learned missing-modality handling, E21/E35). Attention only as a sub-layer: SmartFPS
  `hua2023smartfps`, DamLoc `wang2024damloc`.
- Modality-dropout precedent (⚑ cite as origin, not ours): ModDrop `neverova2014moddrop` (E19).
- Real multi-rate datasets that motivate async: `silva2023dataset`, `abdalla2025dataset` (E5).
- Surveys: `wangahmad2025survey`, `lukasik2024survey`.

**Gap left.** (i) resampling/interpolation to a fixed grid, even the async precedents align to fixed
states (E3, E4, E5); (ii) per-modality branches + late fusion, attention only a sub-layer (E12, E13);
(iii) robustness is hand-set/filter-based, not learned per-instant dropout, and evaluation is mostly
random/cross-user — only WIO-EKF reports cross-day, via an EKF (E21, E24, E25). **⚑ Honesty:** do not
present SmartFPS (BT+IMU) or DamLoc (Mag+BLE) as plain WiFi+IMU (F2).

**Transition.** "These limits are architectural — they stem from how time and modality structure are
represented; attention over sets offers a different primitive."

**Backed by:** E3, E4, E5, E12, E13, E19, E21, E24, E25.

---

## ¶4 — Attention, permutation-invariant sets, and continuous-time models (the architectural neighbours)

**Topic sentence.** Our architecture draws on attention and permutation-invariant set learning, and on
the continuous-time time-series literature that handles irregular sampling without resampling.

**Arc to cite:**
- Pillars: Transformer `vaswani2017transformer` (positional encoding assumes equidistant positions —
  E1), Set Transformer `lee2019settransformer` (E10), Deep Sets `zaheer2017deepsets` (E11);
  Perceiver `jaegle2021perceiver` (⚑ "video dropout" precedent — E20).
- Attention-for-localization neighbours: **AFT-VO `kaygusuz2022aftvo`** (closest async-attention
  fusion — bins time, cameras-only, IMU future work — E2/E16/E30), A-KIT `cohen2024akit` (set-T tunes
  EKF — E15/E33), EffLoc `xiao2024effloc`, SCM-PR `lin2025scmpr`, PI-RNN `diazguerra2023pirnn`.
- Continuous-time: ODE family Neural ODE `chen2018neuralode` / Latent ODE `rubanova2019latentode`
  (E8), ContiFormer `chen2023contiformer` (ODE-in-attention, heavy — E7); learned-time-embedding
  ancestors Time2Vec `kazemi2019time2vec` / mTAN `shukla2021mtan` (E6); set/triplet transformers
  **SeFT `horn2020seft` / STraTS `tipirneni2022strats`** (closest architecture (i)+(ii), clinical —
  E18/E31); leave-sensors-out Raindrop `zhang2022raindrop` (closest robustness (i)+(iii), clinical —
  E23/E32).

**Gap left.** These primitives exist but are scattered: async-attention bins time (AFT-VO) or delegates
to an EKF (A-KIT); set/continuous-time transformers are clinical (SeFT/STraTS); robustness work is
graph-based clinical (Raindrop); ODE routes are heavy (ContiFormer). **None combines a learned
real-valued Δt + one unified perm-invariant set-transformer + missing-modality dropout for WiFi+IMU
localization** (E36–E38).

**Transition.** "Bringing these threads together for asynchronous WiFi+IMU localization is the gap
this work fills."

**Backed by:** E1, E2, E6, E7, E8, E10, E11, E15, E16, E18, E20, E23.

---

## ¶5 — Gap → Motivation → Added value (the pivot)

**Gap (synthesise the four ¶ above into 2–3 sentences).** Across WiFi, inertial, fusion, and the
architectural literature, no prior work brings together — for asynchronous WiFi+IMU indoor
localization — (i) a learned encoding of each observation's real-valued elapsed Δt without resampling
or an ODE solver, (ii) a single permutation-invariant set-transformer whose one self-attention block
performs cross-modal and cross-time fusion at once, and (iii) train-time modality- and instant-dropout
giving graceful degradation under missing/stale sensors. The nearest works each hold a subset: iMoT
(transformer, inertial-only, fixed-rate), AFT-VO (one async-fusion transformer, single-modality,
binned time), SeFT/STraTS (set + continuous-time, clinical), Raindrop (leave-sensors-out, clinical),
WIO-EKF (cross-session WiFi+IMU, but an EKF over branches). *(Verified: all 3 adversarial refuters
found no full-conjunction counterexample — E36–E38.)*

**Motivation.** Real deployments sample modalities at unequal rates (WiFi ~1 Hz vs IMU ~30 Hz), drop
sensors, and serve stale fingerprints; fingerprints also drift across sessions (E27). A method that
ingests observations as a time-stamped set and stays robust when some are missing or stale is what
these conditions demand.

**Added value (3 bullets = our 3 contributions, framed against the gap).**
1. A learned sinusoidal encoding of real-valued Δt added per token — a lightweight alternative to ODE
   solvers (E7) and to resampling (E3–E5), extending the time-embedding lineage (E6) to async WiFi+IMU.
2. One permutation-invariant set-transformer where a single self-attention block does cross-modal +
   cross-time fusion — unifying the branched fusion topology (E12) on the Set-Transformer pillar (E10).
3. Modality- and **per-instant** dropout for missing/stale robustness — extending ModDrop/Perceiver
   (E19/E20) from whole-modality to the (modality, time) token set — validated by **real cross-session**
   generalization (the MSILN result), which the nearest competitors do not pair with (i)+(ii) (E25, E31).

**Backed by:** E36–E38 (conjunction), E27 (drift), E3–E7/E10/E12/E19/E20/E25/E31 (per added-value bullet).

---

## Suggested citation density (fit the 1-page budget)

- **Must cite (load-bearing):** `torressospedra2014ujiindoorloc`, `song2019cnnloc`, `nguyen2024aat`,
  `bhatia2025locaris`, `aristorenas2025set`, `yan2019ronin`, `nguyen2025imot`, `zhou2024wioekf`,
  `wei2021sensorfusion`, `neverova2014moddrop`, `geneva2018async`, `vaswani2017transformer`,
  `lee2019settransformer`, `zaheer2017deepsets`, `kaygusuz2022aftvo`, `cohen2024akit`,
  `horn2020seft`, `tipirneni2022strats`, `zhang2022raindrop`, `kazemi2019time2vec`/`shukla2021mtan`,
  `chen2023contiformer`, plus the WiFi/IMU baselines-of-record (wlan_localization repo, RoNIN).
- **Cite if space (breadth):** `bahl2000radar`, `tiku2022anvil`, `rao2022ctin`, `brotchie2023riot`,
  `zheng2024neurit`, `yu2022multimodal`, `herath2021fusiondhl`, `lajoie2023peoplex`,
  `jaegle2021perceiver`, `chen2018neuralode`/`rubanova2019latentode`, the surveys.
- **Can drop / group-cite:** Horus, DeepFi, HyTra, Turgut, Ott, TIPS, Abdullah, EffLoc, SCM-PR,
  PI-RNN, IDOL/RIDI/OxIOD (cite only as benchmark provenance), GRU-D, IP-Nets, Neural CDE,
  GRU-ODE-Bayes, DGODE, Eang, Feng-reviews.
