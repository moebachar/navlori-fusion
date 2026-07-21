# Related Work — Gap Matrix & Novelty Positioning (Phase 4)

**Purpose.** The argument spine of §2: one comparable scoring of every near-competitor on our
contribution axes, the ranked closest works, the adversarially-verified conjunction claim, and the
reviewer rebuttals to keep ready. This is what the "gap → motivation → added value" paragraph is
built from.

**Axes (identical rubric applied to every competitor):**
- **MODS** — modalities actually fused.
- **ATT** — attention/transformer used *as the fusion*.
- **CT** — continuous-time: real-valued Δt / async, **no resampling and no ODE solver**.
- **ROB** — explicit **missing/stale-modality** robustness (e.g. modality dropout).
- **XSESS** — real-world **cross-session** (cross-day / cross-env / cross-subject) generalization.
- **UNIFIED** — a **single** fusion block over all tokens vs per-modality branches.

Legend: ✓ yes · ~ partial · ✗ no. `bibkey` → capsule in `related-work-dossiers.md`; quotes in
`related-work-raw/`. Relative-only numbers flagged ⚑ (never present as absolute — see evidence F3).

---

## Master competitor matrix

| bibkey | MODS | ATT | CT | ROB | XSESS | UNIFIED |
|--------|------|-----|----|----|-------|---------|
| **OURS** | **WiFi RSSI + IMU** | **✓ one self-attn block** | **✓ learned sinusoidal Δt, no resample/ODE** | **✓ modality + instant dropout** | **✓ real cross-session** | **✓ single perm-invariant set** |
| *— WiFi+IMU (or WiFi+inertial) fusion —* | | | | | | |
| `zhou2024wioekf` | WiFi+IMU | ✗ (EKF) | ✗ fixed 1 s | ~ AP-mask | ✓ cross-day 10 d | ✗ branches+EKF |
| `wei2021sensorfusion` | WiFi+inertial | ✗ | ✗ interp→100 ms | ✓ NULL-vector | ✗ random split | ✗ concat branches |
| `zhang2021lstm` | WiFi+PDR | ✗ (LSTM) | ✗ resample 20 Hz | ~ stale smoothing | ✗ cross-user | ✗ single LSTM |
| `yu2022multimodal` | WiFi+IMU+UWB(+CSI) | ✗ (LSTM) | ✗ discrete T | ✓ importance wts | ✗ random | ✗ per-mod streams |
| `yang2025wimu` | WiFi+IMU | ✗ (PF) | ✗ PDR-fill | ~ filter params | ✗ | ✗ branches+PF |
| `herath2021fusiondhl` | WiFi+IMU+floorplan | ✗ (opt+CNN) | ~ sparse constraint | ~ sparse WiFi | ✗ cross-building | ✗ staged |
| `chen2015kalman` | WiFi+PDR+landmarks | ✗ (KF) | ✗ aligned step | ~ PDR fallback | ✗ | ✗ KF branches |
| `zhang2023aarescnn` | WiFi CSI + IMU | ~ attn-CNN | ✗ discrete | ✗ | ✓ cross-env (track) | ✗ 2 stages |
| `hua2023smartfps` ⚑BT | **BT**+IMU | ~ attn sub-layer | ✗ 1 s windows | ~ attn-select | ✗ cross-device | ✗ hybrid branches |
| `wang2024damloc` ⚑Mag | **Mag+BLE**+ctx | ✓ attn | ✗ interp | ~ ctx-zero | ✗ single env | ✗ multi-branch |
| `lajoie2023peoplex` | IMU+UWB+BLE+WiFi | ✗ (factor graph) | ~ async-as-available | ✓ opportunistic | ✗ | ~ one graph (classical) |
| *— async / attention fusion (other domains) —* | | | | | | |
| `kaygusuz2022aftvo` | multi-cam (vision) | ✓ transformer | ~ **binned** time | ✗ (redundancy) | ✗ weather splits | ~ late-fuse preds |
| `cohen2024akit` | IMU+DVL | ~ set-T tunes EKF | ✗ fixed 1 s | ✗ | ~ same-session segs | ✗ EKF fuses |
| `geneva2018async` | LIDAR+stereo+GPS | ✗ (graph) | ~ interp-to-states | ✗ | ✗ | ~ one graph (classical) |
| `lin2025scmpr` | RGB+LiDAR map | ✓ cross-modal attn | ✗ | ✗ (appearance) | ✓ day/night/season | ✗ branches |
| `xiao2024effloc` | single camera | ✓ ViT | ✗ static | ✗ | ✓ cross-day | n/a single-mod |
| `diazguerra2023pirnn` | acoustic | ~ RNN+attn | ✗ | ✗ | ✗ | ~ over sources |
| *— inertial transformers —* | | | | | | |
| `nguyen2025imot` | inertial only | ✓ enc/dec attn | ✗ rate-in-token | ✗ | ✓ cross-subject | ✗ enc/dec branches |
| `rao2022ctin` | inertial only | ✓ hybrid attn | ✗ window | ✗ | ✓ unseen subj | ✗ ResNet+dec |
| `brotchie2023riot` | inertial 9D | ✓ self-attn | ✗ 100 Hz | ✗ | ✓ unseen | ~ one enc-dec (1 mod) |
| `zheng2024neurit` | inertial 9D | ✓ TF-BRT | ✗ window | ✗ | ✓ cross-building | ✗ RNN+TF+conv |
| `herath2022niloc` | IMU only | ✓ 2-branch T | ✗ dist-resample | ✗ | ✓ per-scene | ✗ branches |
| `chen2021rninvio` | vision+IMU | ~ (EKF fuses) | ✗ interp 100 Hz | ✓ drop-visual | ✓ cross user/bldg | ✗ EKF |
| *— WiFi transformers (single/RF-only) —* | | | | | | |
| `aristorenas2025set` | WiFi RSSI | ✓ Set-T | ✗ single scan | ~ missing-AP | ✗ cross-building | ✓ set (1 modality) |
| `bhatia2025locaris` | WiFi FTM+RSSI | ✓ decoder LLM | ✗ token-per-read | ✓ any-modality | ✓ cross-env few-shot | ✓ token stream |
| `abdullah2025ris` | CSI+RSS+RIS | ✓ transformer | ✗ sync tokens | ✗ ablation | ✗ sim | ✓ [CLS]+8 tokens |
| `tiku2022anvil` | WiFi RSSI | ✓ MHA | ✗ static | ~ AP-dropout | ✓ cross-device | ✓ (1 modality) |
| `ott2024radiofm` | 5G CIR | ✓ transformer | ✗ fixed seq | ~ mask-pretext | ✓ cross-site | ✓ (1 modality) |
| `nguyen2024aat` | WiFi RSS | ✓ transformer | ✗ static | ✗ impute | ~ unseen split | ✓ (1 modality) |
| `nasir2024hytra` | WiFi RSS | ✓ enc-only T | ✗ fixed order | ✗ impute | ✓ 4-mo split | ✓ (1 modality) |
| *— continuous-time set/triplet (clinical) —* | | | | | | |
| `horn2020seft` | clinical multivar | ~ set+attn agg | ✓ trig time, no ODE | ✗ | ✗ | ✓ set over triplets |
| `tipirneni2022strats` | clinical triplets | ✓ transformer | ✓ CVE, no ODE | ✗ | ✗ | ✓ triplet set-T |
| `zhang2022raindrop` | clinical/HAR sensors | ~ graph | ✓ trig time, no ODE | ✓ leave-sensors-out | ~ cross-group | ✗ multi-stage graph |
| `chen2023contiformer` | generic TS | ✓ CT-attn | ~ **ODE-in-attn** (heavy) | ✗ | ✗ | ✓ (single stream) |
| `shukla2021mtan` | clinical/HAR | ✓ time-attn | ~ embed but interp-grid | ✗ | ✗ | ✗ enc-dec ref-points |

**The decisive reading.** No row except OURS is ✓ on the conjunction **CT + UNIFIED-attention +
ROB** *and* applied to **WiFi+IMU** with **XSESS**. The matrix shows every near-work is missing at
least one of these — and the WiFi+IMU rows are missing two or three.

---

## Closest competitors, ranked (the works §2 must explicitly position against)

1. **iMoT** `nguyen2025imot` — *closest in architecture spirit.* Transformer + cross-modal attention,
   cross-subject eval. **Lacks:** WiFi (inertial-only), real-valued Δt (rate baked into token dim),
   a single unified block (enc-self / dec-cross branches), modality dropout. → We add the absolute
   WiFi anchor, learned Δt, one perm-invariant block, and dropout robustness.
2. **AFT-VO** `kaygusuz2022aftvo` — *closest async-attention fusion.* One transformer fusing
   asynchronous sources without resampling. **Lacks:** cross-modality (cameras only; IMU = explicit
   future work), real-valued Δt (bins time), modality dropout; outdoor VO. → We fuse heterogeneous
   WiFi+IMU, encode real-valued Δt (no binning), and train with dropout.
3. **SeFT / STraTS** `horn2020seft` / `tipirneni2022strats` — *closest architecture, satisfy (i)+(ii).*
   Single perm-invariant set/triplet transformer with continuous time, no ODE. **Lacks:** the
   localization domain, missing-modality robustness, cross-session. → We carry this recipe to WiFi+IMU
   with dropout + cross-session validation (the hard part for WiFi fingerprints).
4. **Raindrop** `zhang2022raindrop` — *closest robustness, satisfies (i)+(iii).* Leave-sensors-out +
   cross-group generalization + continuous time. **Lacks:** a single unified block (multi-stage graph),
   localization. → We get robustness from dropout inside one block, for (x,y) localization.
5. **WIO-EKF** `zhou2024wioekf` — *closest cross-session WiFi+IMU.* Genuine 10-day cross-day eval.
   **Lacks:** transformer/Δt (EKF + branches + fixed windows). → We replace the filter with one
   continuous-time set-transformer.
6. **Wei 2021 / MM-Loc** `wei2021sensorfusion` — *closest WiFi+IMU missing-modality handling.*
   NULL-vector for missing WiFi + imbalanced rates. **Lacks:** attention/Δt/cross-session; hand-set
   flag, not learned dropout.
7. **A-KIT** `cohen2024akit` — *a set-transformer with two nav sensors* — but it only regresses EKF
   noise; the EKF fuses. Fixed windows, underwater, single-session.

---

## The (i)–(iv) novelty grid (for the contribution claim / a possible table in §2)

| Criterion | Anyone with it? | But missing | Therefore |
|-----------|-----------------|-------------|-----------|
| (i) learned real-valued Δt, no resample/ODE | mTAN/Time2Vec (embed), SeFT/STraTS (set) | not WiFi+IMU; not with (ii)+(iii) | uncontested for our setting |
| (ii) one perm-invariant set-T = cross-modal+cross-time | SeFT/STraTS (clinical), Aristorenas (1 modality) | not WiFi+IMU multimodal; not with (i)+(iii) | uncontested for our setting |
| (iii) modality+instant dropout, missing/stale | ModDrop, Perceiver, Raindrop (precedents) | gesture/clinical; whole-modality only; not localization | per-instant + WiFi+IMU is ours |
| (iv) real cross-session WiFi localization | WIO-EKF (EKF), UJI 4-mo split | not in a unified continuous-time transformer | ours is the first to pair it with (i)+(ii)+(iii) |

**Contribution statement = the conjunction:** the first to put learned real-valued Δt **and** a single
permutation-invariant set-transformer (cross-modal + cross-time at once) **and** modality+instant
dropout into one model for **asynchronous WiFi+IMU indoor localization**, validated **cross-session**.

---

## Adversarial verification (3 independent refuters)

All three returned **`conjunction_holds = TRUE`**:

- **Fusion-domain refuter:** in the fusion notebook, **zero** papers use any transformer/set-transformer
  and **zero** encode real-valued Δt → (i) and (ii) are uncontested there; conjunction unbreakable
  regardless of (iii). Closest partial = Wei-2021, then WIO-EKF, then Geneva-2018.
- **Transformer-domain refuter:** closest = AFT-VO (single async-fusion transformer) but cameras-only +
  binned time + no dropout; A-KIT only tunes EKF; cross-notebook check confirms iMoT = inertial-only.
- **Continuous-time-applied refuter:** only 2/15 papers are localization-applied (both UWB/EKF, no
  pillars); SeFT/STraTS satisfy (i)+(ii) but clinical; Raindrop satisfies (i)+(iii) but multi-stage +
  clinical. Cross-notebook sweep over all 5 corpora: no full-conjunction paper.

---

## Residual reviewer risks → rebuttals (keep these ready)

| # | Risk a reviewer raises | Rebuttal (grounded) |
|---|------------------------|---------------------|
| R1 | "Each ingredient is known prior art — incremental." | True for each primitive; we **cite every precedent** (ModDrop, Perceiver, SeFT/STraTS, Time2Vec/mTAN). The contribution is the **conjunction + per-instant dropout + the specific async WiFi/IMU profile + cross-session**, none of which co-occur in any prior work (E36–E38). |
| R2 | "AFT-VO already fuses async sources in one transformer." | AFT-VO is **same-modality** (multi-camera), **bins** time (not real-valued Δt), has **no modality dropout**, and is outdoor VO; IMU is explicitly future work (E30). |
| R3 | "SeFT/STraTS already do continuous-time set transformers." | Clinical EHR, **no missing-modality robustness test, no cross-session, not localization** (E31). We port and harden the recipe for WiFi+IMU. |
| R4 | "A-KIT is a set-transformer fusing IMU + another sensor." | The set-transformer only **regresses EKF noise**; the **EKF** performs fusion; fixed windows; underwater; single-session (E33). |
| R5 | "Modality dropout / cross-attention readout aren't new." | Correct — we **cite ModDrop and Perceiver** as the origins; our addition is **per-instant (token) dropout** and the conjunction (E19/E20, F1). |
| R6 | "Your sub-metre WiFi numbers look optimistic." | Disclose the Webots WiFi is GPR-synthesised; the **real result is the cross-session MSILN improvement**, not the sim number (consistent with project honest-findings). |

---

## Must-cite precedents (honesty ledger — frame as "building on", not "novel to us")

- **Modality dropout:** `neverova2014moddrop` (origin) + `jaegle2021perceiver` (video dropout).
- **Continuous-time set/triplet transformer:** `horn2020seft`, `tipirneni2022strats`.
- **Learned time embedding (Δt primitive):** `kazemi2019time2vec`, `shukla2021mtan`.
- **Permutation-invariant attention pillars:** `lee2019settransformer`, `zaheer2017deepsets`,
  `vaswani2017transformer`.
- **Leave-sensors-out robustness:** `zhang2022raindrop`.
- **Async-without-resampling (classical contrast):** `geneva2018async`, `lajoie2023peoplex`.
