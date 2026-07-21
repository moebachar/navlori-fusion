# Verified Related-Work References for an ICINCO 2026 Set-Transformer WiFi/IMU Localization Paper

## TL;DR
- Both "pillar" references are real and fully verified: **Vaswani et al. "Attention Is All You Need"** (NeurIPS/NIPS 2017, arXiv:1706.03762, pp. 5998–6008) and **Lee et al. "Set Transformer"** (ICML 2019, PMLR vol. 97, pp. 3744–3753, arXiv:1810.00825); the Set Transformer carries **no DOI** (confirmed by DBLP).
- I recommend six additional verified set/permutation-invariant-fusion works: **Deep Sets** (Zaheer 2017), **SeFT** (Horn 2020), **mTAN** (Shukla & Marlin 2021), **Perceiver** (Jaegle 2021), a set-based WiFi-RSSI localization preprint (**Aristorenas 2025**), and a Transformer multi-sensor indoor-localization paper with "Sensor Snapshot Tokenization" (**Masrur 2025**).
- The closest analogues to the paper's core idea (one self-attention block doing cross-modal + cross-time fusion, continuous-time Δt encoding, async robustness) are **SeFT** and **mTAN** for continuous-time/irregular sampling, and **Perceiver** for modality-agnostic set fusion; no single existing work combines all three contributions for WiFi+IMU localization, which substantiates the paper's novelty claim.

## Key Findings
- All eight primary references are confirmed real publications with canonical metadata below. Three supplementary domain references (IONet, RoNIN, Diaz-Guerra) are also verified for inertial/IMU and tracking context.
- Two carry preprint-relevant caveats: the Aristorenas set-based RSSI localization work appears to be **preprint-only** (arXiv:2506.00656, June 2025, single author, Stanford), and the Masrur localization work has a **published IEEE ICC Workshops 2025 version** (DOI 10.1109/ICCWorkshops67674.2025.11162366, published 8 June 2025) plus a more extended arXiv preprint (arXiv:2501.07774).
- Set Transformer, Deep Sets, Attention Is All You Need, Perceiver, SeFT, and mTAN have **no DOIs** (NeurIPS/PMLR/OpenReview proceedings); where a DOI exists (IEEE) it is given and labeled.
- Verification note on permutation-invariance axis: SeFT and mTAN are invariant/continuous over **observations/time points**; Deep Sets, Set Transformer and Perceiver over **generic set elements/modalities**; Aristorenas and Masrur over **sensors/access points**. Only SeFT and mTAN explicitly handle asynchronous/irregular time sampling.

## Details

### Foundational Pillars — Category: attention/transformer/set

**[Pillar 1] Vaswani et al. (2017) — Attention Is All You Need**
```bibtex
@inproceedings{vaswani2017attention,
  title         = {Attention Is All You Need},
  author        = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle     = {Advances in Neural Information Processing Systems 30 (NIPS 2017)},
  pages         = {5998--6008},
  year          = {2017},
  publisher     = {Curran Associates, Inc.},
  eprint        = {1706.03762},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
- **Verified:** 8 authors (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin); venue NIPS 2017 (Advances in Neural Information Processing Systems 30), Long Beach, 4–9 December 2017; arXiv 1706.03762. Pages are **5998–6008** in the official NeurIPS proceedings; some secondary sources (SCIRP, ACM/Curran reprint) cite **6000–6010** — same paper. **No DOI** is assigned by NeurIPS.
- **Method summary:** Introduces the Transformer, a sequence-transduction architecture relying entirely on multi-head scaled dot-product self-attention, dispensing with recurrence and convolution. Self-attention is permutation-equivariant unless positional encodings are added, which is the mechanism the described paper repurposes for set fusion.
- **Relation:** Foundational; the paper's single self-attention fusion block is a direct descendant of multi-head attention. Category: **attention/transformer/set**.

**[Pillar 2] Lee et al. (2019) — Set Transformer**
```bibtex
@inproceedings{lee2019set,
  title         = {Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
  author        = {Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam R. and Choi, Seungjin and Teh, Yee Whye},
  booktitle     = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  series        = {Proceedings of Machine Learning Research},
  volume        = {97},
  pages         = {3744--3753},
  year          = {2019},
  publisher     = {PMLR},
  eprint        = {1810.00825},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
- **Verified:** 6 authors (Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh); ICML 2019, PMLR vol. 97, pp. 3744–3753 (Long Beach, 09–15 June 2019); arXiv 1810.00825. DBLP explicitly states this publication has **no DOI**.
- **Method summary:** Defines an attention-based, permutation-invariant set-encoding framework using Set Attention Blocks (SAB), Induced Set Attention Blocks (ISAB) — which reduce self-attention cost from quadratic to linear via inducing points — and Pooling by Multihead Attention (PMA). Models pairwise interactions among set elements rather than simple sum/mean pooling.
- **Relation:** The paper's single self-attention fusion block is essentially a Set Transformer applied to a heterogeneous set of (modality, time) observations; permutation invariance here is over generic set elements. Category: **attention/transformer/set**.

### Permutation-Invariant Set Architectures — Category: attention/transformer/set

**[3] Zaheer et al. (2017) — Deep Sets**
```bibtex
@inproceedings{zaheer2017deepsets,
  title         = {Deep Sets},
  author        = {Zaheer, Manzil and Kottur, Satwik and Ravanbakhsh, Siamak and Poczos, Barnabas and Salakhutdinov, Ruslan and Smola, Alexander J.},
  booktitle     = {Advances in Neural Information Processing Systems 30 (NIPS 2017)},
  pages         = {3391--3401},
  year          = {2017},
  publisher     = {Curran Associates, Inc.},
  eprint        = {1703.06114},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
- **Verified:** 6 authors (Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, Alexander J. Smola); NIPS 2017, pp. 3391–3401; arXiv 1703.06114. No DOI (NeurIPS proceedings; ACM mirror uses DL handle 10.5555/3294996.3295098, not a Crossref DOI).
- **Method summary:** Proves that any permutation-invariant set function can be written as ρ(Σ φ(x)), giving a sum-decomposition blueprint for neural networks on unordered sets of variable size, plus conditions for permutation equivariance. Establishes the theoretical basis later refined by Set Transformer's attention pooling.
- **Relation:** Theoretical justification for treating asynchronous multi-sensor observations as an unordered set; the paper uses attention pooling (Set Transformer) rather than plain sum pooling. Permutation invariance over generic set elements. Category: **attention/transformer/set**.

### Continuous-Time / Irregular Sampling via Sets and Attention — Category: continuous-time/async

**[4] Horn et al. (2020) — Set Functions for Time Series (SeFT)**
```bibtex
@inproceedings{horn2020seft,
  title         = {Set Functions for Time Series},
  author        = {Horn, Max and Moor, Michael and Bock, Christian and Rieck, Bastian and Borgwardt, Karsten},
  booktitle     = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  series        = {Proceedings of Machine Learning Research},
  volume        = {119},
  pages         = {4353--4363},
  year          = {2020},
  publisher     = {PMLR},
  eprint        = {1909.12064},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
- **Verified:** 5 authors (Max Horn, Michael Moor, Christian Bock, Bastian Rieck, Karsten Borgwardt); ICML 2020, PMLR vol. 119, pp. 4353–4363; arXiv 1909.12064. No DOI (confirmed via official PMLR BibTeX pmlr-v119-horn20a and DBLP).
- **Method summary:** Reframes classification of irregularly-sampled, asynchronous time series as learning a set function over a set of (time, value, modality) observation tuples, using a Deep-Sets/attention aggregation that needs no imputation or resampling. Each observation's timestamp is encoded so unequal/unaligned sampling rates are handled natively, with per-observation attention contributions for interpretability.
- **Relation:** Very close analogue — the same "observations as a time-stamped set, no resampling" philosophy as contributions (i) and (iii), but applied to healthcare classification rather than WiFi/IMU regression to (x, y). Permutation invariance is over the set of individual observations across channels/time; **explicitly handles asynchronous/irregular sampling**. Category: **continuous-time/async**.

**[5] Shukla & Marlin (2021) — Multi-Time Attention Networks (mTAN)**
```bibtex
@inproceedings{shukla2021mtan,
  title         = {Multi-Time Attention Networks for Irregularly Sampled Time Series},
  author        = {Shukla, Satya Narayan and Marlin, Benjamin M.},
  booktitle     = {International Conference on Learning Representations (ICLR)},
  year          = {2021},
  eprint        = {2101.10318},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
- **Verified:** 2 authors (Satya Narayan Shukla, Benjamin M. Marlin); ICLR 2021; arXiv 2101.10318. No DOI (OpenReview/ICLR; forum id 4c0J6lwQ4_).
- **Method summary:** Introduces a learned **continuous-time embedding** combined with a time-attention mechanism that re-represents irregularly-sampled series at a fixed set of reference points, using observed time points as keys and reference points as queries. The authors describe mTANs as "fundamentally continuous-time, interpolation-based models."
- **Relation:** Directly relevant to contribution (i) — learned continuous-time encoding of Δt; differs in using fixed reference points and interpolation rather than a single permutation-invariant fusion block over modalities. Continuous handling is over irregular time points (per channel). Category: **continuous-time/async**.

### Modality-Agnostic / Multimodal Set Fusion via Attention — Category: multimodal fusion

**[6] Jaegle et al. (2021) — Perceiver**
```bibtex
@inproceedings{jaegle2021perceiver,
  title         = {Perceiver: General Perception with Iterative Attention},
  author        = {Jaegle, Andrew and Gimeno, Felix and Brock, Andrew and Zisserman, Andrew and Vinyals, Oriol and Carreira, Joao},
  booktitle     = {Proceedings of the 38th International Conference on Machine Learning (ICML)},
  series        = {Proceedings of Machine Learning Research},
  volume        = {139},
  pages         = {4651--4664},
  year          = {2021},
  publisher     = {PMLR},
  eprint        = {2103.03206},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
- **Verified:** 6 authors on the official PMLR/ICML version (Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, João Carreira); ICML 2021, PMLR vol. 139, pp. 4651–4664; arXiv 2103.03206. No DOI. (Note: the later arXiv "Perceiver" lineage lists more authors, but the canonical ICML 2021 paper has these six.)
- **Method summary:** Uses asymmetric cross-attention to map a large, modality-agnostic input "byte array" into a fixed latent bottleneck, then applies a deep stack of latent self-attention; handles images, audio, point clouds, video, and audio+video with no per-modality branches. Inputs are treated as a permutation-invariant set tagged with Fourier positional/modality encodings.
- **Relation:** Strong architectural analogue to contribution (ii) — a single attention stack fusing arbitrary modalities without per-modality branches; differs in domain (perception/classification) and lacks continuous-time Δt encoding and async-dropout robustness. Permutation invariance over input elements across modalities. Category: **multimodal fusion** (also attention/transformer/set).

### Localization-Specific Set / Multi-Sensor Attention Fusion — Categories: WiFi fingerprinting & multimodal fusion

**[7] Aristorenas (2025) — Set-Based Indoor Localization with Learned RSSI Embeddings**
```bibtex
@misc{aristorenas2025permutation,
  title         = {Permutation-Invariant Transformer Neural Architectures for Set-Based Indoor Localization Using Learned RSSI Embeddings},
  author        = {Aristorenas, Aris J.},
  year          = {2025},
  eprint        = {2506.00656},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Preprint; no peer-reviewed version found as of June 2026}
}
```
- **Verified:** **Single author** (Aris J. Aristorenas, Department of Computer Science, Stanford University); arXiv:2506.00656, submitted June 2025; **preprint-only** — no peer-reviewed venue located (flag explicitly).
- **Method summary:** Models each WiFi scan as an unordered set of (BSSID, RSSI) pairs with learned BSSID embeddings concatenated with signal strength, processed by a Set Transformer for variable-length, sparse inputs, evaluated across a six-building campus dataset. Per the abstract, "a simple LSTM consistently outperformed all other models, achieving the lowest mean localization error across three tasks (E1–E3), with average errors as low as 2.23 m. The Set Transformer performed competitively, ranking second in every experiment and outperforming the MLP, RNN, and basic attention models."
- **Relation:** Closest WiFi-domain analogue — set-based permutation invariance over access points for RSSI localization; differs by being WiFi-only (no IMU), no continuous-time fusion, and no async/modality dropout. Permutation invariance is over access points/sensors (the set of RSSI readings). Category: **WiFi fingerprinting** (set-based).

**[8] Masrur et al. (2025) — Transformer Indoor Localization with Sensor Snapshot Tokenization**
```bibtex
@inproceedings{masrur2025transformer,
  title     = {Transformer-Based Resource-Efficient Indoor Localization for 5G/6G NLOS-Rich Environments},
  author    = {Masrur, Saad and Cheng, Jung-Fu and Khamesi, Atieh R. and G{\"u}ven{\c{c}}, Ismail},
  booktitle = {2025 IEEE International Conference on Communications Workshops (ICC Workshops)},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ICCWorkshops67674.2025.11162366}
}
```
- **Verified:** 4 authors (Saad Masrur, Jung-Fu (Thomas) Cheng, Atieh R. Khamesi, İsmail Güvenç); IEEE ICC Workshops 2025, published **8 June 2025** (per ResearchGate record), DOI 10.1109/ICCWorkshops67674.2025.11162366; extended preprint **arXiv:2501.07774** ("Transforming Indoor Localization: Advanced Transformer Architecture for NLOS Dominated Wireless Environments with Distributed Sensors," submitted to *IEEE Transactions on Machine Learning in Communications and Networking*). **Proceedings page numbers were not located** and should be confirmed on IEEE Xplore before camera-ready.
- **Method summary:** Introduces "Sensor Snapshot Tokenization" (SST), creating one token per distributed-sensor measurement (power-delay-profile / RSSI) so multi-head attention fuses over the set of spatially distributed sensors; a lightweight Swish-Gated Linear Unit Transformer (L-SwiGLU-T) reduces computation. Per the arXiv:2501.07774 abstract, the SST + L-SwiGLU-T design achieves "substantial accuracy and efficiency gains, outperforming larger Transformer and CNN baselines by over 40% while using significantly fewer FLOPs and training samples," and supports variable sensor counts without architectural changes.
- **Relation:** Closest multi-sensor-fusion-via-attention localization analogue; differs by being RF-only across spatial sensors (not WiFi+IMU cross-modal/cross-time) and not addressing asynchronous time sampling. Permutation invariance is over distributed sensors. Category: **multimodal/multi-sensor fusion** (also WiFi fingerprinting).

### Supplementary verified domain references (optional — inertial/IMU, WiFi, tracking context)
These are not set-attention fusion works but are standard, verifiable anchors for the inertial/IMU and tracking categories if the Related Work needs domain grounding:
- **IONet** — Changhao Chen, Xiaoxuan Lu, Andrew Markham, Niki Trigoni, "IONet: Learning to Cure the Curse of Drift in Inertial Odometry," AAAI 2018, pp. 6468–6476, arXiv:1802.02209. Category: **inertial/IMU**. (No Crossref DOI; AAAI proceedings article 12102.)
- **RoNIN** — Hang Yan, Sachini Herath, Yasutaka Furukawa, "RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods," IEEE ICRA 2020, pp. 3146–3152, DOI 10.1109/ICRA40945.2020.9196860, arXiv:1905.12853. Category: **inertial/IMU**.
- **PI-RNN** — David Diaz-Guerra, Archontis Politis, Antonio Miguel, Jose R. Beltran, Tuomas Virtanen, "Permutation Invariant Recurrent Neural Networks for Sound Source Tracking Applications," Forum Acusticum 2023 (10th Convention of the European Acoustics Association, EAA), DOI 10.61782/fa.2023.1132, arXiv:2306.08510. Permutation invariance over **sound sources** (state-equivariant over tracked tracks); a tracking/state-estimation analogue, single-input (not multi-sensor fusion). Category: **attention/transformer/set** (tracking).

## Recommendations
1. **Cite Vaswani 2017 and Lee 2019 as the two foundational pillars.** Both BibTeX blocks above are camera-ready; do not invent DOIs for them — neither has one.
2. **Anchor contributions (i) continuous-time encoding and (iii) async robustness on SeFT (Horn 2020) and mTAN (Shukla & Marlin 2021).** These are the strongest "no-resampling, time-as-input" analogues and should be discussed as the most direct prior art; explicitly state that they target classification/healthcare and single-input series, not cross-modal WiFi+IMU regression.
3. **Motivate contribution (ii) (single permutation-invariant fusion block, no per-modality branches) with Perceiver (Jaegle 2021) and Deep Sets (Zaheer 2017).** Perceiver is the cleanest "one attention stack, many modalities, no branches" precedent.
4. **Position the localization-specific gap with Aristorenas 2025 and Masrur 2025.** Use them to argue that set/attention fusion has reached indoor localization but only single-modality (WiFi-only or RF sensors-only), with no continuous-time cross-modal WiFi+IMU fusion — this is the wedge for the paper's novelty.
5. **Optionally add IONet and RoNIN** to populate the inertial/IMU category and PI-RNN for the tracking/state-estimation framing.
6. **Thresholds that change these recommendations:** (a) If a peer-reviewed venue is later found for Aristorenas, upgrade its entry from `@misc` to `@inproceedings`/`@article`. (b) Confirm Masrur's ICC Workshops page numbers on IEEE Xplore before submission. (c) If exhaustive search later surfaces a WiFi+IMU continuous-time set-transformer, reclassify it from "analogue" to "direct competing work" and add an explicit head-to-head comparison.

## Caveats
- **No DOIs** exist for the NeurIPS/ICML/ICLR papers (Vaswani, Lee, Zaheer, Horn, Shukla, Jaegle); cite via arXiv ID + proceedings. Do not fabricate DOIs for them.
- **Vaswani page numbers differ by source** (5998–6008 in NeurIPS proceedings vs 6000–6010 in the ACM/Curran reprint); both refer to the same paper. I have used the NeurIPS proceedings range.
- **Aristorenas 2025 is preprint-only** (single-author, Stanford) with no peer-reviewed version found as of June 2026 — flag this in the manuscript per double-blind norms.
- **Masrur 2025**: the IEEE ICC Workshops DOI and 8-June-2025 date are verified, but **proceedings page/volume numbers were not located**; confirm on IEEE Xplore. The extended arXiv:2501.07774 version was submitted (not confirmed accepted) to *IEEE Transactions on Machine Learning in Communications and Networking*.
- **Diaz-Guerra disambiguation:** the permutation-invariant tracking paper is published at **Forum Acusticum 2023** (DOI 10.61782/fa.2023.1132), **not** IEEE TASLP. Do not conflate it with the same group's icosahedral-CNN DoA-estimation paper (IEEE/ACM TASLP, DOI 10.1109/TASLP.2022.3224282), which is a different work.
- **Author name diacritics:** ensure your BibTeX preserves "Łukasz Kaiser," "João Carreira," and "İsmail Güvenç" (escaped in the blocks above) so they render correctly.
- All metadata was cross-checked against primary sources (arXiv abstract pages, official PMLR/NeurIPS proceedings, DBLP, AAAI/IEEE records); residual uncertainty is limited to the two items explicitly flagged (Aristorenas publication status; Masrur page numbers).