# Verified References for the ICINCO 2026 Related Work — Continuous-Time Set-Transformer for Asynchronous WiFi/IMU Indoor Localization

## TL;DR
- **All four "pillar" references are real and verified against their canonical versions**: mTAN (ICLR 2021, arXiv 2101.10318), Neural ODEs (NeurIPS 2018, arXiv 1806.07366, a 2018 Best Paper), Latent ODEs (NeurIPS 2019, arXiv 1907.03907), and Time2Vec (arXiv 1907.05321) — the last of which **exists only as a preprint with no peer-reviewed publication**, the single most important caveat for your bibliography.
- Beyond the pillars, the most relevant verified additional works are **Neural CDEs, Set Transformer, SeFT, Raindrop, GRU-ODE-Bayes, STraTS, IP-Nets, GRU-D, IONet, MM-Loc, and CNNLoc** — 11 references spanning all five of your groups, each with confirmed metadata and an explicit statement of how it handles irregular/async time.
- **No surveyed model uses your exact mechanism** (a learned sinusoidal encoding of real-valued Δt added to each token inside one permutation-invariant set-transformer block); the field instead relies on ODE/CDE solvers (Neural ODE, Latent ODE, GRU-ODE-Bayes, Neural CDE), kernel interpolation onto reference points (IP-Nets, mTAN), set/triplet embeddings (SeFT, STraTS, Set Transformer), graph message passing (Raindrop), or time-decay gates (GRU-D) — usually with per-modality branches rather than a single fused block.

## Key Findings
The four pillars all check out. The only one without a formal venue is **Time2Vec**, which never moved beyond arXiv and must be cited as a preprint. The continuous-time / irregular-sampling literature cleanly partitions into five mechanistic families that you can use as contrast classes in your Related Work narrative:

| Family | Representative works here | Time mechanism | Contrast with your method |
|---|---|---|---|
| ODE/CDE solvers | Neural ODE, Latent ODE, GRU-ODE-Bayes, Neural CDE | Numerical integration of a learned vector field between observations | You use **no solver**; Δt is encoded additively |
| Kernel interpolation | IP-Nets, mTAN | Re-represent series at fixed reference points via RBF/attention smoothing | You use **no resampling/interpolation** |
| Set/triplet embeddings | SeFT, STraTS, Set Transformer | Treat observations as a set; embed time as a feature of each element | You also tokenize as a set but add a **learned sinusoidal Δt** per token and fuse modalities in one block |
| Graph message passing | Raindrop | Learned inter-sensor graph propagates across misaligned readouts | You use a single self-attention block, not a GNN |
| Time-decay RNN | GRU-D | Exponential decay gates + masking/time-interval inputs | You are **recurrence-free** |

The localization-specific works (IONet, MM-Loc, CNNLoc) confirm that the indoor-positioning field still overwhelmingly relies on **per-modality branches with late fusion** (MM-Loc) or **single-modality fixed-length inputs** (IONet for IMU, CNNLoc for WiFi) — which is precisely the design your single permutation-invariant fused block is positioned against.

## Details

Each entry below gives (1) BibTeX-ready metadata, (2) a method summary, and (3) the relation/group line. Verification notes follow where relevant.

---

### GROUP: continuous-time/async — THE FOUR PILLARS (verified in depth)

#### (a) mTAN — Multi-Time Attention Networks
**Metadata.** Satya Narayan Shukla and Benjamin M. Marlin. *Multi-Time Attention Networks for Irregularly Sampled Time Series.* International Conference on Learning Representations (ICLR), 2021. arXiv:2101.10318. (OpenReview id 4c0J6lwQ4_; no separate publisher DOI — ICLR is open-access via OpenReview. Note a later PMC-indexed reprint exists, *Int Conf Learn Represent.* 2021;2021:14897–14911.)

**Method.** mTAN re-represents a sparse, irregularly sampled multivariate series at a fixed set of reference points using *multiple learned continuous-time embeddings* combined with a time-attention mechanism in which the reference points act as queries and the observed timestamps as keys/values, effectively a learned kernel smoother. It is deployed inside an encoder–decoder VAE for interpolation and classification.

**Relation / group.** This is the closest pillar to your contribution: it also learns a continuous-time embedding, but it interpolates onto fixed reference points and uses attention as a similarity kernel, whereas you add a learned sinusoidal Δt encoding directly to each observation token and let one set-transformer block do cross-modal and cross-time fusion without any interpolation. **Group: continuous-time/async.**

#### (b) Neural Ordinary Differential Equations
**Metadata.** Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. *Neural Ordinary Differential Equations.* Advances in Neural Information Processing Systems 31 (NeurIPS 2018), pp. 6572–6583. arXiv:1806.07366. (NeurIPS proceedings paper 7892. **Verified as one of the four NeurIPS 2018 Best Paper Award winners**, Montréal; DBLP key conf/nips/ChenRBD18. Author names per DBLP: "Tian Qi Chen" is the same person who publishes as "Ricky T. Q. Chen.")

**Method.** Instead of a discrete stack of layers, the model parameterizes the *derivative* of the hidden state with a neural network and computes the output with a black-box ODE solver, yielding continuous-depth models with constant memory cost and adaptive evaluation; it also introduces continuous normalizing flows and continuous-time latent-variable models.

**Relation / group.** Foundational continuous-time model; it absorbs irregular time by numerical integration of a learned vector field, while your model performs no integration at all — elapsed time enters only through an additive learned encoding. **Group: continuous-time/async.**

#### (c) Latent ODEs for Irregularly-Sampled Time Series
**Metadata.** Yulia Rubanova, Ricky T. Q. Chen, and David Duvenaud. *Latent ODEs for Irregularly-Sampled Time Series.* Advances in Neural Information Processing Systems 32 (NeurIPS 2019). arXiv:1907.03907. (ACM DL entry 10.5555/3454287.454765 / proceedings hash 42a6845a…)

**Method.** Generalizes RNN state transitions to continuous-time dynamics defined by a Neural ODE, giving the **ODE-RNN**, which then serves as the recognition network of a **Latent ODE** trained as a VAE; both can handle arbitrary gaps between observations and can optionally model observation *times* with a Poisson process.

**Relation / group.** Handles irregular Δt by integrating an ODE between successive observations within a recurrent encoder; you discard both recurrence and ODE solves, encoding Δt directly per token. **Group: continuous-time/async.**

#### (d) Time2Vec
**Metadata.** Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, and Marcus A. Brubaker. *Time2Vec: Learning a Vector Representation of Time.* arXiv:1907.05321, July 2019. **PREPRINT ONLY — no peer-reviewed conference/journal publication exists.** (arXiv DOI 10.48550/arXiv.1907.05321.)

**Method.** A model-agnostic vector representation of time combining one linear term with *k* sinusoidal terms whose frequencies and phases are learned, designed to be concatenated/added into any architecture (e.g., LSTMs, transformers) to better capture periodic and non-periodic temporal patterns.

**Relation / group.** The most direct intellectual ancestor of your time encoding — you adapt the same learned-sinusoid idea but apply it to *real-valued elapsed Δt* per observation and add it to each token of a permutation-invariant set-transformer. **Group: continuous-time/async.**

---

### GROUP: attention/transformer/set

#### Set Transformer
**Metadata.** Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR vol. 97, pp. 3744–3753. arXiv:1810.00825. **No DOI assigned** (confirmed via DBLP conf/icml/LeeLKKCT19; cite via PMLR pmlr-v97-lee19d or arXiv).

**Method.** An attention-based, permutation-invariant architecture for set inputs: a Set-Attention-Block / Induced-Set-Attention-Block (ISAB) encoder models interactions among set elements, and a Pooling-by-Multihead-Attention (PMA) decoder aggregates them; inducing points reduce self-attention cost from quadratic to linear in set size.

**Relation / group.** The architectural backbone of your model — you inherit its permutation invariance over the set of observations and extend it with per-token Δt encoding so a single block performs cross-modal and cross-time fusion. **Group: attention/transformer/set.**

#### STraTS — Self-Supervised Transformer for Time-Series
**Metadata.** Sindhu Tipirneni and Chandan K. Reddy. *Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series.* ACM Transactions on Knowledge Discovery from Data (TKDD), vol. 16, no. 6, art. 105, July 2022, 17 pp. DOI: 10.1145/3516367. arXiv:2107.14293.

**Method.** Represents a multivariate series as a *set of observation triplets* (time, variable, value), embeds continuous time and value with a Continuous Value Embedding (avoiding discretization), and processes the triplet set with a multi-head-attention transformer; self-supervised forecasting pretraining improves performance under limited labels.

**Relation / group.** Very close in spirit — it also tokenizes observations as a set and embeds continuous time, but it keeps separate time/value/variable embeddings for clinical data; you target asynchronous WiFi/IMU fusion with a single fused self-attention block and a learned sinusoidal Δt encoding. **Group: attention/transformer/set.**

---

### GROUP: continuous-time/async — additional models

#### Neural Controlled Differential Equations (Neural CDE)
**Metadata.** Patrick Kidger, James Morrill, James Foster, and Terry Lyons. *Neural Controlled Differential Equations for Irregular Time Series.* Advances in Neural Information Processing Systems 33 (NeurIPS 2020) — NeurIPS 2020 Spotlight. arXiv:2005.08926. (Mathematical Institute, University of Oxford / Alan Turing Institute.)

**Method.** Interpolates the discrete observations into a continuous *control path* (e.g., natural cubic splines), then drives a controlled differential equation whose learned vector field integrates against that path, so the latent trajectory can be continuously updated by later observations — with memory-efficient adjoint backpropagation even across observations.

**Relation / group.** Resolves Neural ODEs' inability to incorporate later data, but still requires interpolation **and** an ODE/CDE solver; you use neither, encoding Δt additively. **Group: continuous-time/async.**

#### SeFT — Set Functions for Time Series
**Metadata.** Max Horn, Michael Moor, Christian Bock, Bastian Rieck, and Karsten Borgwardt. *Set Functions for Time Series.* Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR vol. 119. arXiv:1909.12064. (ETH Zurich / SIB; ACM DL 10.5555/3524938.3525343.)

**Method.** Treats an irregularly sampled, asynchronous series as a *set* of observation tuples (time tⱼ, value zⱼ, modality mⱼ), summarizes the set with a differentiable, highly parallelizable set function, and applies an attention mechanism to weight individual observations; this directly accounts for irregular sampling and unsynchronized measurements.

**Relation / group.** A strong conceptual sibling — also permutation-invariant over observation tuples and explicitly motivated by *asynchronous* sensors — but it aggregates via a set function with fixed sinusoidal time encodings rather than your learned sinusoidal Δt inside full self-attention. **Group: continuous-time/async.**

#### Raindrop — Graph-Guided Network for Irregularly Sampled Multivariate Time Series
**Metadata.** Xiang Zhang, Marko Zeman, Theodoros Tsiligkaridis, and Marinka Zitnik. *Graph-Guided Network for Irregularly Sampled Multivariate Time Series.* International Conference on Learning Representations (ICLR), 2022. arXiv:2110.05357. (OpenReview id Kwm8I7dU-l5; Harvard / MIT Lincoln Laboratory / University of Ljubljana.)

**Method.** Represents each sample as a sensor graph with learnable adjacency, then uses a message-passing operator over the latent sensor graph to embed irregularly sampled, misaligned observations and a hierarchical attention to produce multi-scale embeddings; it explicitly targets robustness when subsets of sensors malfunction (leave-sensor-out).

**Relation / group.** Tackles the same missing/stale-sensor robustness you demonstrate via modality- and instant-dropout, but through learned graph message passing rather than a single attention block plus Δt encoding. **Group: continuous-time/async.**

#### GRU-ODE-Bayes
**Metadata.** Edward De Brouwer, Jaak Simm, Adam Arany, and Yves Moreau. *GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series.* Advances in Neural Information Processing Systems 32 (NeurIPS 2019). arXiv:1905.12374. (KU Leuven, ESAT-STADIUS.)

**Method.** Couples a continuous-time GRU (GRU-ODE, a differential-equation form of the GRU that propagates the hidden state between observations) with a Bayesian update network (GRU-Bayes) that assimilates sporadic, dimension-wise-irregular observations; it encodes a continuity prior and can represent Fokker–Planck dynamics.

**Relation / group.** Continuous-time recurrence with ODE integration between events; you are recurrence-free and solver-free. **Group: continuous-time/async.**

#### IP-Nets — Interpolation-Prediction Networks
**Metadata.** Satya Narayan Shukla and Benjamin M. Marlin. *Interpolation-Prediction Networks for Irregularly Sampled Time Series.* International Conference on Learning Representations (ICLR), 2019. arXiv:1909.07782. (OpenReview id r1efr3C9Ym.)

**Method.** A semi-parametric interpolation network of several RBF kernel layers re-represents a sparse, irregular multivariate series against a set of reference time points (sharing information across dimensions), after which any standard deep network performs the downstream prediction.

**Relation / group.** Direct predecessor to mTAN; handles irregularity by *fixed-kernel* interpolation onto reference points, which you replace with a learned additive Δt token encoding and no resampling. **Group: continuous-time/async.**

#### GRU-D
**Metadata.** Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu. *Recurrent Neural Networks for Multivariate Time Series with Missing Values.* Scientific Reports, vol. 8, art. 6085, 2018. DOI: 10.1038/s41598-018-24271-9. arXiv:1606.01865. (USC / NYU / MIT.)

**Method.** Augments a GRU with trainable *exponential decay* on hidden states and inputs, plus masking and time-interval inputs, to exploit "informative missingness" in multivariate clinical time series.

**Relation / group.** Handles irregular Δt through time-decay gates inside an RNN; you use a feed-forward set-transformer with explicit Δt encoding and modality/instant dropout for graceful degradation. **Group: continuous-time/async.**

---

### GROUP: inertial/IMU

#### IONet
**Metadata.** Changhao Chen, Chris Xiaoxuan Lu, Andrew Markham, and Niki Trigoni. *IONet: Learning to Cure the Curse of Drift in Inertial Odometry.* Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), 2018, pp. 6468–6476. ISSN 2374-3468. arXiv:1802.02209. (All authors University of Oxford; DBLP conf/aaai/ChenLMT18. AAAI papers do not carry DOIs — cite the AAAI proceedings / OJS article 12102.)

**Method.** Breaks the unbounded drift of double-integrated inertial dead reckoning by segmenting IMU data into independent windows and using a deep recurrent network to regress per-window latent polar displacement (and orientation), generalizing even to non-periodic motion (e.g., a trolley).

**Relation / group.** A pure-IMU deep-odometry baseline operating on fixed-rate windows; your model fuses IMU with WiFi at their *native asynchronous rates* inside one architecture. **Group: inertial/IMU.**

---

### GROUP: multimodal fusion

#### MM-Loc — Sensor-Fusion for Smartphone Location Tracking
**Metadata.** Xijia Wei, Zhiqiang Wei, and Valentin Radu. *Sensor-Fusion for Smartphone Location Tracking Using Hybrid Multimodal Deep Neural Networks.* Sensors (MDPI), vol. 21, no. 22, art. 7488, 2021. DOI: 10.3390/s21227488. ISSN 1424-8220. (Published 11 November 2021; Special Issue "Multisensors Indoor Localization." Authors: Xijia Wei, University College London; Zhiqiang Wei, University of Edinburgh; Valentin Radu, University of Sheffield. **Author list corrected from the publisher's citation metadata** — earlier extracts showed only Radu.)

**Method.** An end-to-end hybrid multimodal network with modality-specific sub-networks — an LSTM branch for time-sequential inertial data and a DNN branch for WiFi RSS — each reducing its input to a 128-d embedding; the two embeddings are concatenated into a 256-d fused vector and regressed through fully-connected layers (128, 64) to 2-D coordinates. The paper explicitly states it is "tested on cross-modality samples characterised by *different sampling rate* and data representation," absorbing rate differences inside each per-modality encoder rather than by resampling.

**Relation / group.** The most directly comparable application (WiFi+IMU fusion at different sampling rates), but it uses *separate per-modality branches with late concatenation* — exactly the multi-branch design that your single permutation-invariant fused block is built to avoid. **Group: multimodal fusion.**

---

### GROUP: WiFi fingerprinting

#### CNNLoc
**Metadata.** Xudong Song, Xiaochen Fan, Xiangjian He, Chaocan Xiang, Qianwen Ye, Xiang Huang, Gengfa Fang, Liming Luke Chen, Jing Qin, and Zumin Wang. *CNNLoc: Deep-Learning Based Indoor Localization with WiFi Fingerprinting.* 2019 IEEE SmartWorld/UIC/ATC/SCALCOM/IOP/SCI, pp. 589–595. (IEEE Xplore document 9060340.) An extended journal version appeared in **IEEE Access, DOI: 10.1109/ACCESS.2019.2933921.**

**Method.** Combines a Stacked Auto-Encoder for compact feature extraction from sparse Received Signal Strength vectors with a one-dimensional CNN classifier for multi-building / multi-floor localization, evaluated on UJIIndoorLoc and the Tampere dataset, where it reports 100% building-level and ~95% floor-level success rates.

**Relation / group.** A WiFi-only fingerprinting baseline operating on fixed-length 520-dim RSS vectors; your model adds IMU and treats scans as time-stamped tokens fused continuously in time. **Group: WiFi fingerprinting.**

---

## Recommendations

**Stage 1 — Cite and correctly classify the pillars.** Cite all four; in the bibliography render Time2Vec as `@misc`/`@article{...journal={arXiv preprint arXiv:1907.05321}}` and add a half-sentence in the text noting it is a preprint, since reviewers of a double-blind venue may flag an unverified "venue." Use the NeurIPS proceedings entries (not arXiv) for Neural ODE and Latent ODE, and OpenReview/ICLR for mTAN.

**Stage 2 — Map each reference to a contribution.** Structure Related Work around your three contributions: (i) for *continuous-time Δt encoding*, contrast against Time2Vec (ancestor), mTAN / IP-Nets (interpolation), and the ODE/CDE family (Neural ODE, Latent ODE, GRU-ODE-Bayes, Neural CDE) — emphasizing you need neither resampling nor a solver; (ii) for *single permutation-invariant fused block*, anchor on Set Transformer and contrast SeFT and STraTS (set/triplet methods that still separate time/value/modality embeddings); (iii) for *async-robustness via dropout*, contrast Raindrop's leave-sensor-out and GRU-D's informative-missingness. Frame the application gap with MM-Loc (two-branch late fusion), IONet (IMU-only), and CNNLoc (WiFi-only).

**Stage 3 — Decide on borderline items.** Include MetaGraphLoc (arXiv 2411.17781) **only if** preprints are acceptable in your reference list, and describe it accurately as a GNN/meta-learning fusion method (it is *not* attention/transformer-based). If you need a peer-reviewed *transformer-specific* WiFi+IMU fusion citation, run one more targeted search before submission — none was confirmable here.

**Benchmarks that would change these recommendations.** If you can locate a peer-reviewed (IEEE/ACM/Springer, with DOI) transformer-based WiFi+IMU fusion paper, promote it above MM-Loc as your closest application baseline and demote MetaGraphLoc. If a journal version of any preprint appears (e.g., Time2Vec or MetaGraphLoc getting accepted), switch the citation to the published version and remove the preprint caveat.

## Caveats
- **Time2Vec has no peer-reviewed publication** — it must be cited as an arXiv preprint (1907.05321). This is the most likely metadata point a reviewer will scrutinize.
- **Set Transformer, IP-Nets, mTAN, Raindrop, Neural ODE, Latent ODE, Neural CDE, GRU-ODE-Bayes, and IONet have no DOIs** (ICLR/NeurIPS/AAAI/PMLR are open-access or DOI-less); cite via the official proceedings (PMLR volume + pages where available) or arXiv. DOIs are confirmed only for **GRU-D** (10.1038/s41598-018-24271-9), **STraTS** (10.1145/3516367), **MM-Loc** (10.3390/s21227488), and **CNNLoc's** IEEE Access version (10.1109/ACCESS.2019.2933921).
- The "Tian Qi Chen" / "Ricky T. Q. Chen" discrepancy across DBLP and arXiv refers to the **same author**; use "Ricky T. Q. Chen" (the author's preferred form) to match the canonical arXiv listing.
- **MetaGraphLoc** (arXiv 2411.17781; authors Yaya Etiabi, Eslam Eldeeb, Mohammad Shehab, Wafa Njima, Hirley Alves, Mohamed-Slim Alouini, El Mehdi Amhoud) is a **preprint with no confirmed journal DOI** and is a **graph-neural-network / meta-learning** method, not attention/transformer — do not present it as peer-reviewed or as a transformer model, and do not attach a fabricated DOI.
- For MM-Loc, the full author list (Wei, Wei, Radu) was reconstructed from the MDPI citation metadata; confirm once more against the live MDPI/CrossRef record before camera-ready, as early extracts of the page showed only Valentin Radu.