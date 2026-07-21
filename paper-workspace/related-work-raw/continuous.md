# Related Work — Group: Continuous-time / async / irregularly-sampled

NotebookLM notebook_id: `49ffba8a-0198-4c43-99a9-34c487d983ae`
Group focus: pillar group for our contribution (i) CONTINUOUS-TIME ENCODING (learned sinusoidal Delta-t added to each token). For each paper: HOW it handles irregular time (ODE solver vs learned time embedding vs interpolation/attention) vs OUR learned sinusoidal Delta-t, and whether it was ever applied to localization / sensor fusion.

All quotes below are verbatim from NotebookLM-cited source text. Source_id is given next to every quote. Nothing is invented; ungroundable facts are marked NOT GROUNDED.

---

## KEY UPFRONT FINDINGS (the gap this group leaves)

1. **Every method in this group is single-stream time-series modeling, almost all on CLINICAL or generic benchmarks (MIMIC-III, PhysioNet, UEA). NONE is applied to indoor localization or WiFi+IMU sensor fusion.** Confirmed per-paper below.
2. **Two mechanistic families for irregular time:** (a) ODE-solver based (Neural ODE, Latent ODE, Neural CDE, GRU-ODE-Bayes, ContiFormer, Shou DGODE) and (b) learned/parametric time-encoding (mTAN, Time2Vec, SeFT, STraTS) or interpolation (IP-Nets) or decay (GRU-D) or graph messaging (Raindrop).
3. **Our Delta-t encoding has direct ancestors here:** mTAN (learned sin+linear time embedding as attention keys/queries), Time2Vec (learned sin+linear scalar-time vector), SeFT (trigonometric time encoding on a SET of (time,value,modality) triplets), STraTS (Continuous Value Embedding of time in a triplet transformer). NONE of these adds a learned sinusoidal Delta-t (elapsed time) directly to each token for WiFi+IMU localization, and none uses our modality-dropout + instant-dropout robustness with cross-session real-world eval.
4. **ContiFormer is the closest "continuous-time transformer" but pays ODE-solver cost** ("substantial time and GPU memory overhead", ~4x slower at length 1000) — this grounds our claim that a learned sinusoidal Delta-t is a lightweight alternative to putting an ODE solver inside attention.
5. **SeFT is the closest architectural ancestor to our unified set-transformer** (permutation-invariant set function over (time, value, modality) triplets) but uses sum-decomposition/DeepSets + a fixed trigonometric time encoding, is single-stream clinical classification, and has no missing-modality robustness eval or localization.

---

## CAPSULES (PILLARS / RELATED METHODS)

### mTAN — Shukla & Marlin 2020/2021 (PILLAR; our Delta-t ancestor) — source_id `cfe566d2`
- **Citation:** Shukla & Marlin (2021). Multi-Time Attention Networks for Irregularly Sampled Time Series. ICLR 2021. (preprint 2020)
- **Method:** Learns an embedding of continuous time and uses attention to produce a fixed-length representation; re-represents the series at a fixed set of reference points, with reference times as queries and observed times as keys.
- **Time handling:** Learned time embedding (sin + linear) used inside an attention/interpolation kernel. The time embedding `phi_h(t)[i]` is linear for i=0 and `sin(omega_ih*t + alpha_ih)` for 0<i<dr, with learnable omega/alpha. KEY: this is exactly the sin+linear primitive we add to each token — but mTAN uses it for kernel-smoothed interpolation to fixed reference points, not as an added per-token Delta-t.
- **Modalities/datasets:** Clinical (PhysioNet 2012, MIMIC-III) + a Human Activity dataset (3D body-tag positions for activity classification). NOT localization, NOT WiFi/IMU fusion.
- **Headline result:** "performs as well or better than a range of baseline ... models while offering significantly faster training times" (no single number grounded for us to cite).
- **Limitation vs ours:** interpolation to reference grid (resampling-like); clinical/activity only; no missing-modality robustness mechanism; no cross-session localization.
- **key_quote:** "Multi-Time Attention Networks learn an embedding of continuous time values and use an attention mechanism to produce a fixed-length representation of a time series containing a variable number of observations." [`cfe566d2`]
- **time-embedding quote:** "phi_h(t)[i] = { omega0h . t + alpha0h, if i = 0 ; sin(omegaih . t + alphaih), if 0 < i < dr ... The periodic terms can capture periodicity ... The linear term ... can capture non-periodic patterns ..." [`cfe566d2`]
- **datasets quote:** "we present interpolation and classification experiments using a range of models and three real-world data sets (Physionet Challenge 2012, MIMIC-III, and a Human Activity dataset)." [`cfe566d2`]
- **diff_vs_ours:** mTAN adds learned sin+linear time embedding to interpolate onto a fixed reference grid (query=reference time); we ADD a learned sinusoidal Delta-t directly to each observation token and never resample — and we apply it to WiFi+IMU localization with modality/instant dropout.

### Time2Vec — Kazemi et al. 2019 (PILLAR; learned vector representation of time) — source_id `6155e781`
- **Citation:** Kazemi et al. (2019). Time2Vec: Learning a Vector Representation of Time. arXiv:1907.05321 (preprint, 2019). (venue grounded only as ArXiv via citation in `cfe566d2`)
- **Method:** Model-agnostic learnable vector embedding of a scalar time, with a linear term + periodic (sine) terms, droppable into any architecture.
- **Time handling:** Learned time embedding. `t2v(tau)[i] = omega_i*tau + phi_i if i=0; F(omega_i*tau + phi_i) if 1<=i<=k`, with F=sine and learnable omega/phi. This is the literal sinusoidal-of-time primitive we use.
- **Modalities/datasets:** Synthetic, Event-MNIST, N_TIDIGITS18 (audio digits), Stack Overflow, Last.FM, CiteULike (classification + recommendation). NOT localization, NOT sensor fusion.
- **Headline result:** "using Time2Vec instead of the time itself offers a boost in performance" (no single number grounded to cite).
- **Limitation vs ours:** it is only a representation of absolute time tested on event/recsys/audio tasks; no fusion architecture, no robustness mechanism, no localization.
- **key_quote:** "we develop a learnable vector representation (or embedding) for time as a vector representation can be easily combined with many models or architectures. We call this vector representation Time2Vec." [`6155e781`]
- **formula quote:** "t2v(tau)[i] = { omega_i*tau + phi_i, if i = 0. F(omega_i*tau + phi_i), if 1 <= i <= k. ... We chose F to be the sine function" [`6155e781`]
- **diff_vs_ours:** Time2Vec is a generic time-embedding component (absolute scalar time) tested on recsys/audio classification; we apply a learned sinusoidal encoding of REAL-VALUED ELAPSED Delta-t per token inside a permutation-invariant set-transformer for WiFi+IMU localization with async robustness.

### Neural ODE — Chen et al. 2018 (PILLAR; ODE solver) — source_id `107ff554`
- **Citation:** Chen, Rubanova, Bettencourt, Duvenaud (2018). Neural Ordinary Differential Equations. NeurIPS 2018.
- **Method:** Parameterize the derivative of the hidden state by a neural network; output computed by a black-box ODE solver (continuous-depth).
- **Time handling:** ODE SOLVER (continuous dynamics). Irregular times handled by integrating over arbitrary intervals.
- **Modalities/datasets:** MNIST, continuous normalizing flows (toy densities), bi-directional spiral time-series. NOT localization, NOT WiFi/IMU.
- **Headline result:** ODE-Net MNIST test error 0.42% with O(1) memory (quoted; not relevant to localization).
- **Limitation vs ours:** generic continuous-depth model; needs an ODE solver; no fusion / multimodal / robustness / localization.
- **key_quote:** "Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. The output of the network is computed using a black-box differential equation solver." [`107ff554`]
- **mechanism quote:** "we parameterize the continuous dynamics of hidden units using an ordinary differential equation (ODE) ... This value can be computed by a black-box differential equation solver" [`107ff554`]
- **venue quote:** "32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montreal, Canada." [`107ff554`]
- **diff_vs_ours:** Neural ODE solves an ODE to model continuous dynamics; we avoid any ODE solver entirely — a learned sinusoidal Delta-t added to each token is our lightweight continuous-time mechanism.

### Latent ODE / ODE-RNN — Rubanova et al. 2019 (PILLAR; ODE solver) — source_id `cbcd353a`
- **Citation:** Rubanova, Chen, Duvenaud (2019). Latent ODEs for Irregularly-Sampled Time Series. NeurIPS 2019.
- **Method:** Generalize RNNs to continuous-time hidden dynamics defined by ODEs (ODE-RNN); evolve hidden state between observations via ODESolve, update discretely with an RNN cell at each observation.
- **Time handling:** ODE SOLVER between observations + RNN update. Handles arbitrary time gaps.
- **Modalities/datasets:** Toy periodic trajectories, MuJoCo (Hopper physics), PhysioNet 2012, Human Activity (3D belt/chest/ankle tags for activity classification). NOT localization, NOT WiFi/IMU fusion (the "activity" data are body-tag positions, not indoor positioning).
- **Headline result:** Latent ODE test predictive RMSE 0.1346 (100/100 obs) on spiral vs RNN 0.1813 (quoted; spiral toy task, not localization).
- **Limitation vs ours:** sequential ODE+RNN; clinical/physics/activity; no missing-modality robustness; no cross-session localization.
- **key_quote:** "We generalize RNNs to have continuous-time hidden dynamics defined by ordinary differential equations (ODEs), a model we call ODE-RNNs." [`cbcd353a`]
- **mechanism quote:** "We define the state between observations to be the solution to an ODE: h'_i = ODESolve(f_theta, h_{i-1}, (t_{i-1}, t_i)) and then at each observation, update the hidden state using a standard RNN update h_i = RNNCell(h'_i, x_i)." [`cbcd353a`]
- **diff_vs_ours:** sequential ODE solver + RNN; we use one parallel permutation-invariant set-transformer with a learned Delta-t token encoding, no ODE, no recurrence.

### Neural CDE — Kidger et al. 2020 (PILLAR; controlled diff. eq. solver) — source_id `5e708532`
- **Citation:** Kidger, Morrill, Foster, Lyons (2020). Neural Controlled Differential Equations for Irregular Time Series. NeurIPS 2020 (text marked "Preprint. Under review."; venue grounded via `d4fc0e36` citation: NeurIPS 33:6696-6707, 2020).
- **Method:** Hidden state driven by the data process X (a controlled differential equation), so the trajectory adapts to subsequent observations; memory-efficient adjoint backprop even across observations.
- **Time handling:** CDE / ODE SOLVER driven by data; handles partially-observed irregularly-sampled multivariate series.
- **Modalities/datasets:** CharacterTrajectories, PhysioNet sepsis, Speech Commands. NOT localization, NOT WiFi/IMU.
- **Headline result:** "state-of-the-art performance against similar (ODE or RNN based) models" (no single number grounded for us).
- **Limitation vs ours:** CDE solver; generic/clinical/speech; no multimodal fusion / robustness / localization.
- **key_quote:** "We demonstrate how controlled differential equations provide a natural extension to the Neural ODE model, which we refer to as the neural controlled differential equation (Neural CDE) model." [`5e708532`]
- **mechanism quote:** "The resulting neural controlled differential equation model is directly applicable to the general setting of partially-observed irregularly-sampled multivariate time series, and (unlike previous work on this problem) it may utilise memory-efficient adjoint-based backpropagation even across observations." [`5e708532`]
- **diff_vs_ours:** Neural CDE drives a differential equation by the data stream; we replace the differential-equation machinery with a learned sinusoidal Delta-t token feature inside attention.

### GRU-ODE-Bayes — De Brouwer et al. 2019 (PILLAR; ODE solver for sporadic series) — source_id `793ea3c3`
- **Citation:** De Brouwer, Simm, Arany, Moreau (2019). GRU-ODE-Bayes. NeurIPS 2019.
- **Method:** Dual-mode filter: GRU-ODE evolves the hidden state in continuous time between observations; GRU-Bayes discretely updates it when an observation arrives (ODE with jumps); like a learnable Kalman filter.
- **Time handling:** ODE SOLVER (continuous-time GRU) between sporadic observations + Bayesian discrete update.
- **Modalities/datasets:** MIMIC-III (EHR), USHCN-DAILY (climate), synthetic 2D Ornstein-Uhlenbeck. NOT localization, NOT WiFi/IMU.
- **Headline result:** not grounded (no single number captured).
- **Limitation vs ours:** sequential ODE+Bayes filter; clinical/climate; no multimodal fusion / robustness / localization.
- **key_quote:** "Instead of the encoder-decoder architecture where the ODE part is decoupled from the input processing, we introduce a tight integration by interleaving the ODE and the input processing steps." [`793ea3c3`]
- **mechanism quote:** "The GRU-ODE is used to evolve the hidden state h(t) in continuous time between the observations and GRU-Bayes transforms the hidden state, based on the observation y, from h(t-) to h(t+)." [`793ea3c3`]
- **kalman-contrast quote:** "Like the celebrated Kalman filter, it alternates between a prediction (GRU-ODE) and a filtering (GRU-Bayes) phase. ... unlike the Kalman filter, our approach is able to learn complex dynamics" [`793ea3c3`]
- **diff_vs_ours:** GRU-ODE-Bayes is a learnable continuous-time filter (ODE + update); we do not filter or solve ODEs — temporal smoothing emerges from set-attention over Delta-t-encoded tokens.

### ContiFormer — Chen et al. 2024 (continuous-time transformer; ODE inside attention) — source_id `d4fc0e36`
- **Citation:** Chen, Ren, Wang, Fang, Sun, Li (2023/2024). ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling. NeurIPS 2023 (file labeled 2024).
- **Method:** Incorporates Neural-ODE continuous dynamics INSIDE transformer attention (CT-MHA): defines latent ODE trajectories per observation, extends discrete dot-product to a continuous-time inner product over an interval, approximated via ODE solver + Gauss-Legendre quadrature.
- **Time handling:** ODE SOLVER inside attention (continuous-time multi-head attention). This is the most direct "continuous-time transformer" competitor to our idea — but it pays ODE cost.
- **Modalities/datasets:** 2D spirals, pendulum, UEA (UWaveGestureLibrary, RacketSports, BasicMotions), event prediction (Neonate, Traffic, MIMIC-III, BookOrder, StackOverflow), forecasting (ETT, Exchange, Weather, ILI). NOT localization, NOT WiFi/IMU fusion.
- **Headline result:** spiral interpolation RMSE 0.49e-2 vs Transformer 1.37e-2 / Latent ODE 2.09e-2 (quoted); ILI forecasting 10% MSE reduction (2.874 -> 2.632) (quoted). These are time-series tasks, not localization.
- **COST (load-bearing for our lightweight claim):** "Utilizing continuous-time modeling in ContiFormer often results in substantial time and GPU memory overhead." [`d4fc0e36`]; "becomes approximately four times slower as the input length extends to 1000 ... significantly higher memory cost" [`d4fc0e36`].
- **Limitation vs ours:** ODE solver embedded in attention => heavy compute/memory; no modality fusion / missing-modality robustness; no localization / cross-session.
- **key_quote:** "we propose ContiFormer that extends the relation modeling of vanilla Transformer to the continuous-time domain, which explicitly incorporates the modeling abilities of continuous dynamics of Neural ODEs with the attention mechanism of Transformers." [`d4fc0e36`]
- **diff_vs_ours:** ContiFormer puts an ODE solver inside attention to make a transformer continuous-time; we get continuous-time behavior by simply ADDING a learned sinusoidal Delta-t to each token — no ODE solver, no quadrature, far lighter.

### SeFT (Set Functions for Time Series) — Horn et al. 2020 (closest to (i)+(ii)) — source_id `2c3d6b02`
- **Citation:** Horn, Moor, Bock, Rieck, Borgwardt (2020). Set Functions for Time Series. ICML 2020 (PMLR 119).
- **Method:** Treats the whole time series as an UNORDERED SET of (time, value, modality) triplets; learns a permutation-invariant set function (DeepSets sum-decomposition) + attention aggregation; explicitly designed for irregular AND unsynchronized measurements.
- **Time handling:** Fixed/parametric trigonometric time encoding per observation (positional-encoding variant on the time axis); NO resampling, NO ODE.
- **Multimodal:** YES, multivariate — each observation carries a modality indicator m_j (triplet (t_j, z_j, m_j)). This is the same triplet idea our set-transformer uses.
- **Modalities/datasets:** MIMIC-III, PhysioNet 2012, PhysioNet 2019 sepsis (clinical only). NOT localization, NOT WiFi/IMU.
- **Headline result:** not grounded as a single number (claims ~order-of-magnitude runtime improvement; no exact figure captured).
- **Limitation vs ours:** clinical classification only; FIXED (non-learned) trigonometric time encoding; sum-decomposition/DeepSets aggregation rather than full self-attention cross-modal+cross-time fusion; NO missing-modality robustness experiment; NO localization / cross-session.
- **key_quote:** "With SEFT, we propose to rephrase the problem of classifying time series as classifying a set of observations." [`2c3d6b02`]
- **set quote:** "we define f to be a set function, i.e. a function that operates on a set and thus has to be invariant to the ordering of the elements in the set." [`2c3d6b02`]
- **time-encoding quote:** "the time encoding converts the 1-dimensional time axis into a multi-dimensional input by passing the time t of each observation through multiple trigonometric functions of varying frequencies." [`2c3d6b02`]
- **triplet quote:** "each observation s_j is represented as a tuple (t_j, z_j, m_j), consisting of a time value t_j ... an observed value z_j ... and a modality indicator m_j" [`2c3d6b02`]
- **diff_vs_ours:** SeFT is the closest ancestor (permutation-invariant set over (time,value,modality) triplets with trigonometric time encoding) but uses a FIXED time encoding + DeepSets aggregation on clinical classification with no missing-modality robustness; we use a LEARNED sinusoidal Delta-t inside ONE self-attention block doing cross-modal AND cross-time fusion, with modality/instant dropout and cross-session WiFi+IMU localization.

### GRU-D — Che et al. 2018 (RNN with missing values; decay) — source_id `974ad9a2`
- **Citation:** Che, Purushotham, Cho, Sontag, Liu (2018). Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports 2018.
- **Method:** GRU with trainable decay on inputs and hidden state, exploiting "informative missingness" via masking + time-interval representations.
- **Time handling:** Trainable exponential DECAY over time gaps (not ODE, not learned sinusoid, not interpolation-to-grid). Decays last observation toward empirical mean as the gap grows.
- **Modalities/datasets:** MIMIC-III, PhysioNet 2012 (clinical), synthetic Gesture. NOT localization, NOT WiFi/IMU.
- **Headline result:** not grounded as a single number.
- **Limitation vs ours:** recurrent + decay heuristic; clinical; missingness handled but no explicit multimodal-dropout robustness eval; no localization.
- **key_quote:** "we develop a novel deep learning model based on GRU, namely GRU-D, to effectively exploit two representations of informative missingness patterns, i.e., masking and time interval." [`974ad9a2`]
- **mechanism quote:** "we propose a GRU-based model called GRU-D ... in which a decay mechanism is designed for the input variables and the hidden states" [`974ad9a2`]
- **diff_vs_ours:** GRU-D handles gaps with a learned decay inside a recurrent net; we handle gaps with a learned sinusoidal Delta-t token feature inside permutation-invariant attention.

### IP-Nets (Interpolation-Prediction Networks) — Shukla & Marlin 2019 (interpolation) — source_id `586bbf61`
- **Citation:** Shukla & Marlin (2019). Interpolation-Prediction Networks for Irregularly Sampled Time Series. ICLR 2019.
- **Method:** Semi-parametric RBF interpolation layers re-represent inputs at uniform reference points (smooth + transient + intensity), then a standard prediction net (GRU).
- **Time handling:** INTERPOLATION to a regular reference grid (resampling-like), via RBF kernels in continuous time.
- **Modalities/datasets:** MIMIC-III, UWaveGesture. NOT localization, NOT WiFi/IMU.
- **Headline result:** not grounded as a single number.
- **Limitation vs ours:** explicitly interpolates onto a regular grid (the resampling we avoid); clinical/gesture; no robustness/localization.
- **key_quote:** "The interpolation network allows for information to be shared across multiple dimensions of a multivariate time series during the interpolation stage, while any standard deep learning model can be used for the prediction network." [`586bbf61`]
- **mechanism quote:** "The architecture is based on the use of several semi-parametric interpolation layers organized into an interpolation network, followed by the application of a prediction network" [`586bbf61`]
- **diff_vs_ours:** IP-Nets interpolate the async series onto a regular grid before prediction; we never resample — Delta-t is encoded per token and fused directly.

### STraTS — Tipirneni & Reddy 2022 (triplet transformer; Continuous Value Embedding) — source_id `a7665dca`
- **Citation:** Tipirneni & Reddy (2022). Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series (STraTS). ACM TKDD 16(6), 2022.
- **Method:** Treats series as a set of (time, variable, value) triplets; embeds continuous time and value via a Continuous Value Embedding (one-to-many FFN), no discretization; multi-head attention over triplets; self-supervised forecasting pretext.
- **Time handling:** Learned Continuous Value Embedding of time (parametric, no ODE, no resampling). Triplet/set transformer like ours.
- **Modalities/datasets:** MIMIC-III, PhysioNet 2012 (clinical). NOT localization, NOT WiFi/IMU.
- **Headline result:** not grounded as a single number.
- **Robustness:** robust to sparsity by avoiding imputation, but NO explicit leave-modality-out experiment (contrast: Raindrop).
- **Limitation vs ours:** clinical; no modality-dropout/stale-sensor robustness eval; no localization / cross-session.
- **key_quote:** "we propose a Self-supervised Transformer for Time-Series (STraTS) model, which overcomes these pitfalls by treating time-series as a set of observation triplets instead of using the standard dense matrix representation." [`a7665dca`]
- **time-embedding quote:** "It employs a novel Continuous Value Embedding technique to encode continuous time and variable values without the need for discretization." [`a7665dca`]
- **diff_vs_ours:** STraTS is a triplet/set transformer with a learned continuous time embedding — architecturally very close — but clinical, no async-robustness eval, no localization, and it embeds absolute time/value rather than our learned sinusoidal ELAPSED Delta-t for WiFi+IMU.

### Raindrop — Zhang et al. 2022 (graph-guided; leave-sensors-out robustness) — source_id `a068da78`
- **Citation:** Zhang, Zeman, Tsiligkaridis, Zitnik (2022). Graph-Guided Network for Irregularly Sampled Multivariate Time Series (RAINDROP). Venue NOT GROUNDED in notebook (commonly ICLR 2022 — do not cite venue from notebook).
- **Method:** Each sample = a sensor graph; a message-passing operator models time-varying inter-sensor dependencies; estimates embeddings for unobserved sensors from observed ones.
- **Time handling:** Graph message passing between sensors at irregular timestamps (no ODE, no fixed grid).
- **Modalities/datasets:** P19 (PhysioNet sepsis 2019), P12 (PhysioNet 2012), PAM (PAMAP2 wearable activity). NOT spatial localization / WiFi+IMU positioning (PAM is wearable-IMU activity classification, not positioning).
- **Headline result:** "outperforms state-of-the-art methods by up to 11.4% (absolute F1-score points)" [`a068da78`]; leave-sensors-out: "outperforms baselines by up to 24.9% in accuracy ... in F1 score" on PAM [`a068da78`].
- **Robustness:** YES — explicit leave-fixed-sensors-out and leave-random-sensors-out (up to 50% sensors missing). This is the one paper in the group with a real missing-channel robustness protocol — but it is single-task clinical/activity CLASSIFICATION, not localization, and uses a graph, not modality-dropout in a unified set-transformer.
- **Limitation vs ours:** clinical/activity classification; graph-based per-sensor messaging (not unified set-attention); no localization; no cross-session real-world positioning protocol.
- **key_quote:** "we introduce RAINDROP, a graph neural network that embeds irregularly sampled and multivariate time series while also learning the dynamics of sensors purely from observational data." [`a068da78`]
- **robustness quote:** "RAINDROP can compensate for missing sensor observations by exploiting dependencies between sensors. To this end, we test whether RAINDROP can achieve good performance when a subset of sensors are completely missing." [`a068da78`]
- **diff_vs_ours:** Raindrop is the closest on robustness (leave-sensors-out) but solves it with a sensor-dependency GRAPH for clinical/activity classification; we get missing/stale robustness from modality-dropout + instant-dropout in ONE unified set-transformer, evaluated by cross-session WiFi+IMU localization.

### Shou et al. 2024 — DGODE (dynamic graph neural ODE; multimodal emotion) — source_id `92ad96ff`
- **Citation:** Shou, Meng, Ai, Li (2024). Dynamic Graph Neural Ordinary Differential Equation Network for Multi-modal Emotion Recognition in Conversation (DGODE). Venue NOT GROUNDED (preprint format), 2024.
- **Method:** Models the discrete GCN propagation as a continuous Graph-ODE; adaptive mixhop aggregation; captures temporal dependency of speaker emotions over a conversation.
- **Time handling:** Graph NEURAL ODE (continuous propagation via ODE). Uses RoBERTa for text features + Bi-GRU for sequence (transformer only for text feature extraction, not for temporal fusion).
- **Modalities/datasets:** Text + audio + video (MERC). NOT localization, NOT WiFi/IMU.
- **Headline result:** not grounded as a single number.
- **Limitation vs ours:** Graph-ODE on conversational multimodal emotion; no localization; ODE solver; no missing-modality robustness eval.
- **key_quote:** "Our DGODE method introduces an adaptive mixhop mechanism ... and uses ordinary differential equations to model the temporal dependence of emotion changes." [`92ad96ff`]
- **modalities quote:** "Each utterance u_i contains audio data v_a, video data v_f, and text data v_t." [`92ad96ff`]
- **diff_vs_ours:** DGODE is a graph-ODE for multimodal emotion (text/audio/video); we use a non-ODE set-transformer with learned Delta-t for WiFi+IMU localization.

---

## CONTEXT / CONTRAST PAPERS (classical Kalman+NN)

### Feng et al. 2023 — Review of Kalman filter + NN hybrid state estimation (SURVEY/CONTEXT) — source_id `eba958aa`
- **Citation:** Feng, Li, Zhang, Jian, Duan, Wang (2023). A review: state estimation based on hybrid models of Kalman filter and neural network. Systems Science & Control Engineering, 11(1), 2173682.
- **Role:** classical contrast — the pre-deep-learning / hybrid lineage for state estimation we improve upon with attention.
- **key_quote:** "this paper reviews the introduction of Kalman filter family, the structure and conception of neural network and the hybrid models of Kalman filter and neural network for state estimation." [`eba958aa`]
- **diff_vs_ours:** surveys Kalman+NN hybrids (filtering); we replace explicit filtering with self-attention temporal fusion over Delta-t tokens.

### Eang & Lee 2024 — DNN-EKF for UWB indoor localization (CONTEXT; classical+DL localization contrast) — source_id `369ed09c`
- **Citation:** Eang, C.; Lee, S. (2024). An Integration of Deep Neural Network-Based Extended Kalman Filter (DNN-EKF) Method in Ultra-Wideband (UWB) Localization for Distance Loss Optimization. Sensors 2024, 24, 7643.
- **Method:** MLP regressor (3 hidden layers x 50 neurons, ReLU) refining EKF-filtered UWB positions; per-axis nets for x and y.
- **Modalities:** UWB (+ webcam ground truth, LiDAR, odometry on a TurtleBot 3). It IS indoor localization — but UWB, not WiFi RSSI; MLP, not attention; fixed-rate/discrete time steps.
- **Time handling:** Fixed-rate discrete steps ("For each time step t"); NO continuous-time / async mechanism, NO ODE.
- **Headline result (grounded):** "the DNN-EKF method achieved an optimal performance with a learning rate of 0.1, yielding a minimum distance loss of 68.06 mm on average, whereas the NN-EKF model resulted in an average distance loss of 73.35 mm, and LPF-EKF showed an average distance loss of 75.30 mm." [`369ed09c`]
- **Limitation vs ours:** UWB (active beacons), not WiFi RSSI; MLP+EKF, no attention; no continuous-time/async handling; no missing-modality robustness; no cross-session generalization.
- **key_quote:** "this paper presents an innovative approach that integrates a deep neural network with an extended Kalman filter (DNN-EKF) to significantly enhance indoor localization accuracy for mobile robots." [`369ed09c`]
- **diff_vs_ours:** DNN-EKF is classical filtering + MLP on UWB at fixed rate; we use a continuous-time set-transformer on asynchronous WiFi+IMU with no filter and no ODE.

---

## COMPETITOR RUBRIC TABLE

Scope note: per the brief, these are PILLARS/related methods, NOT localization competitors; the rubric is applied identically so columns are comparable and the gap is explicit. MODS = modalities fused; ATT = attention/transformer for fusion; CT = continuous-time/async/irregular real-valued gaps WITHOUT resampling or ODE; ROB = explicit missing/stale-modality robustness; XSESS = cross-session/subject/env real-world generalization; UNIFIED = single unified block over all tokens vs per-modality branches.

| bibkey | MODS | ATT | CT | ROB | XSESS | UNIFIED |
|---|---|---|---|---|---|---|
| shukla2021mtan | clinical/activity multivariate (NOT WiFi/IMU) | yes (time-attention) | partial — learned time embedding BUT interpolates to fixed reference grid | no | no | hybrid (encoder-decoder over reference points) |
| kazemi2019time2vec | none (generic scalar time component) | n/a (drop-in component) | partial — learned sin+linear time embedding, but only absolute time, no fusion | no | no | n/a (component) |
| chen2018neuralode | none (single-stream) | no | no — uses ODE solver | no | no | n/a |
| rubanova2019latentode | clinical/physics/activity (NOT WiFi/IMU) | no (RNN+ODE) | no — ODE solver + RNN | no | no | branches (sequential) |
| kidger2020neuralcde | clinical/speech (single-stream) | no | no — CDE/ODE solver | no | no | n/a |
| debrouwer2019gruodebayes | clinical/climate (single-stream) | no | no — ODE solver + Bayes update | no | no | n/a |
| chen2023contiformer | generic/clinical time series (NOT WiFi/IMU) | yes (continuous-time attention) | partial — continuous-time BUT via ODE solver inside attention (heavy) | no | no | unified (single transformer, single stream) |
| horn2020seft | multivariate w/ modality indicator (clinical) | partial (set + attention aggregation, DeepSets core) | yes — fixed trigonometric time encoding, no resampling/ODE | no | no | unified (set function over triplets) |
| che2018grud | clinical multivariate (NOT WiFi/IMU) | no (GRU) | partial — trainable decay over time gaps (not learned sinusoid/ODE) | partial (handles missingness, no leave-modality eval) | no | branches (recurrent) |
| shukla2019ipnets | clinical/gesture multivariate | no (GRU prediction net) | no — interpolates to a regular reference grid (resampling) | no | no | hybrid (interp net + pred net) |
| tipirneni2022strats | clinical multivariate triplets | yes (transformer over triplets) | yes — learned Continuous Value Embedding of time, no resampling/ODE | no (robust to sparsity, no leave-modality eval) | no | unified (triplet/set transformer) |
| zhang2022raindrop | multivariate sensors (clinical/wearable PAM) | partial (graph message passing) | yes — graph messaging at irregular times, no ODE/resampling | yes (leave-fixed/random-sensors-out, up to 50%) | no (cross-sample, not cross-session positioning) | branches (per-sensor graph nodes) |
| shou2024dgode | text+audio+video (MERC, NOT WiFi/IMU) | partial (RoBERTa text only; Bi-GRU+Graph-ODE temporal) | no — Graph Neural ODE solver | no | no | branches (graph + GRU) |
| eang2024dnnekf | UWB (+LiDAR/odom; NOT WiFi RSSI) | no (MLP+EKF) | no — fixed-rate discrete steps | no | no | branches (per-axis MLP + EKF) |

OURS (for reference, not scored): WiFi RSSI + IMU | ATT yes (one self-attention block) | CT yes (learned sinusoidal real-valued Delta-t added per token, no resample, no ODE) | ROB yes (modality-dropout 0.4 + instant-dropout 0.45) | XSESS yes (real-world cross-session) | UNIFIED unified (single permutation-invariant set-transformer over (modality,time) tokens).

KEY COLUMN TAKEAWAYS:
- CT=yes WITHOUT an ODE: only SeFT, STraTS, Raindrop (and our method). mTAN/Time2Vec=partial (learned embedding but interpolation-grid / absolute-time only). All ODE methods are CT-via-solver (scored "no" on our strict "without ODE solver" criterion).
- ROB=yes: only Raindrop (leave-sensors-out). No other pillar has an explicit missing/stale-modality robustness experiment.
- XSESS=yes: NONE. No paper in this group reports cross-session real-world positioning generalization.
- MODS: NONE fuses WiFi RSSI + IMU. NONE is applied to indoor localization (only Eang 2024 is localization, but UWB + EKF, not in the irregular-time-series family and not async/continuous-time).
- UNIFIED + CT + ROB + XSESS together: NONE. The conjunction is the gap.

---

## ATOMIC CLAIMS (load-bearing, each grounded)

See StructuredOutput `claims` array. Each claim has source_ids + verbatim quote + which contribution/gap it supports.

---

## GROUP GAP SYNTHESIS

This group is the methodological backbone for handling irregular/asynchronous time, and it splits into two camps: ODE-solver models that integrate continuous hidden dynamics (Neural ODE [`107ff554`], Latent ODE [`cbcd353a`], Neural CDE [`5e708532`], GRU-ODE-Bayes [`793ea3c3`], ContiFormer [`d4fc0e36`], Shou DGODE [`92ad96ff`]) and lighter parametric time-representation models that encode each observation's time directly — mTAN's learned sin+linear embedding [`cfe566d2`], Time2Vec [`6155e781`], SeFT's trigonometric encoding over (time,value,modality) set triplets [`2c3d6b02`], and STraTS's Continuous Value Embedding [`a7665dca`] — plus interpolation (IP-Nets [`586bbf61`]), decay (GRU-D [`974ad9a2`]), and graph messaging (Raindrop [`a068da78`]). The clear trend is away from resampling/imputation toward either solving an ODE or learning a time embedding, with the transformer/set-function variants (SeFT, STraTS, ContiFormer) converging on exactly the set-of-tokens-with-time-encoding view our architecture uses. But three gaps remain relative to our contributions: (i) our learned sinusoidal real-valued Delta-t added per token is a lightweight alternative to ODE solvers — ContiFormer itself documents "substantial time and GPU memory overhead" and ~4x slowdown at length 1000 [`d4fc0e36`] for the ODE-in-attention route, while the cheaper time-embedding ancestors (mTAN, Time2Vec, SeFT, STraTS) were never carried into our WiFi+IMU localization setting; (ii) although SeFT [`2c3d6b02`] and STraTS [`a7665dca`] realize a single permutation-invariant set/triplet transformer, every method here is single-task clinical/generic time-series modeling and none performs unified cross-modal AND cross-time fusion for localization; and (iii) only Raindrop [`a068da78`] has an explicit missing-channel robustness protocol (leave-sensors-out), no method has modality-dropout + instant-dropout for stale-sensor graceful degradation, and critically NONE of the fifteen sources reports cross-session real-world generalization or is applied to indoor WiFi/IMU positioning at all. Thus the conjunction we claim — a learned sinusoidal Delta-t (no ODE) inside ONE permutation-invariant set-transformer that fuses WiFi+IMU, hardened by modality/instant dropout and validated cross-session — is exactly what this otherwise-mature literature leaves open.
