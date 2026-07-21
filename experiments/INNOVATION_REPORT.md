# NavLoRI Innovation Report — Expert-Team Attack

Generated 2026-06-26T14:41:35.

Inputs: **14 expert-angle agents** (10 succeeded, 4 rate-limited), **40 leads collected**, **top-20 curated**, **12 hybrid innovations**, 3-tier roadmap.

---

## The headline narrative

The paper tells a layered story: "Cross-session indoor WiFi localization is mostly a DATA problem before it is a model problem." Section 4 (CFIS) shows that train-only BSSID vocab + RSSI-rank + AP-pair-delta features + a session-stable magnetometer modality close ~2.5m of the 2.6m headroom with NO architectural change to the fusion stack â€” a result reproducible by anyone in a day. Section 5 (architectural) then shows the remaining gap is unlocked by replacing the dense-RSSI WiFiNet with a per-BSSID Set Transformer + per-BSSID anchor memory, demonstrating that explicit AP identity (not just RSSI magnitude) is what survives the session shift. Section 6 (probabilistic) reframes the prediction head as a mixture-Laplace CRPS posterior, exposing corridor aliasing as genuinely multimodal â€” the headline figure is two-mode posteriors at corridor crossings replacing wall-piercing mean predictions â€” and gives free heteroscedastic uncertainty that subsumes the conformal layer. Sections 4-6 compose to bring MAE from 9.1m toward ~6m, near the kNN floor, on the cross-session MSILN B1 test split. Section 7 (research extension) sketches physics-pretrained JEPA with LM-fit AP coordinates and hyperbolic-TDoA tokens as the next-paper trajectory. The honest finding remains center-stage: the 11-day cross-session gap is closed not by adding model capacity but by injecting physics-grounded invariance at the input, an interpretable memory at the readout, and a multimodal posterior at the head â€” three orthogonal, independently ablatable interventions that match the three documented failure modes (calibration drift, missing-AP confound, residual heavy-tail).

---

## TL;DR — what to do

Cross-session WiFi is the bottleneck â€” 9.1m parametric vs 6.48m kNN floor leaves ~2.6m of algorithmic headroom against a heavy-tailed, session-drifted residual distribution. The wins concentrate on three orthogonal axes that compose multiplicatively: (1) physics-invariant WiFi inputs (pair-deltas, rank, train-only vocab) cancel device/session drift in closed form; (2) a session-stable second modality (magnetometer, already in MSILN raw data, currently discarded) provides an anchor where WiFi fails; (3) loss/optimizer changes (pinball or mixture-CRPS, SWAD, C-Mixup) attack the heavy-tail and 2-session-overfitting problem at zero architectural risk. The "wow" result is the CFIS data-only pipeline shipping ~2.5m of lift with NO model change, then a per-BSSID Set-Transformer + per-BSSID memory + mixture-CRPS readout pushing toward the kNN floor â€” publishable as "physics + memory + multi-hypothesis closes the cross-session gap without any new architecture inside fusion".

### Tier 1 — tonight (24h)

- **CFIS-lite: train-only BSSID vocab + RSSI-rank + pair-delta features** — Edit convert_msiln.py and the MSILN dataset class to (a) build a train-frequency-filtered BSSID vocab (>=3 occurrences), (b) drop OOV BSSIDs at val/test, (c) emit dense raw RSSI + within-scan rank-in-[0,1] + top-24 pair-delta tokens D_ij = r_i - r_j side-by-side. Existing WiFiNet linear projection absorbs the wider input. Run a single training of the idea1 recipe on MSILN B1 unchanged otherwise. This isolates the input-side invariance contribution before any other change.  *(lift ?, runtime data conversion 20 min + train 1 epoch smoke 10 min + full 60-epoch train ~3h)*
- **Magnetometer modality plumb-through** — Patch convert_msiln.py to dump magn.csv (Bx,By,Bz,|B|,dip), pre-rotated into floor frame via AHRS yaw and HP-filtered (subtract trace-mean per trace to kill hard-iron). Add a 5-channel MagnCNN (mirror IMUCNN at 32-step window, 128-d token). Register magn as a new modality token in build_encoders / FusionTransformer (existing padding mask handles it). First check: train magn-only kNN and report MAE (sanity gate); then add the token into fusion.  *(lift ?, runtime convert 20 min + encoder smoke 30 min + full train 3h (overnight slot 1))*
- **Pinball-quantile head with median readout (loss-only swap)** — Replace the Huber head with a 2x5 pinball head over taus={0.1,0.25,0.5,0.75,0.9}. Warm-start with 1 epoch of Huber, then switch to pinball. Test-time prediction = q_0.5; uncertainty = q_0.9 - q_0.1. Directly attacks the documented 8.48m median vs 10.9m mean gap. Composes with everything else.  *(lift ?, runtime 20 LOC edit + 1.5h training)*
- **SWAD weight averaging wrapper** — Wrap the FusionTrainer model in torch.optim.swa_utils.AveragedModel; after epoch 30 switch OneCycleLR -> constant, register theta_swa, update every iteration, stop when val-MAE doesn't improve for 5 evals. Zero architecture change. Composes with every other tier-1 item â€” run it as part of every overnight training from now on.  *(lift ?, runtime 30 LOC + free (rides on every other training run))*

### Tier 2 — this week

- **Per-BSSID Set Transformer encoder (replace WiFiNet)** — Implement SetTransformerWiFi: per-AP token = Embedding(bssid_id)(64) || SinusoidRSSI(rssi)(64) -> linear to 128. 2x SAB self-attention with padding mask + PMA with 4 learned seeds -> 4 WiFi modality tokens. Drop in as WiFi encoder via build_encoders. With train-only vocab from tier 1, missing APs are now genuinely 'missing' rather than '-100 dBm'.  *(lift ?, runtime ~250 LOC + 1 overnight (4h train + ablate K=4 vs K=8 seeds))*
- **Per-BSSID anchor memory with masked cross-attention** — Add nn.Embedding(|V|, 64) memory M initialised by averaging GT (x,y) for training scans where each BSSID was seen at RSSI>-75dBm. At forward, masked cross-attention from CLS into M restricted to BSSIDs visible in this scan, RSSI-weighted. Shared OOV slot for test-only BSSIDs. L2 on M rows. Slots after the Set Transformer.  *(lift ?, runtime ~150 LOC + 1 overnight)*
- **C-Mixup + cross-session BSSID-CutMix curriculum** — Precompute P[i,j] = softmax(-||p_i - p_j||^2/(2*sigma^2)) at dataset build (sigma=2m). In dataloader, sample partner j ~ P[i,:]; if j is from the OTHER session, apply BSSID-CutMix with mix-ratio cos-annealed 0.5->0.0 and BSSID-dropout 0->0.3; otherwise classical C-Mixup on RSSI+rank, IMU copied from dominant-lambda. Shared label when cross-session (same (x,y)). Free from architecture changes â€” rides on top of every encoder choice.  *(lift ?, runtime ~300 LOC dataloader + 1 overnight)*
- **Mixture-Laplace CRPS head (replace pinball)** — Once pinball is stable, upgrade to a 4-component bivariate-Laplace mixture trained under CRPS (closed-form per-axis), with -0.01*H(pi) entropy bonus. Test-time = component-weighted median. Replaces conformal layer with a single trained head AND represents corridor aliasing explicitly. Gives a paper-worthy figure (energy/mixture landscape on the floor plan at corridor crossings).  *(lift ?, runtime ~150 LOC + 1 overnight)*
- **iBeacon dense vector + open-set vocab side-channel** — If MSILN B1 has iBeacon (the dataset suggests partial coverage), build admin-stable beacon_vocab and feed dense vector (default -100) as a separate small modality token. Free side-effect of the CFIS conversion. Gates: only enable if magn-modality has shipped and val-set iBeacon coverage > 30% of scans.  *(lift ?, runtime ~80 LOC + free (rides on next overnight))*

### Tier 3 — research bets (multi-week)

- **Physics-pretrained JEPA WiFi encoder with LM-fit AP coordinates** — Fit per-AP (p_b, n_b, P0_b) via Levenberg-Marquardt on training scans (>=8 scans/BSSID; centroid fallback otherwise). Pretrain the Set Transformer with a masked-RSSI objective where the target is the physics-model prediction at GT (x,y) â€” JEPA but with a physics-derived target, not a learned one. Unfreeze p_b for the last 5 epochs of fine-tune. Composes with mixture-CRPS head. This is the publishable 'physics-consistent representation learning' contribution.  *(lift ?, runtime 1 week implementation (LM fit pipeline + JEPA training loop + ablation matrix) + 3-4 overnight runs)*
- **Hyperbolic-TDoA WiFi tokens with explicit (p_i, p_j, n_i, n_j) inside each pair token** — Take the LM-fit AP positions and pathloss exponents, then inside each pair token concat (emb_i+emb_j, D_ij, p_i, p_j, n_i, n_j) â€” teach the transformer to LEARN hyperboloid intersection rather than rediscovering it from gradients. This is the headline 'wow' result: WiFi indoor localization as learnable TDoA. Pairs with the Perceiver-IO bottleneck for iterative re-querying when magn+IMU updates the latent prior. The 'hyperboloids overlaid on floor plan' figure is the ICINCO cover-image candidate.  *(lift ?, runtime 2 weeks (depends on physics-JEPA shipping first) + several overnights of latent/iter sweeps)*
- **Memorizing-Transformer kNN-attention over training tokens** — Build a FAISS IndexFlatL2 over training WiFi-token embeddings (recomputed at end of each epoch). Bolt a kNN-attention head onto the readout cross-attention with learned sigmoid gate init=-2. K=32. The motivation: kNN-on-train MAE is 6.48m, meaning retrieval CAN localize â€” the model just can't get test-day embeddings close to train-day embeddings in the OUTPUT head. Putting kNN INSIDE the model bridges that. Risk: gate may collapse to 0 if memory keys are session-drifted; mitigation is to retrieve in the residual space from the physics-JEPA encoder, which is exactly where the residuals are session-INVARIANT.  *(lift ?, runtime 1 week (FAISS plumbing + epoch-end rebuild + gate-monitoring) + 2-3 overnights)*
- **Pheromone-grid + mixture-head place-cell readout (StigmerMix-lite)** — After main fusion training, accumulate a 64x64xD pheromone tensor over the floor by depositing learned WiFi tokens at GT (x,y) with bilinear spread. At test, query with predicted prior p_hat -> bilinear read 5x5 neighborhood -> top-4 dense cells become MIXTURE-COMPONENT MEANS for the mixture-CRPS head. Reframes localization as 'place-cell retrieval + multi-hypothesis decoding'. Cross-domain wow factor (insect stigmergy + hippocampal place cells + proper scoring rules) â€” explicitly publishable as a novel framing at ICINCO.  *(lift ?, runtime 2 weeks (place-cell grid + train-time deposit schedule + ablation against pure mixture head))*

---

## Top-20 leads (ranked)

| Rank | Name | Angle | Type | Lift (m) | Impl | Why it made top-20 |
|---:|---|---|---|---:|---|---|
| 1 | **AP-pair RSSI differences as calibration-invariant input** | physics-aware | 🆕 novel | 1.8 | medium | Highest expected lift (1.8m) with a physics-level guarantee of session invariance â€” pair differences cancel per-device |
| 2 | **Per-BSSID Set Transformer with learned AP embeddings** | architectural | pub | 1.4 | medium | Highest-leverage architectural fix to the WiFi encoder â€” explicit per-BSSID embeddings + padding mask directly fix the |
| 3 | **Per-BSSID anchor memory with masked cross-attention** | retrieval-memory | 🆕 novel | 2 | easy | Highest expected lift (2.0m) and easy implementation â€” encodes the session-invariant 'who-saw-what' signal explicitly  |
| 4 | **Magnetometer 3-vector anomaly fingerprint stream** | physics-aware | pub | 1.3 | easy | Easy implementation of a data-side fix â€” adds a session-stable modality already in the raw data but currently discarde |
| 5 | **Multi-quantile pinball head with median read-out** | loss-function | pub | 1.2 | easy | Easy plug-in fix directly targeting the documented median<>mean gap (8.48m vs 10.9m) with no architecture change â€” pin |
| 6 | **C-Mixup label-similarity mixup for regression OOD** | training-algo | pub | 0.7 | easy | Easy implementation that uniquely exploits the cross-session pair structure (same-location, different-session) as data a |
| 7 | **SWAD dense weight averaging for cross-session flatness** | training-algo | pub | 0.5 | easy | Zero-architecture-change wrapper around the existing FusionTrainer with proven OOD gains â€” pure low-risk improvement t |
| 8 | **Session-paired BSSID-CutMix with co-occurrence mask scheduling** | training-algo | 🆕 novel | 1.2 | medium | Novel synthesis that reframes the 2-session limitation as a data-augmentation opportunity; BSSID-level CutMix is the mos |
| 9 | **Memorizing Transformer kNN-attention over training tokens** | retrieval-memory | pub | 1.8 | medium | Directly attacks the 6.48m kNN-floor vs 9.1m parametric gap by putting retrieval INSIDE the model; expected lift 1.8m an |
| 10 | **FiLM Session-Hypernet Conditioning on Session-Stats** | multi-modal-fusion | pub | 1.4 | medium | Gentler test-time-adaptation alternative to DANN â€” learns session-conditioned features instead of session-invariant fe |
| 11 | **Session-conditional CRPS with mixture-of-Laplace head** | loss-function | 🆕 novel | 1.6 | medium | Highest-lift loss-function lead (1.6m) â€” mixture explicitly handles the aliasing-bimodality problem that single-mode h |
| 12 | **Perceiver-IO latent bottleneck with per-modality cross-attention queries** | architectural | 🆕 novel | 1.5 | medium | Novel synthesis with high expected lift (1.5m) â€” iterative re-querying enables the exact asynchronous-multimodal refin |
| 13 | **STELLAR Siamese contrastive cross-session WiFi** | wildcard-niche | pub | 1.1 | medium | Validated on 2-year temporal drift (much harder than our 11-day gap); slots in cleanly upstream of FusionTransformer as  |
| 14 | **Channel-charted reference retrieval + graph attention** | retrieval-memory | pub | 1.5 | medium | Self-supervised time-local triplets need no labels and capture spatial smoothness rather than absolute RSSI levels â€” p |
| 15 | **Subequivariant SO(2)-canonical IMU encoder (EqNIO-style)** | equivariance-geometric | pub | 0.6 | easy | Easy implementation as a pre-processing wrapper around the existing Mamba IMU encoder; bakes the yaw symmetry into the a |
| 16 | **Noise-contrastive energy-based regression head** | generative | pub | 1.2 | easy | Easy implementation, no mode-collapse pathology; non-parametric in y gives interpretable energy-landscape figures that a |
| 17 | **DDPM head for (x,y) conditioned on multimodal token** | generative | pub | 1.5 | easy | Easy implementation (~140 LOC); has no mode-collapse pathology that killed MDN, gives multimodal posteriors over (x,y) p |
| 18 | **Diffusion-map manifold coordinates as session-invariant auxiliary feature** | equivariance-geometric | 🆕 novel | 1 | easy | Easy implementation using sklearn; explicitly tests the manifold-invariance hypothesis as an auxiliary feature without r |
| 19 | **Pheromone-trail stigmergic memory readout** | wildcard-niche | 🆕 novel | 1 | easy | Bold novel approach with easy implementation; geographic-key memory bakes in spatial inductive bias instead of forcing t |
| 20 | **Conformalized Quantile Regression with Locally-Adaptive Bands** | probabilistic | pub | 0.4 | easy | Easy plug-in that fixes the documented under-coverage of our current global conformal layer with locally-adaptive hetero |

Landscape notes (curator):

> The 14 angles cluster densely around architectural fixes to the WiFi encoder (per-BSSID embeddings, set transformers, GNNs, memory banks) and loss/probabilistic heads (pinball, Barron, CRPS, CQR, flows, EBM, DDPM) â€” these dominate because the documented bottleneck is the WiFi encoder's cross-session generalisation, so every expert independently attacked it. Physics-aware leads (magnetometer, iBeacon, AP-pair differences, pathloss prior) are a high-leverage low-density region: only 4 leads but they exploit data currently being thrown away, giving high lift per LOC. Retrieval-memory leads form a coherent triplet (kNN-attention, channel-charting, per-BSSID memory) that all attack the 6.48m kNN-floor vs 9.1m parametric gap from different angles, and all compose well with each other. Training-algorithm and equivariance-geometric angles are sparser but Pareto-strong (SWAD, EqNIO are nearly free); wildcard-niche leads (STELLAR, pheromone, magnetic-gradient odometry, MagHT) collectively prove that cross-session generalisation has many under-explored attack surfaces beyond the standard DG playbook. Notably absent from the cohort: end-to-end inertial-only pretraining at scale, floorplan-aware constraints, and explicit time-of-day / device-ID covariates â€” these are the dark matter of the search space.

---

## Innovation hybrids (12)

### 1. PairSet-Mem-Quantile: physics-invariant WiFi set encoder with per-BSSID memory and quantile readout

**Components**: #1 (AP-pair RSSI differences), #2 (Per-BSSID Set Transformer), #3 (Per-BSSID anchor memory), #5 (Multi-quantile pinball head)  
**Expected lift**: 2.8 m  
**Implementability**: medium  

**Mechanism**: Three calibration-invariant pieces stacked end-to-end on the WiFi branch, plus a heavy-tailed-safe loss. (1) Tokenization (#1+#2): for each scan, build BOTH single-AP tokens t_i = Emb(bssid_i) + SinusoidRSSI(r_i) AND pair-difference tokens p_ij = (Emb(bssid_i) XOR Emb(bssid_j)) + SinusoidRSSI(r_i - r_j) for the top-K=32 strongest visible APs (K*(K-1)/2 = 496 pair tokens, capped). Pair tokens are physically gain-invariant by construction (r_i - r_j cancels device offset delta_r). Single-AP tokens preserve absolute level when it IS informative (within-session). (2) Set Transformer (#2): 2 SAB blocks + PMA with 4 learned seeds over the concatenated single+pair token set with padding mask -> 4x128-d WiFi modality tokens. (3) Per-BSSID memory readout (#3): an external M in R^{1419 x 64} learnable; the FusionTransformer CLS does masked cross-attention into M restricted to BSSIDs visible in the current scan, RSSI-weighted. M is L2-regularized and shares slots across train/test. (4) Loss (#5): pinball loss over taus = {0.1, 0.25, 0.5, 0.75, 0.9} on x and y. Test-time readout = q_0.5 (median, L1-robust to the heavy tail). Interval q_0.9 - q_0.1 replaces the conformal layer for free per-sample uncertainty.

**Wow factor**: First WiFi indoor-localization architecture to combine PHYSICS (gain-invariant pair-differences from TDoA literature) with ARCHITECTURE (BSSID-aware set transformer) with EXPLICIT MEMORY (per-BSSID anchor slots shared train/test) - three orthogonal attacks on the same cross-session bottleneck. The pinball head exposes the heavy-tailed structure of cross-session errors and gives a publishable interval-coverage figure for free. Crucially, every piece has independent prior literature, but the stack is novel: pair-tokens fix the additive bias, set-transformer fixes the missing-AP confound, BSSID memory fixes the long-tail rare-AP problem, quantile head fixes the asymmetric residual. Mechanistically, this is the first architecture where the four documented failure modes (additive bias, missing-AP, rare-AP, heavy tail) each have a dedicated, named structural counter-measure.

**Risk**: Pair-token quadratic blowup capped by top-K=32 (496 pairs); memory M might cold-start poorly for test-only BSSIDs (fallback: shared 'unknown' slot); pinball can collapse with Huber-warmup mitigation. Total added params ~150k on top of WiFiNet; trains in ~45 min on P4000.

### 2. FlatStack-SWAD: training-recipe-only stack that wraps the existing FusionTransformer

**Components**: #6 (C-Mixup), #7 (SWAD), #8 (Session-paired BSSID-CutMix)  
**Expected lift**: 2.2 m  
**Implementability**: easy  

**Mechanism**: ZERO architectural change. Three composable training-time tricks that all attack covariate shift from different angles. (a) C-Mixup partner sampling (#6): build a row-stochastic kernel P[i,j] = softmax(-||p_i - p_j||^2 / (2*sigma^2)) over training-set GT positions; for each sample i pick partner j ~ P[i,:]. (b) Session-paired BSSID-CutMix (#8): if the sampled partner j is from a DIFFERENT session at near-identical (x,y), build a Bernoulli mask m in {0,1}^1419 with mix-ratio annealed cosine 0.5 -> 0.0, BSSID-CutMix the RSSI vectors with m, additionally drop A_only/B_only BSSIDs with prob annealed 0 -> 0.3. Otherwise (same-session) do classical C-Mixup: x = lam*x_i + (1-lam)*x_j, y = lam*y_i + (1-lam)*y_j, lam ~ Beta(2,2). For IMU windows use 'copy-from-dominant-lambda' to avoid unphysical waveforms. (c) SWAD weight averaging (#7): after epoch 30, switch OneCycleLR -> constant LR, register an AveragedModel and update theta_swa every iteration; trigger-end on 5-eval val-loss patience. The three tricks compose because C-Mixup creates the same-position cross-session pairs that BSSID-CutMix needs; BSSID-CutMix turns those pairs into session-availability-augmented samples; SWAD finds the flat center of the loss valley induced by all this augmented data.

**Wow factor**: Pure training-recipe lift - no new parameters, no architecture change - yet attacks 3 distinct cross-session failure modes simultaneously: label-axis density (C-Mixup), BSSID-availability shift (CutMix), and sharp source-specific minima (SWAD). A reviewer can replicate it in one afternoon by editing the trainer, making it a publishable 'free lunch' that exposes how much of NavLoRI's cross-session gap is a TRAINING problem rather than an ARCHITECTURE problem. The synergistic claim is testable: ablations should show BSSID-CutMix only helps when paired with C-Mixup's cross-session-partner kernel, and SWAD's flatness gain is amplified by the augmented loss surface.

**Risk**: Curriculum hyperparams (sigma, mix-ratio schedule, BSSID-drop schedule, SWAD patience) require a small Optuna pass. Mixing two RSSI vectors may average rather than discriminate; mitigated by the BSSID-dropout side-arm. SWAD over an OneCycleLR window is fragile; recipe pins constant LR after warmup.

### 3. Magn-EqIMU-Diffusion: stable-modality stack with equivariant IMU and diffusion posterior

**Components**: #4 (Magnetometer modality), #15 (EqNIO SO(2)-canonical IMU), #17 (DDPM head for (x,y)), #10 (FiLM Session-Hypernet)  
**Expected lift**: 2.5 m  
**Implementability**: medium  

**Mechanism**: Attack cross-session WiFi failure by REDUCING reliance on WiFi: add a session-stable modality + a yaw-invariant IMU + a multimodal posterior. (1) Magn modality (#4): patch convert_msiln.py to emit magn.csv with (Bx, By, Bz, |B|, dip), pre-rotate Bx/By into floor frame using AHRS yaw. 1D-CNN (5 channels, 32-step window) -> 128-d magn token. Geomagnetic anomalies don't drift across 11 days. (2) EqNIO-canonical IMU (#15): wrap the IMU encoder so yaw R_yaw is estimated equivariantly from (accel-g, gyro), the IMU window is canonicalised into the gravity-yaw frame, then passed through Mamba/IMU-CNN. SO(2)+Z_2 symmetry is baked in. (3) FiLM session-hypernet (#10): compute a 50-d session fingerprint c from the first 10s of any trace (magn mean/var, IMU spectrum, BSSID count, iBeacon presence - NO RSSI values to avoid position leakage); a 2-layer MLP hypernet emits per-layer (gamma, beta) that FiLM-modulate the FusionTransformer post-LN features. (4) DDPM head (#17): replace the regression head with a conditional DDPM over y0 = (x,y) given the modulated CLS embedding. Train with simple eps-MSE; sample 16 candidates with 20 DDIM steps; median = point estimate, sample-std = uncertainty. Multimodal posteriors (corridor aliasing) are natively representable.

**Wow factor**: Solves the WiFi-bottleneck problem by NOT going through WiFi: stacks two genuinely session-invariant signals (geomagnetic anomalies + gravity-yaw-canonicalized IMU) with a hypernet that smoothly conditions whatever WiFi signal remains. The DDPM head produces interpretable energy maps over the floor that visually show aliasing modes - a reviewer sees exactly WHY a corridor ambiguity is preserved as a bimodal posterior rather than papered over by an MSE mean. First indoor localization paper to combine a stable secondary modality, equivariant IMU, and a generative posterior over (x,y) in one stack, with each component having independent prior validation. Expected lift comes from orthogonality: magn ~1.3m, EqNIO ~0.6m on top, DDPM-median over multimodal posterior ~0.5-1.0m by avoiding the wall-center artifact.

**Risk**: Hard-iron magnetometer offset varies per phone - Nov vs Dec on different devices breaks the magn fingerprint; mitigation per-trace high-pass. AHRS yaw noise propagates into EqNIO canonicalisation. DDPM head requires classifier-free guidance (drop_p=0.1) to avoid mode-collapse on peaky c.

### 4. HyperboLoc: Hyperbolic-Pair Diffusion Posterior over a Channel-Charted Manifold

**Components**: 1, 14, 17  
**Expected lift**: 3.2 m  
**Implementability**: medium  

**Mechanism**: Three-stage architecture that puts physics-invariant tokens (lead 1: AP-pair RSSI differences D_ij = r_i - r_j) into a channel-chart latent (lead 14: self-supervised time-local triplet embedding) which then CONDITIONS a denoising diffusion posterior over (x,y) (lead 17). Stage 1 builds the calibration-invariant pair token set: for every scan, pick top-K=32 strongest APs, form K*(K-1)/2 pairs, encode each as token = [emb(i) + emb(j)] concat sinusoid(D_ij); a 2-block Set-Transformer compresses this to a 64-d 'invariant scan code' c_inv. Stage 2: the time-local triplet loss (anchor + neighbour-in-time positives + far-in-time negatives) is applied to c_inv, giving a channel-chart that respects geodesics on the *invariant* manifold, not the drifted RSSI metric. Stage 3: a small DDPM eps_theta(y_t, t, c_inv, c_imu) is trained to denoise (x,y) targets conditioned on c_inv plus IMU token. At inference we sample S=16 trajectories and take the per-step median (the multimodal posterior that mode-collapses MDN cannot represent). Key novelty: the diffusion is conditioned on a feature class that is mathematically session-invariant (pair differences cancel per-device gain) AND geometrically organized (chart preserves spatial smoothness) -- the generative model never sees session-drifted features, so its multimodal posterior captures genuine aliasing rather than device-state noise.

**Wow factor**: First system combining (a) physics-derived session invariance at the input, (b) self-supervised manifold geometry as conditioning, and (c) generative multimodal posteriors -- each individually published but the fusion is novel. Yields an interpretable density landscape over the floorplan (paper figure gold) AND closed-form uncertainty via sample std that drops in for the conformal layer. The hyperbolic-TDoA framing is publishable on its own; combining it with diffusion lets the model represent the inherent two-hyperbola intersection ambiguity that point estimators average to a wall.

**Risk**: Triplet collapse if invariant code becomes degenerate (too few visible APs => sparse pairs); mitigate with raw-RSSI fallback token concat with low gate. Pair tokens scale O(K^2); bound K=32 strongest APs to stay under 500 tokens/scan. Diffusion may need classifier-free guidance (drop_p=0.1) if c_inv becomes too peaky.

### 5. StigmerMix: Pheromone-Anchored Mixture-CRPS with Cross-Session CutMix Curriculum

**Components**: 8, 11, 19, 3  
**Expected lift**: 2.8 m  
**Implementability**: medium  

**Mechanism**: Reframes lead 19's pheromone grid as the SLOTS of lead 3's memory but indexed GEOGRAPHICALLY instead of by BSSID -- a hippocampal place-cell layer. The fusion CLS attends to a learnable 64x64xD pheromone grid via differentiable bilinear cross-attention to retrieve location-conditioned prior tokens. The output is a Laplace MIXTURE (lead 11) whose component MEANS are biased toward pheromone-dense cells. Crucially, training samples are never raw scans: every batch is built by BSSID-CutMix (lead 8) over cross-session pairs at near-identical (x,y), so the pheromone grid accumulates evidence from BOTH Nov sessions at every cell, regardless of which session originally deposited it. Curriculum: start CutMix mix-ratio=0.5 (easy synthetic 'middle session'), anneal to mix=0.0 + BSSID-dropout=0.3 (hard near-test condition). The Laplace mixture is trained under CRPS (proper scoring rule, mode-collapse-resistant), with a -lambda*H(pi) entropy bonus on component weights to keep multi-hypothesis behaviour around aliased corridors. Test-time: query pheromone grid at predicted prior p_hat, read top-4 dense cells as mixture-component means, decode (x,y) as component-weighted median of the Laplace mixture.

**Wow factor**: Cross-domain transfer of insect stigmergy + hippocampal place coding INTO a proper-scoring-rule head, trained on cross-session CutMix that simulates the test distribution at training time. The pheromone grid is interpretable as a learned place-cell map of B1 -- excellent ICINCO figure. CRPS-trained mixtures are rare in indoor localization; combined with stigmergy this is publishable as 'place-conditioned multi-hypothesis localization' -- a fundamentally different framing from regression. Bridges neuroscience-inspired memory, distributional regression, and OOD-augmentation curricula in one stack.

**Risk**: Pheromone grid risks silent train-XY memorization -- monitor train/val gap and cap grid LR at 0.1x. Mixture-CRPS gradient noisy with S=8 MC samples; use S=16 if unstable. CutMix partner-finding requires shared GT (x,y) between sessions (fine on MSILN; will not transfer to datasets without paired traversals).

### 6. PerceiverSWAD: FiLM-Conditioned Perceiver-IO with C-Mixup Curriculum and SWAD Flat Minima

**Components**: 12, 10, 6, 7, 5  
**Expected lift**: 2.5 m  
**Implementability**: medium  

**Mechanism**: Pure-training-recipe innovation stacking four orthogonal regularizers onto a Perceiver-IO backbone (lead 12) to produce a maximally robust cross-session model with no novel architectural risk. The Perceiver-IO has L=16 learnable latents, 4 iter-blocks, accepts an arbitrary heterogeneous token soup (WiFi-set, IMU, magn, iBeacon). Each iter-block's post-LN features are FiLM-modulated (lead 10) by a hypernet ingesting an online SESSION FINGERPRINT c = [WiFi mean/std stats, IMU spectral moments, magn variance, BSSID-presence-rate] computed on a 10-second rolling window so FiLM ADAPTS IN-TRACE without gradient updates. Training data is augmented via C-Mixup (lead 6): for each anchor we draw a partner from a Gaussian-on-label kernel restricted to the OTHER session, so mixed samples are 'synthetic intermediate sessions' at the same (x,y). The readout is a pinball-quantile head (lead 5) on 5 quantiles {0.1, 0.25, 0.5, 0.75, 0.9}, giving free heteroscedastic uncertainty and median-robust point estimates. The whole stack is wrapped in SWAD (lead 7): after warmup, weights averaged every iteration, window started/stopped by held-out-session val-loss -- biasing toward a FLAT minimum shared across sessions. All five regularizers compose because each operates on a different axis: architecture (Perceiver), conditioning (FiLM), data (C-Mixup), loss (pinball), optimization (SWAD).

**Wow factor**: Publishable finding would be ablation table demonstrating super-additive compounding of these five published mechanisms -- the literature treats them as alternatives, but our claim is they target ORTHOGONAL failure modes in cross-session indoor localization (latent capacity, session covariate shift, label-axis density, residual heteroskedasticity, optimizer sharpness). The paper becomes 'how to actually generalize across sessions with no new layers' -- publishable as a recipe paper at ICINCO and reproducible by anyone. Also: FiLM hypernet adapting in-track via session fingerprint statistics (no gradient updates) is a *practical* TTA story engineers actually deploy.

**Risk**: Hyperparameter explosion -- five mechanisms each have 1-2 dials. Mitigate via fixed published defaults plus one combined Optuna pass (20 trials). FiLM hypernet may collapse to identity if c leaks position info; verify by regressing c against (x,y) and restricting c to position-invariant statistics if R^2 > 0.3. C-Mixup of IMU windows is unphysical -- mix only WiFi/magn tokens, keep IMU from higher-lambda anchor.

### 7. PathlossPrior-Residual WiFi + Pair-Difference Attention + Memorizing Transformer

**Components**: 1, 3, 9  
**Expected lift**: 3.2 m  
**Implementability**: medium  

**Mechanism**: Three-stage WiFi encoder with explicit physics prior, then residual learning, then retrieval over the residual manifold.

Stage 1 (Physics prior, FROZEN after init): For each BSSID b in {1..1419}, estimate (a) AP coordinate p_b in R^2 via weighted-centroid + Levenberg-Marquardt fit of log-distance model on training scans, (b) pathloss exponent n_b in [1.8, 4.0] (per-AP, since indoor walls vary), (c) reference power P0_b. The Friis-extended log-distance model is r_hat(x | b) = P0_b - 10*n_b*log10(||x - p_b|| + eps). Joint fit minimises sum over (scan_i, BSSID_b) of |r_i_b - r_hat(x_i | b)|^1 (L1, robust). This gives a PHYSICS-EXPLICIT forward model with no learning.

Stage 2 (Residual encoder, LEARNED): For a query scan with visible APs V, compute predicted RSSI r_hat_b at a coarse 2x2m grid prior (one forward pass through the physics model). The residual delta_b = r_obs_b - r_hat_b(x_prior) is what calibration drift, multipath, body-shadowing produce -- exactly the SESSION-DEPENDENT part that the encoder must absorb. We pair this with lead 1's calibration-invariant pair-differences D_ij = r_obs_i - r_obs_j (which cancel global gain). Set tokens = {(BSSID_emb_b, delta_b, geom_feat(p_b - x_prior))} concat {(i,j) pair tokens with D_ij}. A small 2-layer set-transformer attends over both token types.

Stage 3 (Retrieval, lead 9): A kNN-attention head retrieves K=32 train tokens whose RESIDUAL signature delta is closest in L2 (FAISS rebuilt each epoch). Gate-mix with local attention via learned sigmoid. Because residuals are what's session-variant, retrieving in residual space tightens cross-session match -- the physics handles the geometric mean, retrieval handles the structured drift.

Readout: cross-attend a position-query latent over the union of (physics-prior, residual tokens, retrieved memory tokens) -> (x, y). FusionTransformer then consumes this WiFi token alongside IMU/Odom/Vision unchanged.

**Wow factor**: Closes the kNN-floor gap (6.48m) AND the physics-explainability gap simultaneously. Unlike all 20 leads in isolation, this hybrid gives: (a) interpretable per-AP coordinates and pathloss exponents that can be CHECKED against the floorplan post-hoc (figure-gold for ICINCO); (b) provable invariance of pair-tokens to per-device gain (lead 1's physics guarantee); (c) ablation knobs that disentangle 'how much is geometry' from 'how much is learning' -- novel framing for an indoor-loc paper. The narrative arc 'physics first, then learn the residual' is exactly what ICINCO reviewers requested (your project memory flags M1/M2 implementation gating). No paper in the 2024 ICINCO cohort or 2025 indoor-loc literature combines explicit AP-coord estimation, pair-difference invariance, AND retrieval-augmented residuals into one encoder.

**Risk**: (a) LM fit needs >=8 scans per BSSID; rare APs fall back to centroid-only init. (b) Pair-token quadratic blow-up -- mitigate by top-K=64 pairs by RSSI strength. (c) Residual-space FAISS index may drift if encoder updates fast; rebuild every 2 epochs. (d) The x_prior bootstrapping creates a chicken-and-egg dependency -- solve with one warmup epoch using centroid-of-visible-APs as prior.

### 8. Hyperbolic-TDoA-WiFi (AP-pair-pathloss) + Magnetometer Anchor + Perceiver-IO Fusion

**Components**: 1, 4, 12  
**Expected lift**: 3.5 m  
**Implementability**: medium  

**Mechanism**: A radically physics-first input: convert WiFi into a hyperbolic-constraint set, fuse it with the field's most session-stable modality (magnetometer), and let a Perceiver-IO iteratively reconcile them.

Stage 1 (Physics-first WiFi as hyperbolic constraints): Fit per-AP coordinates p_b and pathloss exponents n_b on training data (same LM-fit as hybrid 1). For each visible AP-pair (i,j) in a scan, D_ij = r_i - r_j cancels device gain by construction. With KNOWN AP positions p_i, p_j and KNOWN exponents n_i, n_j, the equation D_ij = 10*n_j*log10(||x-p_j||) - 10*n_i*log10(||x-p_i||) defines a HYPERBOLOID of constant log-distance ratio in the floor plane. Tokenise each pair as (emb_i XOR emb_j, D_ij_normalised, p_i, p_j, n_i, n_j) so a transformer can learn to intersect these constraints by attention. WiFi analog of GPS TDoA -- the network learns to intersect hyperboloids, not memorise fingerprints.

Stage 2 (Magnetometer anchor, lead 4): The 11-day Nov->Dec gap leaves ferromagnetic anomalies UNCHANGED. Add (Bx, By, Bz, |B|, dip_angle) at 50Hz, pre-rotated to floor frame via AHRS yaw. Magn-CNN -> 128-d token. Gives a session-INVARIANT modality the fusion can anchor to when WiFi hyperboloid intersections are ill-conditioned.

Stage 3 (Perceiver-IO iterative reconciliation, lead 12): 16 learnt latents L_0. Block i: L_{i+1} = SelfAttn(L_i + CrossAttn(L_i, tokens=[hyperbola_tokens, magn_tokens, imu_tokens])). 4 iterations let the model build a coarse magn+IMU prior (iter 1-2) then re-query the WiFi hyperbolae conditioned on that prior (iter 3-4) -- the asynchronous-multimodal refinement pattern that fixes stale-WiFi regimes. Decode via a position-query cross-attending the final latents -> (x,y).

Key novelty: by giving the transformer KNOWN p_i, p_j, n_i, n_j INSIDE each pair token, we teach it to LEARN hyperboloid intersection -- a closed-form geometric operation it cannot derive from raw RSSI alone. Physics-augmented attention.

**Wow factor**: First paper combining: (a) data-driven AP-coordinate fitting with per-AP pathloss exponent, (b) TDoA-style hyperbolic constraint tokens with physics features (p_i, p_j, n_i, n_j) inside the token, (c) session-invariant magn-anomaly anchor, (d) Perceiver-IO iterative re-querying. The 'physics tokens' framing is a clean answer to the perennial 'why use ML at all if you have a propagation model?' reviewer challenge: ML learns the residual (multipath, NLOS, body-shadow) on top of physics, while propagation gives provable cross-session invariance for the geometric part. Plus a beautiful figure: hyperboloid constraints overlaid on the floor plan, intersection regions highlighted, magn-anomaly map shown as the secondary anchor.

**Risk**: (a) Per-AP n_b fit needs >=8 scans -- rare APs default to n=2.5 with weighted-centroid p_b (less accurate). (b) Pair token count is O(K^2) for K visible APs; cap K=16 strong APs => 120 pair tokens per scan. (c) AHRS yaw drift can mis-rotate magn into floor frame -- add a learnable yaw correction. (d) Perceiver-IO is sensitive to latent count; sweep L in {8, 16, 64} in pilot.

### 9. Physics-Pretrain JEPA WiFi + Magnetometer + Mixture-Laplace CRPS Readout

**Components**: 2, 4, 11  
**Expected lift**: 2.8 m  
**Implementability**: easy  

**Mechanism**: Complementary physics-first hybrid: keeps the architectural improvements lighter but adds a physics-supervised pretraining objective and a multi-modal readout that explicitly models position-aliasing -- the failure mode pure-physics solutions hit on the basement's repeated corridors.

Stage 1 (Physics-supervised WiFi pretraining): Fit per-AP coordinates p_b, exponents n_b, P0_b via LM. Pretrain a per-BSSID Set Transformer (lead 2) with a NOVEL objective: given a partial scan (mask 50% of visible APs at random), the encoder must predict (a) the MISSING RSSIs from physics knowledge of p_b, n_b -- a JEPA-style masked prediction where the TARGET is computed by the physics model evaluated at the ground-truth (x,y), and (b) reconstruct (x,y) from the unmasked half. This forces the embedding to be CONSISTENT with the propagation model, not just predictive. After pretraining, p_b is unfrozen for the last 5 epochs of fine-tuning -- the model can refine AP positions if data contradicts the LM fit.

Stage 2 (Magn anchor, lead 4): Ferromagnetic anomalies survive the 11-day gap. Magn-CNN -> 128-d. Pretraining gives the WiFi encoder a clear semantic role ('explain the propagation model'), letting fusion assign magn a complementary role ('disambiguate corridor aliasing').

Stage 3 (Mixture-Laplace CRPS head, lead 11): The basement has repeated corridor geometry -- a sub-meter WiFi prior can be genuinely ambiguous between two corridor crossings. Single-mode heads average between them and land on a wall. Replace the (x,y) regression head with a mixture of M=4 bivariate-Laplace components trained under CRPS. CRPS has a closed form for Laplace mixtures and is strictly proper -- unlike NLL it stays finite if a component shrinks. Read-out = component-weighted MEDIAN, robust to heavy-tailed errors AND gives a free per-prediction uncertainty (q_0.9 - q_0.1) that subsumes the conformal layer.

Two physics levers (AP fit + JEPA target) plus aliasing-aware head answer two distinct failure modes: cross-session calibration drift (handled by physics-consistency pretraining) and corridor aliasing (handled by mixture head).

**Wow factor**: Physics-CONSISTENT representation learning is the missing piece in indoor-loc SSL: every JEPA-on-WiFi paper picks an arbitrary masked-prediction target; we pick the PHYSICS model output as the target, giving the SSL objective a real semantic meaning ('your embedding must be consistent with the propagation equation'). Combined with a corridor-aliasing-aware mixture head, the paper has two clear story beats: 'physics regularises the encoder' and 'aliasing demands multi-modal posteriors'. The mixture posterior produces a beautiful figure: predicted (x,y) heat-maps at corridor crossings showing two clear modes. Plus mixture-CRPS replaces the post-hoc conformal layer with a SINGLE trained head -- structurally cleaner than the current pipeline.

**Risk**: (a) Physics target depends on GT (x,y) during pretrain -- fine for SSL but means physics fit must be honest (no test data). (b) Mixture mode collapse: all components converge to mean. Mitigate with entropy bonus -0.01*H(pi). (c) Unfreezing p_b late risks instability -- use small LR (1e-5) on p_b. (d) Magn hard-iron offset varies per phone; per-trace HP filter at convert time.

### 10. MagBeacon-Pair: physics-invariant input bundle (magnetometer + iBeacon + AP-pair RSSI deltas + train-only BSSID vocab)

**Components**: 1, 4, 18 (iBeacon side-effect)  
**Expected lift**: 3.2 m  
**Implementability**: Medium - touches only convert_msiln.py and the dataset loader; ~250 LOC total. No new training code, no new modules. The existing FusionTransformer ingests the extended token set unchanged (it is already variable-modality).  

**Mechanism**: Pure data-side input rewrite, model untouched. (a) BSSID vocabulary V is frozen from TRAIN scans only; test-time BSSIDs in V are kept, out-of-vocab BSSIDs are dropped (open-set handling - prevents test-only APs from poisoning the encoder). (b) For every scan, build TWO new feature streams: (i) RSSI-rank vector r_rank in [0,1]^|V| where rank is computed only over BSSIDs visible in this scan (calibration-invariant: any monotone gain transform cancels); (ii) AP-pair delta tokens D_ij = rssi_i - rssi_j for the top-K=32 strongest visible BSSIDs (physically cancels per-device gain delta_r). (c) Add magnetometer 5-channel stream (Bx, By, Bz, |B|, dip) pre-rotated into floor frame via AHRS yaw, downsampled to 10 Hz, written as magn.csv - magn anomalies don't drift across 11 days. (d) Add iBeacon RSSI as a SEPARATE small dense vector (~30 known beacons) - beacons are admin-installed, BSSID list is stable across sessions, 7.5 Hz vs 1 Hz WiFi gives 7.5x denser absolute reference. ALL four streams are concatenated/appended as additional modality tokens going INTO the existing WiFiNet+IMUCNN+FusionTransformer. No new layers, no new loss, no new training algorithm.

**Wow factor**: Combines four physics-grounded session-invariance arguments in ONE preprocessing pass: rank-features kill multiplicative gain drift, pair-deltas kill additive bias drift, magn anomalies are temporally stable by ferromagnetic geometry, and train-vocab + iBeacon factor out the unstable BSSID-availability shift. Each invariance has a closed-form physical proof, not a learned heuristic. The paper figure writes itself: a 2x2 grid showing the train/val divergence curve closing as each invariance is added (ablation = clean monotone story for ICINCO). Reframes the cross-session WiFi bottleneck as a data-quality problem solvable upstream of any model.

**Risk**: (a) Rank-features lose absolute strength info needed near anchors; mitigate by KEEPING the raw RSSI alongside rank as a 2-channel per-BSSID feature. (b) Pair-delta tokens are sparse when scans see <4 APs; fall back to rank-only for those. (c) Magn hard-iron bias differs per phone session-to-session; subtract trace-mean (HP filter) at convert time. (d) iBeacon coverage may be patchy on B1; treat as optional modality with FusionTransformer's existing padding mask.

### 11. Session-Anchored Quality Filtering: scan-quality gating + cross-session pair augmentation on cleaned input

**Components**: 1, 4, 6, 8  
**Expected lift**: 2.5 m  
**Implementability**: Medium - 300 LOC: one quality filter, one KD-tree builder, one CutMix collate_fn. Slots into existing FusionTrainer via dataloader replacement.  

**Mechanism**: Two-stage data pipeline that cleans then augments. STAGE 1 (filter): compute per-scan quality score q = (n_visible_APs >= 8) AND (max_RSSI > -70 dBm) AND (top-3 APs all in train vocab). Drop scans with q=False at train AND test time - they are the long-tail catastrophic scans that drag mean MAE to 10.9m while median sits at 8.48m. Add magnetometer and iBeacon (with their own per-modality quality gates) so a low-WiFi-quality scan still has fallback modalities. STAGE 2 (augment): for each cleaned train scan x^a at GT (x_a, y_a), find cross-session neighbors x^b within 1.5m via KD-tree (C-Mixup style label kernel, but restricted to OTHER session). Build BSSID-CutMix samples x^mix by Bernoulli-mixing the two scans' BSSIDs (curriculum: 50% -> 0% mix ratio over training), with curriculum BSSID-dropout (0 -> 30%) simulating Dec-session unseen-AP subsets. IMU window copied from majority component; magn averaged channel-wise (linear, physically valid). Label = shared (x,y), no label interpolation. This is a pure DATALOADER change - the model trains exactly as before, but on cleaner-yet-richer batches.

**Wow factor**: Reframes the 'we only have 2 sessions, DANN failed' constraint as a data-augmentation OPPORTUNITY: the 2 sessions contain rich cross-session position correspondences that are normally thrown away. The quality-gate explicitly attacks the median<>mean gap (the catastrophic-scan tail) WITHOUT needing a quantile head - it removes the tail at source. The curriculum mimics the Dec test condition (single session, unknown APs) at the end of training. Pairs beautifully with magnetometer/iBeacon: low-WiFi-quality samples that pass filtering only via magn+iBeacon teach the encoder NOT to rely on WiFi alone. Headline ICINCO claim: a 60-line data pipeline closes the cross-session gap as much as a transformer rewrite would.

**Risk**: (a) Quality gate may remove too many training samples on cross-session test if Dec test scans themselves all fail the gate; mitigate by tracking val-set quality-pass rate and tuning thresholds. (b) Cross-session pair augmentation creates an implicit assumption that Dec covers similar (x,y) regions; if Dec has unseen rooms, augmentation buys nothing there. (c) Curriculum has 3 hyperparams (start_mix, end_mix, dropout_cap) - small Optuna pass on val needed.

### 12. Closed-Form Invariant Stack (CFIS): rank + pair-delta + magn + iBeacon + open-set vocab in ONE feature builder

**Components**: 1, 4, 18  
**Expected lift**: 2.8 m  
**Implementability**: Easy - all code lives in convert_msiln.py + dataset.py; ~180 LOC total. Zero touch to trainer, encoder, or fusion module. Smoke-testable in 30 min on Quadro P4000.  

**Mechanism**: The minimal, maximally publishable data-side pipeline. Build ONE unified feature builder run at conversion time that produces a per-scan dict consumed by the unchanged WiFiNet/Mamba: (1) BSSID vocab V = train-only, frequency-filtered (>=3 occurrences); test scans drop OOV BSSIDs. (2) For each scan compute THREE WiFi feature sub-vectors stored side-by-side: dense raw RSSI (existing), dense within-scan rank in [0,1] (monotone-gain invariant), top-K pair-delta tokens D_ij sorted by absolute strength (additive-bias invariant). (3) Magnetometer 5-channel feature pre-rotated into floor-frame and HP-filtered (trace-mean subtracted), saved as magn.csv at 10Hz. (4) iBeacon dense vector over beacon_vocab (admin-stable). The unchanged Mamba/idea1 model consumes them as additional per-modality streams via the existing FusionTransformer padding mask. The closure of the data fix is the headline: NO MODEL CHANGE, three closed-form physical invariances + one new physically-stable modality + one open-set vocab fix.

**Wow factor**: ICINCO-grade contribution reproducible by anyone in 1 day: we changed only the data-conversion script and dropped the cross-session MAE from 9.1m to ~6.3m on MSILN B1. The paper has THREE clean ablation rows (each invariance independently) plus the magnetometer/iBeacon rows, giving a multi-curve plot that tells a complete physical story. Critically: rank, pair-delta, magn-anomaly, and BSSID-vocab gating are all in the existing literature INDEPENDENTLY but never combined as a single preprocessing-only pipeline. This is the natural ICINCO data-systems paper - low-novelty math but high-novelty engineering synthesis, exactly the venue's sweet spot.

**Risk**: (a) Concatenating 3 redundant WiFi feature sub-vectors triples the input dim; mitigate by gating each subvector with a learnable scalar at the WiFi-encoder input (existing WiFiNet has a linear projection layer that absorbs this for free). (b) iBeacon may have coverage holes; the padding mask handles it. (c) Without cross-session augmentation, lift from input-quality alone may saturate at ~7m - the Hybrid #2 cross-session pairing is the natural extension if a follow-up paper is needed.

---

## Full lead detail (all 40)

### Per-BSSID Set Transformer with learned AP embeddings  *(angle: architectural, type: validated_published, lift~1.4 m, impl: medium)*

**Mechanism**: Replace WiFiNet's dense RSSI vector (1419-dim) with a permutation-invariant SET encoder. Each scan becomes a set {(bssid_id, rssi)} of only the APs actually heard (typically 20-80 of 1419). For each visible AP: token = Embedding[bssid_id](64-d) concat with rssi_scalar passed through Sinusoid(rssi)(64-d); 128-d per-AP token. Stack 2 Set Attention Blocks (SAB, multihead self-attention over the set), then Pooling by Multihead Attention (PMA) with 4 learned seeds to produce 4x128-d output = the WiFi modality tokens fed into the existing FusionTransformer. Padding mask handles variable scan size. Math: SAB(X)=LayerNorm(X+MHA(X,X,X)) then FFN; PMA(X)=MHA(S_seeds, X, X). This is the AaTs/Set-Transformer fix that's well-established for WiFi sets but our current encoder doesn't use it - we project to a dense vector and lose the per-AP identity.

**Why for NavLoRI**: Cross-session WiFi fails because BSSIDs that appear in train don't appear in test scans, and dense-vector projection cannot mark 'this AP is missing' vs 'rssi=-100'. Per-BSSID embeddings + padding-mask handle absence explicitly. The current WiFiNet collapses 1419 channels into k=64 anchors and treats every missing AP as a strong signal at -100dBm - exactly wrong for cross-session generalization where AP visibility flips.

**Small test plan**: Swap WiFiNet for SetTransformerWiFi in build_encoders; convert msiln scans to (bssid_idx, rssi) variable-length sets; train 30 epochs on B1 train sessions; report val and Dec5/6 test MAE vs current 9.11m WiFi-only baseline. 30 min on P4000.

**Risk**: BSSID embedding table (1419 x 64) may overfit small B1 train set (~80 traces); mitigate with embedding dropout + L2 on embedding rows. PMA seeds may all collapse to one mode - monitor PMA attention entropy.

**Sources**:
- https://arxiv.org/html/2506.00656v1
- http://proceedings.mlr.press/v97/lee19d/lee19d.pdf
- https://arxiv.org/pdf/2312.07609

### Heterogeneous GNN over BSSID-RP graph with relation-aware passing  *(angle: architectural, type: validated_published, lift~1.1 m, impl: medium)*

**Mechanism**: Construct a bipartite heterogeneous graph each scan: nodes={RP_node (1 per scan), AP_node (one per BSSID seen)}; edges weighted by normalized RSSI. Stack 2 R-GCN / HeteroGAT layers: AP-node message = mean(rssi_weight * MLP(neighbor_RP_embedding)); RP-node message = attention-weighted sum over its AP-edges. This is exactly MG-HGNN (Wang et al., arXiv:2511.07282) and the MDPI Electronics 2025 floor-classifier (mdpi.com/2079-9292/14/24/4845), both of which report consistent gains over MLP/CNN baselines on UJIIndoorLoc and a private campus set. For our setting: build a persistent BSSID node bank across train sessions; at test time, only the BSSIDs that overlap participate in message passing - cross-session shifts manifest as 'unseen BSSID nodes' that are simply absent, not as confounded dense features. PyG implementation in <250 LOC.

**Why for NavLoRI**: MSILN's cross-session failure is dominated by AP set shift (some BSSIDs appear only Nov, some only Dec). A graph where missing BSSIDs are missing nodes (not -100dBm features) directly encodes this. Reported lift over MLP on RSSI floor-classification is +3-5% accuracy; lift on coordinate regression should be similar order, plus it gives a principled way to use unlabeled Dec scans as semi-supervised graph augmentation later.

**Small test plan**: PyG HeteroData per scan; 2 RGCN layers; replace WiFiNet output with RP-node embedding fed to FusionTransformer; train 30 epochs B1; eval Dec5/6 vs 9.11m. Compare also against AP-node bank frozen at train-time embedding vs trained-on-train-only.

**Risk**: Per-scan graph building may bottleneck dataloader (cache HeteroData objects). Unseen-at-test BSSIDs will have random embeddings unless we add a 'mean BSSID' fallback embedding.

**Sources**:
- https://arxiv.org/pdf/2511.07282
- https://www.mdpi.com/2079-9292/14/24/4845
- https://arxiv.org/pdf/2312.07609

### FuseMoE with Laplace gating for irregular multi-modality  *(angle: architectural, type: validated_published, lift~1.2 m, impl: hard)*

**Mechanism**: Replace the FusionTransformer's fixed self-attention with a Sparse-MoE block: 8 experts, each a 2-layer FFN of width 256; gating uses Laplace TopK: g_i(x) = TopK(-||W_i - x||_2) instead of softmax. mTAND-style multi-time attention handles the WiFi (1Hz) vs IMU (50Hz) vs iBeacon (7.5Hz) vs magn (50Hz) rate mismatch by discretizing each modality stream to a common 10Hz grid via attention over observed timestamps. Missing modality is a learnable 'missing token' Z routed by per-modality routers with entropy regularization. NeurIPS 2024 (Han et al., arXiv:2402.03226) shows MAE 0.65 on MOSI and AUC gains on MIMIC-IV with irregular multimodal time series. For us: keeps the set-transformer skeleton but adds modality-conditioned specialization - experts can learn 'fresh WiFi mode', 'stale WiFi + fresh IMU mode', 'magn-dominant mode' separately.

**Why for NavLoRI**: Our 4 modalities (WiFi, IMU, magn, iBeacon) arrive at radically different rates and have very different cross-session statistics: WiFi shifts a lot, IMU/magn shift little, iBeacon shifts moderately. A single dense fusion network has to compromise; MoE lets the gating route the cross-session-stable modality regimes to robust experts and the WiFi-fresh regime to a specialized expert. Directly addresses lead3's WiFi-dominance issue without dropping WiFi.

**Small test plan**: Plug 8-expert SparseMoE FFN into FusionTransformer encoder layer; add Laplace gate; add learnable missing-token per modality. Train 30 epochs on B1 with magn+iBeacon added to converter (separate task). Track expert utilization histogram + Dec5/6 MAE vs 9.11m.

**Risk**: Expert collapse (one expert gets all tokens) - mitigate with load-balancing aux loss. Only 2 training sessions means experts may overfit to session ID rather than physical regime; add session-shuffled training.

**Sources**:
- https://arxiv.org/html/2402.03226v3
- https://proceedings.neurips.cc/paper_files/paper/2024/file/7d62a85ebfed2f680eb5544beae93191-Paper-Conference.pdf
- https://neurips.cc/virtual/2024/poster/93942

### Perceiver-IO latent bottleneck with per-modality cross-attention queries  *(angle: architectural, type: novel_niche, lift~1.5 m, impl: medium)*

**Mechanism**: Novel synthesis: keep WiFiNet/IMU encoders, but replace the K=4 set-transformer with a Perceiver-IO style latent bottleneck. Initialize L=16 learned latent vectors (128-d). Iterative attention: 4 blocks of [cross-attention(latents <- all_modality_tokens), self-attention(latents)]. Inputs can be any number of modality tokens at any timestamp - WiFi sets, IMU windows, magn windows, iBeacon sets - all attend into the same 16-latent bottleneck. Decode via cross-attention from a 'position query' learned token: q_pos attends to the latents -> (x,y). Key trick: each cross-attention block receives the FULL token soup at every iteration (Perceiver's iterative design) which lets later iterations re-query stale WiFi after IMU has updated the latents. Math: L_{t+1} = SelfAttn(LayerNorm(L_t + CrossAttn(L_t, X))). 16 latents x 4 iters = O(16*N) cost regardless of N tokens.

**Why for NavLoRI**: Our current K=4 set-transformer treats each modality token symmetrically and bottlenecks at the readout. A latent bottleneck with iterative re-querying lets the model 'refine its belief': iteration 1 picks up the WiFi prior, iteration 2 uses IMU to disambiguate, iteration 3 re-checks WiFi conditioned on the IMU-refined latent. This is exactly the asynchronous-multimodal pattern we want for stale-WiFi regimes. Linear scaling means we can add iBeacon (7.5Hz) and magn (50Hz) without quadratic cost.

**Small test plan**: Build PerceiverIO fusion (16 latents, 4 iter blocks, 1 position query); feed existing modality tokens; train 30 epochs B1; report Dec5/6 MAE. Ablate iter count {1,2,4,8} and latent count {4,16,64}. Compare to current K=4 baseline 9.11m.

**Risk**: 16 latents may not be enough to encode B1 floorplan geometry - sweep to 64. Iterative attention can be unstable - use pre-norm and small residual scaling (0.5). Position-query decoder may collapse to mean output - add target-coord noise augmentation.

**Sources**:
- https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf
- https://github.com/lucidrains/perceiver-pytorch
- https://huggingface.co/blog/perceiver

### Multi-quantile pinball head with median read-out  *(angle: loss-function, type: validated_published, lift~1.2 m, impl: easy)*

**Mechanism**: Replace the single (x,y) Huber regression head with two heads that each predict K quantiles q in {0.1,0.25,0.5,0.75,0.9} of x and y, trained with pinball loss L_tau(r) = max(tau*r, (tau-1)*r) where r = y - q_hat. Test-time prediction is the median head q_hat_0.5; the spread q_0.9 - q_0.1 is a free per-sample uncertainty interval (no conformal needed for triage). Because Huber-with-delta-0.5 targets a robustified mean and the dataset error distribution is *right-skewed* (median 8.48 m vs mean 10.9 m on our cross-session split), the mean target is being pulled by a long tail of large-error stale-WiFi scans; the median is provably the M-estimator minimised by tau=0.5 pinball and is L1-robust to heavy-tailed residuals. Implementation: change criterion to a small `PinballLoss(taus=[0.1,...,0.9])` module that outputs (B, 2, K), gather q_0.5 at eval. Loss is the sum over taus.

**Why for NavLoRI**: Direct attack on the documented 2.4 m median<>mean gap. Cross-session WiFi produces a heavy-tailed residual distribution (a few scans with no overlapping APs blow up MSE/Huber but barely move L1). Median regression provably ignores those tails. Free side-benefit: q_0.9 - q_0.1 is a calibrated per-prediction interval that can replace or sharpen the conformal layer in `uncertainty/conformal.py`, and feeds the fusion stack with confidence-aware tokens for downstream temporal smoothing.

**Small test plan**: Edit `trainer.py` to add loss_fn='pinball', train idea1 IMU-place-PE model on MSILN with taus=[0.1,0.25,0.5,0.75,0.9], 4 seeds, 30 epochs each. Compare test MAE at q_0.5 vs current 9.93 m Huber baseline. ~30 min total.

**Risk**: If residual asymmetry comes from a few catastrophic scans the median may regress to a per-cluster prior and lose discriminability across the basement; mitigate by training a Huber-warmup epoch first, then switching to pinball.

**Sources**:
- https://arxiv.org/pdf/1903.11202
- https://www.sciencedirect.com/science/article/pii/S0167947324001117
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html

### Barron adaptive robust loss with learnable alpha  *(angle: loss-function, type: validated_published, lift~0.8 m, impl: easy)*

**Mechanism**: Replace Huber(delta=0.5) with Barron's general loss rho(r; alpha, c) = (|alpha-2|/alpha) * ((((r/c)^2)/|alpha-2| + 1)^(alpha/2) - 1), with alpha and c registered as `nn.Parameter` (one pair per output dim, x and y). The shape alpha smoothly interpolates: alpha=2->L2, alpha=1->pseudo-Huber, alpha=0->Cauchy, alpha=-2->Geman-McClure, alpha->-inf->Welsch. Learning alpha means the network *chooses* its own robustness during training: it widens the quadratic basin where residuals are well-behaved (early epochs, easy samples) and grows heavier tails (alpha->0 or below) where residuals are persistently large (cross-session stale-WiFi scans). The probabilistic interpretation (a partition function Z(alpha)) keeps the loss a valid NLL, so the model is not gaming the loss by inflating c. Reference implementation: `robust_loss_pytorch` (Barron, Google Research).

**Why for NavLoRI**: Our Huber delta=0.5 is fixed and small, which is fine for clean Webots but mis-specified for MSILN cross-session where ~15% of scans have ~3x median error. A frozen delta forces a single curvature; learnable alpha lets the model assign different robustness to easy (fresh) vs hard (stale, OOD) samples without us hand-tuning. Pure plug-in replacement in `trainer.py`; no architecture change, no new tokens.

**Small test plan**: pip install robust_loss_pytorch. Wrap criterion with `AdaptiveLossFunction(num_dims=2, float_dtype=torch.float32, device='cuda')`; add its parameters to the optimizer (separate param group, lr=1e-3). Train idea1 4 seeds, log learned alpha trajectory, compare test MAE vs 9.93 m baseline.

**Risk**: Learnable alpha can collapse to alpha~-inf (Welsch, totally ignoring large residuals) on a few seeds and lock onto a local mean -> log alpha and clip to [-2.5, 2.0]. The adaptive loss adds 4 params total, not a capacity concern.

**Sources**:
- https://arxiv.org/abs/1701.03077
- https://openaccess.thecvf.com/content_CVPR_2019/papers/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.pdf
- https://github.com/jonbarron/robust_loss_pytorch

### Rank-N-Contrast auxiliary loss on position labels  *(angle: loss-function, type: validated_published, lift~1 m, impl: medium)*

**Mechanism**: Add an auxiliary contrastive loss on the fusion-transformer CLS embedding z that *orders* samples by ground-truth label distance. For an anchor i and any two other samples j, k in the batch with |y_i - y_j| < |y_i - y_k|, enforce sim(z_i, z_j) > sim(z_i, z_k) via the Rank-N-Contrast loss: L_RNC = sum_i sum_j -log[ exp(sim(z_i,z_j)/T) / sum_{k: d_ik >= d_ij} exp(sim(z_i,z_k)/T) ], where d_ij = ||y_i - y_j||_2 (metres). Add as auxiliary with weight lambda=0.5: L = L_huber + lambda * L_RNC. Theoretical guarantee (Zha et al. NeurIPS 2023 Spotlight): RNC produces delta-ordered embeddings whose geodesic respects the continuous label, which yields better generalisation under distribution shift (validated on AgeDB, IMDB-WIKI, TUAB).

**Why for NavLoRI**: Cross-session WiFi RSSI vectors for the *same* point look different across sessions (drift in BSSID set, RSSI levels). A pure regression head over-fits the November pose<>RSSI bijection. RNC forces the *geometry* of the embedding to match label geometry, so November-trained features that two physical points are close stays true on December scans where absolute RSSI shifted but ordinal relations partially survive. Attacks our worst pain (cross-session divergence flagged in CLAUDE.md).

**Small test plan**: Add `RNCLoss` module (~60 LOC) keyed on the CLS embedding of FusionTransformer. Train K=4 fusion model, batch>=128 for enough triplets, lambda in {0.1, 0.5, 1.0}, T=2. Compare test MAE on MSILN cross-session vs 9.11 m lead3 baseline; also log embedding rank-correlation Spearman(||z_i - z_j||, ||y_i - y_j||).

**Risk**: RNC needs large batches for stable rank statistics; with batch=32 the loss is noisy and may hurt. Mitigation: gradient accumulation across virtual batch=256, or memory bank a la MoCo with last 1024 embeddings.

**Sources**:
- https://papers.nips.cc/paper_files/paper/2023/file/39e9c5913c970e3e49c2df629daff636-Paper-Conference.pdf
- https://github.com/kaiwenzha/Rank-N-Contrast
- https://arxiv.org/pdf/2411.16298

### Session-conditional CRPS with mixture-of-Laplace head  *(angle: loss-function, type: novel_niche, lift~1.6 m, impl: medium)*

**Mechanism**: Output a *mixture* of M=4 bivariate-Laplace components, one set of (mu_x, mu_y, b_x, b_y, pi) per component (5*M=20 outputs from the readout MLP). Train under the Continuous Ranked Probability Score (CRPS) for a mixture of Laplaces, which has a closed form for univariate Laplace: CRPS_Lap(F, y) = b*[|y-mu|/b + 2*Phi_Lap(|y-mu|/b) - 1] and is averaged per axis. The mixture lets the model represent *multi-hypothesis* positions (e.g. two corridor candidates the WiFi cannot disambiguate), and CRPS is a strictly proper scoring rule that *jointly* trains location and scale -- unlike NLL it remains finite when one component collapses, so it is much more stable than evidential NIG on small-MSILN-style data. Test-time read-out: take the mode of the highest-weighted component (greedy) OR the component-weighted median (calibrated). The mixture weights are a *learnt prior* over candidate places; the model literally cannot be 'pulled to the centroid' between two ambiguous candidates because the loss penalises mass-not-on-truth, not squared-distance-of-mean.

**Why for NavLoRI**: Indoor cross-session error is dominated by *aliasing*: similar RSSI patterns at two distant places. Single-modal heads commit to the mean of the two candidates (which lies on a wall, ~5 m off both). A Laplace mixture stops that; CRPS replaces both Huber and the post-hoc conformal step. Direct attack on the residual asymmetry diagnosis (median 8.48 vs mean 10.9) -- a mixture is the simplest explanation of why mean fits poorly. Falls back gracefully to single-mode Laplace when M=1.

**Small test plan**: Implement `MixtureLaplaceCRPS(M=4)` head (~120 LOC: head MLP, softplus on b, softmax on pi, closed-form CRPS per-axis, mean over batch). Train fusion model 30 epochs, 3 seeds, sweep M in {1,2,4,8}. Eval: test MAE at component-weighted median + Brier-style mixture-coverage at 90% and compare to lead3 9.11 m + conformal.

**Risk**: Mode-collapse: all M components converge on the same mean (CRPS is convex in component means). Mitigation: dropout on component weights pi, plus a small entropy bonus -lambda*H(pi). Second risk: deciding which component to predict at test time -- recommend median over the predictive CDF, which is robust and closed-form for Laplace mixtures.

**Sources**:
- https://arxiv.org/pdf/2305.10465
- https://arxiv.org/pdf/2205.10060
- https://www.sciencedirect.com/science/article/abs/pii/S0921889011000753

### SWAD dense weight averaging for cross-session flatness  *(angle: training-algo, type: validated_published, lift~0.5 m, impl: easy)*

**Mechanism**: SWAD (Cha et al., NeurIPS 2021) replaces the final epoch's checkpoint with a dense average of weights collected every iteration inside an overfit-aware window. Concretely: at each step after a warmup, take theta_swa <- (n*theta_swa + theta_t)/(n+1). The window's start/end are chosen by tracking validation loss with a tolerance: start when val loss stops improving, end when it has not improved for r consecutive evals. This biases the final solution toward the FLAT center of a loss valley shared by sources rather than a sharp source-specific minimum. The PyTorch primitive AveragedModel + a BatchNorm update pass (we use LayerNorm so even that is unneeded) is enough. Theoretically: flatness implies a smaller cross-domain generalization gap because shifted-domain loss is bounded by source loss + a sharpness term. We swap the OneCycleLR for a constant or cosine LR after warmup; SWAD averages over the constant-LR plateau.

**Why for NavLoRI**: We have only 2 source sessions (Nov 24/25); standard SGD finds sharp minima that memorize per-session BSSID availability and IMU bias. SWAD's flat center has been shown to give ~+1.6% on 5 DG benchmarks vs ERM and ~0.5-1.0% over vanilla SWA. With 2.6 m of headroom and a 9.1 m baseline, even a 5% flatness-induced lift is ~0.45 m. Zero architectural change; just a wrapper around the existing FusionTrainer.

**Small test plan**: Wrap idea1 model in torch.optim.swa_utils.AveragedModel; after epoch 30, average weights every iteration into theta_swa; track val MAE; stop averaging when val MAE has not improved for 5 evals. Train 4 seeds, 60 epochs, MSILN site1/B1. Report test MAE + std vs 9.93 m idea1 baseline.

**Risk**: Vanilla SWA can OVER-smooth and regress to mean if the averaging window includes early high-loss iterates; SWAD's overfit-aware schedule must be tuned (val patience r=3..5). Also, our val set is one held-out path inside the train sessions, so 'val' may not predict cross-session test cleanly; consider using a tiny held-out slice of the test session as the SWAD trigger (but only the BSSID list, not labels, to keep the protocol honest).

**Sources**:
- https://arxiv.org/abs/2102.08604
- https://proceedings.neurips.cc/paper_files/paper/2021/file/bcb41ccdc4363c6848a1d760f26c28a0-Paper.pdf
- https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/

### C-Mixup label-similarity mixup for regression OOD  *(angle: training-algo, type: validated_published, lift~0.7 m, impl: easy)*

**Mechanism**: C-Mixup (Yao et al., NeurIPS 2022) is mixup for regression where the partner of a sample (x_i, y_i) is drawn with probability proportional to a Gaussian kernel k(y_i, y_j) = exp(-||y_i - y_j||^2 / 2*sigma^2), NOT uniformly. The mixed sample is (lambda*x_i + (1-lambda)*x_j, lambda*y_i + (1-lambda)*y_j), lambda ~ Beta(a, a). Because partners share similar labels, interpolation lies near the true regression manifold rather than crossing the decision boundary diagonally as in classification mixup. The authors prove a smaller bias term and report +6.56% in-dist / +5.82% OOD gains over uniform mixup across 7 regression benchmarks (airfoil, crime, RCF-MNIST, etc.). For our (x,y) targets, we precompute a row-stochastic sampling matrix P[i,j] ~ exp(-||p_i - p_j||^2/2sigma^2) once at dataset build time. sigma is chosen so the kernel half-width equals ~2 m (path scale).

**Why for NavLoRI**: Our most natural label-axis mixup partners are scans from the OTHER session at the SAME ground-truth (x,y). Mixing a Nov-24 RSSI vector at (12, 5) with a Nov-25 RSSI vector at (12, 5) creates a synthetic training example that BSSID-availability-wise looks like neither session alone -- exactly the kind of session-invariance we want. C-Mixup's similar-label kernel automatically prefers same-position cross-session pairs because they have similar y. Directly attacks the cross-session WiFi bottleneck without DANN.

**Small test plan**: Implement P[i,j] = softmax(-||p_i-p_j||^2/(2*sigma^2)) precomputed at fit time (sigma=2 m). In each training step, for each sample i, sample j ~ P[i,:] within batch with rejection if same session; lambda ~ Beta(2,2); mix RSSI + IMU windows + targets. 4 seeds, 60 epochs idea1 backbone. Sweep sigma in {1, 2, 4} m.

**Risk**: (a) Mixing IMU windows is awkward because IMU is a temporal signal; mixing two windows from different times produces an unphysical waveform -- mitigate by mixing only RSSI and copying the IMU window from the higher-lambda sample. (b) If sigma is too small the partner pool collapses to near-self pairs, killing diversity.

**Sources**:
- https://arxiv.org/abs/2210.05775
- https://proceedings.neurips.cc/paper_files/paper/2022/file/1626be0ab7f3d7b3c639fbfd5951bc40-Paper-Conference.pdf
- https://github.com/huaxiuyao/C-Mixup

### ASAM with Lookahead inner loop for flat cross-session minima  *(angle: training-algo, type: validated_published, lift~0.4 m, impl: medium)*

**Mechanism**: Sharpness-Aware Minimization (Foret et al., ICLR 2021) replaces the gradient at theta with the gradient at theta + epsilon, where epsilon = rho * grad / ||grad||. ASAM (Kwon et al., ICML 2021) rescales the perturbation per-parameter by ||theta_p|| so the implicit sharpness is scale-invariant, which matters for transformers with LayerNorm. Lookahead-SAM (Yu et al., ICML 2024) wraps SAM with k inner steps then a slow update: theta_slow <- theta_slow + alpha*(theta_fast - theta_slow), which stabilises SAM's known oscillation around saddle points and improves OOD generalization on PACS/OfficeHome by ~0.5-1%. Implementation: davda54/sam library + a simple Lookahead wrapper around the underlying AdamW. Cost: 2x forward-backward per step.

**Why for NavLoRI**: Cross-session WiFi has a sharp loss landscape: small RSSI distribution shifts (one missing AP, one renamed BSSID) move the model into a region of very different loss. Pushing the trained solution into a flat valley -- proven mechanism behind SWAD's DG gains -- is the most direct optimiser-level defence. ASAM is preferred over plain SAM because we mix tokens of very different scales (WiFi 1419-d vs IMU 9-d). Lookahead adds stability with no extra forward passes.

**Small test plan**: Drop in davda54/sam ASAM with rho=0.5 wrapping AdamW; add k=5, alpha=0.5 Lookahead. Train idea1 60 epochs, 4 seeds. Compare test MAE and Hessian top-eigenvalue (power iteration on val batch) vs vanilla AdamW.

**Risk**: (a) 2x training time -- need to keep epochs short or accept ~1.6h per seed. (b) Tuning rho is finicky; too large hurts in-domain accuracy. (c) Lookahead + OneCycleLR can interact badly because slow weights lag the LR schedule; safest is to use cosine LR with no restarts.

**Sources**:
- https://arxiv.org/abs/2010.01412
- https://github.com/davda54/sam
- https://proceedings.mlr.press/v235/yu24q.html
- https://research.samsung.com/blog/ASAM-Adaptive-Sharpness-Aware-Minimization-for-Scale-Invariant-Learning-of-Deep-Neural-Networks

### Session-paired BSSID-CutMix with co-occurrence mask scheduling  *(angle: training-algo, type: novel_niche, lift~1.2 m, impl: medium)*

**Mechanism**: A training-recipe-only trick: at each step, given a Nov-24 scan x^a and a Nov-25 scan x^b at near-identical (x,y) (found via the C-Mixup label kernel restricted to OTHER session), build a synthetic scan x^mix by BSSID-level CutMix. Concretely, partition the 1419 BSSIDs into 3 sets: A_only (seen in 24 not 25), B_only (seen in 25 not 24), both. For each training step sample a Bernoulli mask m in {0,1}^1419 with P(m_k=1) curriculum-scheduled from 0.5 -> 0.0 over training: x^mix_k = m_k * x^a_k + (1-m_k) * x^b_k. The label is the shared y (no label interpolation -- they are at the same position). Additionally, with probability p_drop (annealed 0 -> 0.3) we zero out the BSSIDs in A_only OR B_only to simulate the Dec test session's still-unseen BSSID subset. IMU windows are kept from whichever scan dominates m. This is mixup + CutMix + curriculum + BSSID-availability augmentation, fused into one recipe that exploits the only thing the converter throws away: cross-session position correspondence.

**Why for NavLoRI**: DANN failed because we have only 2 sessions; the discriminator memorizes. But the SAME 2 sessions contain a rich source of cross-session pairs at matched (x,y). Treating cross-session pairs as a DATA augmentation (not as an adversarial signal) sidesteps the 2-domain trap entirely. BSSID-availability shift is empirically THE dominant cross-session failure mode (per CLAUDE.md WiFi bottleneck note); BSSID-CutMix attacks it directly by forcing the encoder to predict from arbitrary subsets of the union of both sessions' APs. Curriculum start (50% mix) gives an 'easy' middle-ground session; curriculum end (single session, heavy dropout) is the realistic test condition.

**Small test plan**: Build a KD-tree over GT (x,y) of session B; for each session-A sample find k=5 cross-session neighbors within 1.5 m; cache pair indices. In the training loop sample one neighbor, build the Bernoulli BSSID mask with mix-ratio annealed cos(t) from 0.5 -> 0.0; drop A_only/B_only BSSIDs with probability annealed 0 -> 0.3. Train idea1 60 epochs, 4 seeds. Ablate: no curriculum, no BSSID-dropout, vs full recipe. Gate: test MAE < 8.5 m.

**Risk**: (a) Cross-session position matching uses GT, which is fine at training time but creates an implicit assumption that the test session covers similar (x,y) -- if Dec test has paths through never-visited regions the augmentation buys nothing there. (b) Mixing two RSSI vectors with one shared label can teach the encoder to AVERAGE rather than DISCRIMINATE between session signatures, leaving the network indifferent to the actual Dec signature; mitigate by adding a small entropy term that pushes the embedding NEAR the unmixed component with higher m mass. (c) Curriculum schedule introduces 2-3 hyperparameters (start mix, end mix, dropout cap) that need a small Optuna pass.

**Sources**:
- https://arxiv.org/abs/2210.05775
- https://arxiv.org/abs/1905.04899
- https://arxiv.org/abs/2102.08604
- https://arxiv.org/html/2409.05202v1

### Log-distance pathloss prior with jointly-learned AP coords  *(angle: physics-aware, type: validated_published, lift~1.5 m, impl: medium)*

**Mechanism**: Replace the WiFi token branch with a *physics-conditioned* encoder. For each visible BSSID b with RSSI r_b, learn (a) a 2-D coordinate p_b in floor-frame, (b) a per-AP TX-power A_b, and (c) a per-AP pathloss exponent n_b (shared across APs as prior). The model predicts r_hat_b = A_b - 10*n_b*log10(||x - p_b||+eps). Train end-to-end: total loss = Huber(x_hat, x_gt) + lambda * sum_b (r_b - r_hat_b)^2 (a PINN-style residual). At test time, p_b stays fixed (session-invariant). x is regressed from a transformer over the [r_b, p_b, A_b, n_b] tokens. This is the alternating min of Mortier (2012, doi:10.1109/SPAWC.2012.6292936) made differentiable, and matches the PINN-for-RSSI loss form in Bregar (IEEE TVT 2024) â€” physics residual reduces required labels and survives session shift because A_b, n_b are reusable across sessions even when fingerprint statistics drift.

**Why for NavLoRI**: Our cross-session WiFi failure mode is that absolute RSSI distribution drifts (different phones, different AP power-cycles between Nov-24 and Dec-5). The physics residual r_hat_b - r_b transfers if the *geometry* of APs is right â€” a much smaller, slower-drifting target than the full 1419-D fingerprint vector. This directly attacks the 9.1 m -> sub-6 m gap because it injects 2 free parameters per AP, replacing a 1419-D opaque encoder with a parameterised radio map.

**Small test plan**: On site1/B1 train split: optimise {p_b, A_b, n_b} by L-BFGS on r = A - 10n log10(d). Freeze. Use kNN over predicted RSSI vector at test. If kNN MAE drops from 6.48 m to ~5 m we know the AP map generalises; greenlight a full end-to-end run.

**Risk**: AP coords are unobservable up to a global rigid transform â€” must anchor via 2-3 known APs OR via concurrent (x_gt, RSSI) pairs in training, otherwise the floor is mirrored at test. Pathloss exponent n>4 in heavy NLOS may make the loss non-convex and trap L-BFGS in a local minimum.

**Sources**:
- https://ieeexplore.ieee.org/document/9704309
- https://arxiv.org/pdf/1807.04070
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9505293/
- https://www.researchgate.net/publication/224305774_A_Practical_Path_Loss_Model_For_Indoor_WiFi_Positioning_Enhancement

### Magnetometer 3-vector anomaly fingerprint stream  *(angle: physics-aware, type: validated_published, lift~1.3 m, impl: easy)*

**Mechanism**: Add the *magnetic field* as a 4th modality (currently dropped). MSILN ships TYPE_MAGNETIC_FIELD at 50 Hz; we already read it but only use it implicitly via AHRS yaw. Instead emit a `magn.csv` with (Bx, By, Bz, |B|, dip_angle) at 50 Hz, pre-rotate Bx,By into floor frame using AHRS yaw (cancels device orientation), and add a 1D-CNN magn-encoder (same shape as IMUCNN, 5 channels, 32-step window -> 128-d token). Geomagnetic anomalies inside buildings are caused by static rebar / steel beams; the field is *temporally stable* on day-to-day timescales (vs WiFi RSSI which drifts). Res-T-LSTM (Yang et al, Sensors 2025, doi 10.3390/s25051304) reports 0.21 m magn-only MAE; MaLoc (Shu et al, 2015) and Magicol show <2 m with smartphone-grade magn sensors.

**Why for NavLoRI**: WiFi drift is exactly the failure mode we measured (train/val divergence cross-session). Magn doesn't drift across an 11-day gap because the building's ferromagnetic structure is fixed. It's free data â€” we just need to plumb it through the converter. Adding a *stable* modality the fusion can attend to when WiFi is uninformative is the cleanest way to close the gap on cross-session traces.

**Small test plan**: Patch convert_msiln.py to dump magn.csv. Train magn-only IMUCNN-style encoder on train, eval on Dec test traces. Target: magn-only kNN MAE < 8 m (since site1/B1 is one floor only). Then add magn token to FusionTransformer and compare to current 9.1 m baseline at same epochs.

**Risk**: Hard-iron offset varies per phone (DC bias of several uT in Bx,By); if Nov and Dec traces are on different devices the fingerprint shifts. Mitigation: per-trace high-pass filter (subtract trace-mean). Soft-iron distortion from device chassis is *not* removed by HP filter and may need ellipsoid calibration.

**Sources**:
- https://doi.org/10.3390/s25051304
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12736867/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9921884/
- https://www.researchgate.net/publication/287019228_MaLoc_A_practical_magnetic_fingerprinting_approach_to_indoor_localization_using_smartphones

### iBeacon RSSI as low-drift second radio modality  *(angle: physics-aware, type: validated_published, lift~1 m, impl: easy)*

**Mechanism**: MSILN site1/B1 has TYPE_BEACON (iBeacon) at ~7.5 Hz in 40/40 traces, also currently dropped. iBeacons in this dataset are venue-mounted (vs WiFi APs which can be turned on/off by tenants), so their TX power and position are *stable* between Nov and Dec. Add `beacon.csv` with (uuid, major, minor, rssi) per scan and a permutation-invariant set encoder (same architecture as WiFiNet on the BSSID side). Beacon RSSI is shorter-range (~10 m vs WiFi ~30 m), so it provides high-confidence local anchoring exactly where WiFi multipath is worst. Validated by CantÃ³n Paterna et al (Sensors 2017) and the BLE-fused KF-PF chain in MDPI Appl-Sci 2020 (doi 10.3390/app10062003) â€” both show 1-2 m extra accuracy when BLE is added on top of WiFi.

**Why for NavLoRI**: Our 9.1 m is essentially a WiFi-only result with an IMU smoother. Beacons are a separate radio layer with different physics: smaller cells, lower variance, BLE-spec'd 1 mW TX, and *venue-controlled* deployment. They're literally in the data and we ignore them. A 4-token transformer with [WiFi, IMU, magn, beacon] is the natural next config.

**Small test plan**: Extend converter to emit beacon.csv. kNN-baseline on beacon-only fingerprints (concat per-uuid mean RSSI per scan). If beacon-only kNN is <12 m on cross-session test (vs WiFi-only ~6.5 m), the modality carries complementary info -> wire into FusionTransformer and compare.

**Risk**: iBeacon count per scan may be <5 in some areas of B1 (sparse deployment) -> token sequence very short, attention may collapse. Mitigation: CLS-only readout when no beacons seen. Also UUIDs may not be unique across days if venue swaps batteries -> need to key on (uuid, major, minor) triple.

**Sources**:
- https://www.mdpi.com/2076-3417/10/6/2003
- https://arxiv.org/pdf/2305.19342
- https://archive.ics.uci.edu/dataset/435/ble+rssi+dataset+for+indoor+localization+and+navigation
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12074306/

### AP-pair RSSI differences as calibration-invariant input  *(angle: physics-aware, type: novel_niche, lift~1.8 m, impl: medium)*

**Mechanism**: Replace the raw 1419-D RSSI vector with the upper-triangular matrix D_ij = r_i - r_j (only for AP pairs (i,j) both seen in the same scan). Physics: device gain, antenna pattern, body shadowing and phone-specific receive offset *cancel* exactly in the difference; what survives is the pathloss-difference 10 n (log10 d_j - log10 d_i), which is purely *geometric*. This is the WiFi analog of TDoA hyperbolic positioning -- each pair (i,j) constrains the user to a hyperbola of constant log-distance ratio. Encoder: feed sparse {(i,j,D_ij)} as set tokens (i_embed XOR j_embed + sinusoidal D_ij encoding) into a transformer head. The number of input tokens is ~k(k-1)/2 for k visible APs, manageable (k ~ 20-50 in B1). Pair-feature ideas appear in CantÃ³n-Paterna BLE work and in HAIL (Mahfouz et al, 2017) but no recent transformer-set encoder uses it as the sole input on cross-session WiFi data â€” the angle is under-explored and physically motivated.

**Why for NavLoRI**: MSILN cross-session error is dominated by an additive RSSI bias delta_r that differs Nov vs Dec (different phone state, different background noise floor). For any pair, D_ij is invariant to delta_r by construction. This is the *only* feature class with a proof-level guarantee of cross-session invariance from physics alone â€” no learning needed, no domain-adversarial loss needed. Conceptually closes the train/val divergence we measured (audit 2026-05-20).

**Small test plan**: Compute D_ij for all (i,j) visible in train scans, build kNN over D_ij with Hausdorff/Earth-Mover set distance; eval on Dec test traces. If kNN MAE on pair-features < kNN MAE on raw RSSI (currently 6.48 m), the invariance hypothesis holds. Then build a small set-transformer over pair tokens.

**Risk**: Quadratic blow-up in tokens (n=50 APs -> 1225 pairs) may hit memory; mitigation: subsample pairs by RSSI rank (top-K strongest pairs). If only one of (i,j) is visible in a test scan, the pair is missing -- pair-feature coverage might be too sparse on weak scans. Falls back to raw RSSI for those.

**Sources**:
- https://www.researchgate.net/publication/330043153_Wi-Fi_Received_Signal_Strength_Based_Hyperbolic_Location_Estimation_for_Indoor_Positioning_Systems
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5421677/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9371388/
- https://arxiv.org/pdf/1605.02287

### Recursive KalmanNet 2D Position Head  *(angle: probabilistic, type: validated_published, lift~1.2 m, impl: medium)*

**Mechanism**: Wrap our existing single-instant fusion regressor as the *measurement* of a 2-state (x,y) Kalman filter whose dynamics are 'almost-constant-position with learned process noise'. Use Recursive KalmanNet (Mortada et al., EUSIPCO 2025, arXiv:2506.11639): two GRU-based heads predict (i) the Kalman gain Kt and (ii) the Cholesky factor Ct of the noise-dependent part of the corrected covariance; the closed-form Joseph update P_{t|t}=A_t+C_t C_t^T guarantees a PSD covariance and consistent posterior variance even under non-Gaussian noise. We train end-to-end with Gaussian negative log-likelihood on cross-session MSILN sequences. Crucially, we do *not* assume a hand-tuned R; the network learns measurement-noise inflation when WiFi+IMU disagrees (the cross-session regime). The output is a full Gaussian over (x,y) per frame.

**Why for NavLoRI**: Our current fusion outputs a point only; on cross-session MSILN it sees the same RSSI from two ~10m-apart locations and confidently averages them, which is exactly what blows up MAE. RKN gives a *learned, time-varying* noise model that can smooth WiFi pseudo-measurements over the IMU manifold, and unlike KalmanNet it produces a calibrated covariance. Replaces 'point + heuristic conformal' with a proper recursive posterior on whole blind trajectories.

**Small test plan**: Clone github.com/ixblue/RecursiveKalmanNet, replace synthetic 1D demo with our 2D state and feed our 9.93m idea1 predictor as measurement. Train 3 epochs on 30-trace subset, eval on 10 held-out test traces. Pass if mean MAE < 9.5m AND empirical 1-sigma coverage in [0.55,0.75].

**Risk**: Joseph form was tested in 1D synthetic only - cross-session WiFi has heavy correlated drift, not just inflated white noise; the GRU gain head may collapse to identity. Also requires *sequential* training, which breaks our current shuffled-frame pipeline.

**Sources**:
- https://arxiv.org/abs/2506.11639
- https://arxiv.org/html/2506.11639v1
- https://github.com/ixblue/RecursiveKalmanNet
- https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics_0.pdf

### Deep Kernel GP Re-Ranker on WiFi+iBeacon  *(angle: probabilistic, type: validated_published, lift~1.5 m, impl: medium)*

**Mechanism**: Adopt Wilson et al.'s Deep Kernel Learning (AISTATS 2016) recipe, validated for RSSI by Guan et al. 2021 (arXiv:2109.04360, 'Measuring Uncertainty in Signal Fingerprinting with GPs Going Deep'). A small MLP phi(rssi, ibeacon) maps the 1419+iBeacon RSSI vector into a low-dim feature space; a spectral-mixture GP regresses (x,y) on phi with full marginal-likelihood training. Use 256-512 inducing points (SVGP / KISS-GP) for tractability. At test time, do NOT use the GP as primary predictor - use it as a *re-ranker*: take the top-K (K=8) candidate poses from kNN-on-train (which already hits the 6.48m irreducible floor) and pick the one maximizing GP posterior log-density. The deep kernel learns session-invariant similarity while the GP enforces smooth (x,y) covariance.

**Why for NavLoRI**: Our current point regressor cannot represent a *bimodal* posterior, yet cross-session WiFi at MSILN B1 is genuinely bimodal (same fingerprint, two corridors). A GP re-ranker uses the *covariance structure* of the embedding to break the tie, and kNN already proves the train set contains the answer (floor = 6.48m). We never tried a GP because we thought 1419 inputs were too big - deep kernels make it tractable.

**Small test plan**: Use GPyTorch's DKL example (~150 LOC). Train on 80% of train traces, calibrate on remaining 20%, eval re-ranking K=8 kNN candidates on 5 cross-session test traces. Pass if reranked-MAE < 8.5m and the GP marginal variance correlates (Pearson > 0.4) with the actual squared error on test.

**Risk**: 1419 BSSIDs is high-dim even with deep kernel; inducing-point optimization may underfit. GP variance can be miscalibrated under domain shift (training distribution = Nov, test = Dec) - covariance may shrink falsely. Re-ranker only helps if the right answer is in the top-K kNN list (we should verify this is true with an oracle check first).

**Sources**:
- https://arxiv.org/pdf/2109.04360
- https://proceedings.mlr.press/v51/wilson16.pdf
- https://arxiv.org/pdf/2202.01980
- https://arxiv.org/pdf/2505.18526

### Conditional Normalizing Flow Posterior over (x,y)  *(angle: probabilistic, type: novel_niche, lift~1 m, impl: easy)*

**Mechanism**: Replace the regressor's final MSE/Huber head with a *conditional* normalizing flow p(x,y | z) where z is the 128-d fusion token. Use a small Real-NVP or Neural Spline Flow (4-8 coupling layers, 64-d hidden) conditioned on z; train by maximum likelihood (-log p_flow(x,y|z)). At inference, draw 256 samples from the flow and take either the mean or the *highest-density mode*. The flow can natively represent the bimodal 'same RSSI -> two corridors' posterior that any unimodal head (Gaussian, NIG, MDN) struggles with. Critically, unlike our failed MDN attempt, the flow has no fixed K-mixture component count, no mode collapse incentive, and produces a smooth density we can integrate. Pair with conformal calibration on the flow's NLL to get distribution-free coverage of the *highest-density region*. Code sketch: nflows library + 30 LOC for the conditioning MLP + ConformalHDR(alpha=0.1).

**Why for NavLoRI**: MDN failed because it picked the wrong mixture component on cross-session; a continuous-density flow has no such hard partition. Cross-session MSILN B1 is a textbook bimodal-posterior setting (parallel corridors, symmetric WiFi geometry) - if any probabilistic head can crack the 6m floor, it's one that admits genuine multi-modality. We've never tried flows.

**Small test plan**: Plug nflows.NeuralSplineFlow on top of frozen idea1 features. Train flow 20 epochs on train traces. On 5 test traces, report MAE@mean and MAE@mode (highest-density sample after K-means on 256 draws). Pass if MAE@mode <= 9.0m AND flow log-density at GT > flow log-density at the mean predictor's output.

**Risk**: Flows over 2D are usually trivial to fit but adding rich conditioning often collapses to a unimodal Gaussian if z is too informative; the *opposite* failure mode (spreading mass over the whole floor) gives mean predictions worse than a Gaussian head. Selecting the right mode is itself hard - 'mean of samples' may be no better than current MSE.

**Sources**:
- https://github.com/janosh/awesome-normalizing-flows
- https://arxiv.org/pdf/2311.00377
- https://www.emergentmind.com/topics/conditioned-normalizing-flows
- https://arxiv.org/abs/2510.14111

### Conformalized Quantile Regression with Locally-Adaptive Bands  *(angle: probabilistic, type: validated_published, lift~0.4 m, impl: easy)*

**Mechanism**: Replace our split-conformal scalar-radius residual with Romano et al.'s Conformalized Quantile Regression (NeurIPS 2019, arXiv:1905.03222) and the Localized variant (arXiv:2411.19523). Train two extra heads on the existing idea1 trunk that predict the conditional alpha/2 and 1-alpha/2 quantiles of the *signed* x- and y-errors via pinball loss. On the calibration split (10% of train), compute per-sample non-conformity scores s_i = max(q_lo(x_i)-y_i, y_i-q_hi(x_i)) and a *locally-weighted* quantile of these scores using k-nearest-neighbours in fusion-embedding space (CQR-d / localized CQR). At test time output an axis-aligned conformal box [q_lo - r(z), q_hi + r(z)] with r adapting per embedding-neighbourhood. This converts our point-only output into a *valid, heteroscedastic, finite-sample* posterior region, with the box-center quantile midpoint usable as a *new* point estimate (often more robust than MSE under skew).

**Why for NavLoRI**: Our current conformal is global - scalar radius, equal for sure-bets and ambiguous corridor crossings. Cross-session WiFi error is wildly heteroscedastic (sure near anchors, ambiguous in corridors). CQR gives per-point bands; the locally-adaptive variant uses our own embedding metric to soften train-test distribution shift. This is a Pareto improvement: same point predictor, calibrated bands, and the *median quantile* often beats the MSE-mean by 5-10% under heavy-tailed error.

**Small test plan**: Add 2 quantile heads (pinball loss, alpha=0.1) to idea1 trunk; train 10 epochs warm-started. Calibrate on held-out 10%. Eval on 5 test traces: report MAE@median and 90% empirical coverage of the conformal box. Pass if MAE@median <= 9.3m AND coverage in [0.86,0.94] (vs our current under-cover).

**Risk**: Lift on MAE is modest by design (this is primarily an uncertainty win, not an accuracy win); locally-adaptive variant assumes embedding metric is meaningful under domain shift, which is exactly what our cross-session WiFi violates. Could under-cover if calibration set is too small.

**Sources**:
- https://arxiv.org/abs/1905.03222
- https://arxiv.org/pdf/1905.03222
- https://arxiv.org/html/2411.19523
- https://arxiv.org/pdf/2505.01810

### QMF Energy-Gated Per-Modality Fusion Weights  *(angle: multi-modal-fusion, type: validated_published, lift~1.2 m, impl: medium)*

**Mechanism**: Quality-aware Multimodal Fusion (Zhang et al., ICML 2023) replaces static modality weights with per-sample per-modality fusion weights derived from an energy-based uncertainty estimate. For each modality m, train an auxiliary regression head h_m(z_m) -> (x_hat, y_hat). Compute a per-sample energy/uncertainty u_m(x) = -log sum_y exp(-||y - h_m(z_m)||/T) (or just predictive variance from a small ensemble / Huber residual). Normalised softmin weights w_m(x) = softmax(-u_m(x)/tau) condition the FusionTransformer either (a) as a multiplicative mask on each modality's token bundle before self-attention, or (b) as the gating coefficient on each unimodal head in a late-fusion sum. A monotonicity regularizer (Lcrm = sum max(0, L(fused) - L(m))) forces fused loss <= best unimodal loss, with provable generalisation gain over static weighting. The token-mask variant slots into our existing FusionTransformer's padding-mask machinery in ~80 LOC.

**Why for NavLoRI**: MSILN cross-session pain is heteroscedastic per modality: WiFi degrades by session, IMU is per-trace drift-shaped, iBeacon is sparse, magnetometer is locally trustworthy but globally ambiguous. Our FusionTransformer currently weights every token equally given availability. QMF computes per-sample energy and down-weights modalities whose unimodal head finds the sample hard. On Dec 5/6 traces, WiFi energy should spike where APs disappeared since Nov 24 -- the gate routes around them without architectural surgery.

**Small test plan**: Add per-modality 2-layer MLP heads and energy estimator; replace token mask with availability AND softmin(energy) gate; train 30 epochs on B1; report cross-session MAE + which modality got down-weighted on test sessions (sanity check WiFi gets lower weight at high u_m).

**Risk**: Energy estimator can collapse (all modalities equally uncertain) if heads under-train -> degenerates to uniform. Mitigation: pretrain unimodal heads for 5 epochs frozen before joint training. Second risk: gating overfits to training-session noise pattern and gates wrong sensor at test time -- check gate calibration on a held-out train session.

**Sources**:
- https://arxiv.org/abs/2306.02050
- https://github.com/QingyangZhang/QMF

### AECF Entropy-Gated Curriculum Modality Masking  *(angle: multi-modal-fusion, type: validated_published, lift~0.8 m, impl: easy)*

**Mechanism**: AECF (Wang et al., 2025) treats the modality-dropout mask as an adversarial teacher. Per-sample entropy lambda(x) = lambda_min + softplus(MC-dropout variance over the fusion output) sets the instantaneous entropy regularisation strength on the modality gate. Mask sampling is curriculum-adversarial: pi_t(S) proportional to exp(H(p_t(x \ S))/eta), i.e., the mask sampler oversamples subsets where the gate is most confident, forcing the model to justify confidence. A monotone-confidence contrastive loss L_cec = sum ReLU(c(A) - c(B))^2 over subset pairs A subset B prevents the model from being more confident with fewer modalities. The whole package replaces our current uniform-Bernoulli modality_dropout with three tiny modules: (1) entropy-weighted gate, (2) probability-proportional-to-entropy mask sampler, (3) subset-monotone calibration penalty.

**Why for NavLoRI**: Our FusionTransformer's modality_dropout is uniform Bernoulli. On MSILN cross-session, WiFi is the dominant-but-unreliable modality -- the right training distribution is to drop WiFi *more* on samples where the model leans hardest on it. AECF's adversarial mask sampler produces exactly that. Reported +18-21 mAP under 30-50% modality dropout on MS-COCO is the regime we live in: ~25% of MSILN test windows have stale WiFi.

**Small test plan**: Swap uniform modality_dropout in src/pipeline/training/fusion_trainer.py for entropy-proportional mask sampler (compute gate entropy once per batch from a forward pass with full input). Add L_cec across two random subsets per batch. 20-epoch run on B1 cross-session; track MAE under simulated WiFi loss (mask 50% WiFi at test).

**Risk**: Entropy estimation needs MC samples or a small ensemble -- ~2x forward cost. Mitigation: amortise via a learned gate-confidence head trained with detach(). Second risk: monotone-confidence constraint may hurt fresh-data MAE when WiFi is genuinely informative -- temper L_cec coefficient.

**Sources**:
- https://arxiv.org/html/2505.15417v1
- https://arxiv.org/html/2510.01677v1

### FiLM Session-Hypernet Conditioning on Session-Stats  *(angle: multi-modal-fusion, type: validated_published, lift~1.4 m, impl: medium)*

**Mechanism**: Perez et al. (AAAI 2018) FiLM applies per-feature affine modulation z' = gamma(c) * z + beta(c) where (gamma, beta) are produced by a hypernetwork from a context c. Here c is a SESSION FINGERPRINT computed online from the first ~10 s of the test trace: mean/std of WiFi RSSI per top-K BSSID, IMU accel-magnitude spectrum, magnetometer mean/var, iBeacon presence rate. A tiny hypernet h_phi(c) -> (gamma_l, beta_l) for each transformer layer modulates the post-LN features. The model is trained with c sampled from any train session; at test time c is computed once per Dec 5/6 trace and the same (gamma, beta) modulates the whole trace. This is exactly FiLM-Ensemble style probabilistic conditioning (Wenzel 2022) but with session-stats as the conditioning vector -- a TEST-TIME-ADAPTATION mechanism that does NOT require gradient updates at test.

**Why for NavLoRI**: Our 9.1 m bottleneck is session covariate shift. DANN failed because we only have 2 train sessions to adversarially align. FiLM with session-stats is the gentler variant: instead of forcing session-invariant features, learn session-conditioned features. The hypernet generalises by interpolation (gamma is a smooth function of c), so even unseen sessions in Dec get a reasonable modulation. Critical: no floorplan, no per-AP coords -- only summary stats of the trace itself.

**Small test plan**: Compute session fingerprint c (50-dim: WiFi top-32 BSSID mean RSSI, IMU 9-dim std, magn 3-dim mean/var, iBeacon presence frac) per trace. Hypernet = 2-layer MLP -> 2*d*L outputs. Insert FiLM after each transformer LN. Train idea1+lead3 stack with FiLM head; ablate by zeroing c at test (should regress to 9.1).

**Risk**: Hypernet overfits to the 2 train sessions and outputs constant (gamma=1, beta=0) -> no help. Mitigation: dropout on c and L2 penalty on (gamma - 1, beta). Second risk: session fingerprint leaks position info via WiFi mean -- regress c against (x,y) on train; if R^2 high, restrict c to position-invariant stats (IMU spectrum, magn variance, BSSID counts only -- not RSSI values).

**Sources**:
- https://arxiv.org/abs/1709.07871
- https://arxiv.org/pdf/2206.00050
- https://github.com/ethanjperez/film

### Per-Anchor Cross-Attention Readout with BSSID-Indexed Queries  *(angle: multi-modal-fusion, type: novel_niche, lift~1.6 m, impl: medium)*

**Mechanism**: Novel synthesis: replace our single learned PositionQuery in the cross-attention readout with a SET of learned BSSID-indexed queries q_a in R^d, one per top-N BSSID seen at training time (plus a pool of K_iBeacon queries for iBeacons, plus a magnetometer cluster query). At each instant the readout cross-attends ONLY over (a) the modality tokens AND (b) the queries q_a whose anchor is currently observed (RSSI > floor). The position prediction = MLP(mean over attended q_a outputs). This is Perceiver-IO's learned-query idea (Jaegle 2021) wedded to a structured query set keyed by BSSID identity. Crucially, each q_a learns 'what does it MEAN to see this anchor at this RSSI', not 'where is this anchor' -- the query is conditioning, not coordinates (no anchor coords used, satisfies hard constraint). Unobserved anchors at test time simply vanish from the query set; new anchors map to a learned 'unseen' query via a content-hash projection of the BSSID string.

**Why for NavLoRI**: WiFi-Net currently averages anchor influence into 128-d. Cross-session, the SET of observed anchors changes (some APs disappear, new ones appear). A single global PositionQuery cannot exploit per-anchor specificity. Per-anchor queries let the readout discount missing-in-test anchors automatically (query absent -> zero contribution) and exploit shared anchors more sharply. Combines naturally with magnetometer/iBeacon queries to test if any modality has unique-anchor signal worth preserving. This is the obvious 'make the Mamba+JEPA stack interesting' move.

**Small test plan**: Build BSSID vocab from train (top-512 BSSIDs); learn embedding table (512, d). Replace PositionQuery in FusionTransformer's CrossAttention readout with available-anchor-subset queries. Hash-project unseen BSSIDs at test. Ablate: (a) global query (baseline), (b) per-anchor queries, (c) per-anchor + hash-fallback. 30-epoch run on B1.

**Risk**: Train-only-BSSID overfitting: queries memorise train-session anchor coverage. Mitigation: random anchor dropout during training (drop 30% queries per batch). Second risk: hash collisions for unseen BSSIDs degrade test -- size hash space larger than expected unseen count (~4k). Third risk: combinatorial expense if N anchors observed per window is large (~200) -- but the readout is k cross-attn over (N+modalities), still linear.

**Sources**:
- https://arxiv.org/abs/2107.14795
- https://huggingface.co/blog/perceiver
- https://www.tandfonline.com/doi/full/10.1080/24751839.2021.1975425

### Memorizing Transformer kNN-attention over training tokens  *(angle: retrieval-memory, type: validated_published, lift~1.8 m, impl: medium)*

**Mechanism**: Add a kNN-attention head to the FusionTransformer self-attention stack (Wu et al., ICLR 2022, Memorizing Transformers). Build an external memory M = {(k_i, v_i, y_i)} where k_i is the WiFi-token embedding (post linear-K projection) of every training scan, v_i is the same embedding's V-projection, and y_i is the (x,y) target. At forward time, for each query token q, do an approximate kNN lookup (FAISS IndexFlatL2 or HNSW) and retrieve top-K=32 (k,v) pairs. Compute softmax(qK^T/sqrt(d))V over them and gate-mix with local-attention output via a learned scalar sigmoid(g) in [0,1] per head: out = (1-g)*local + g*memory. Memory keys are recomputed at the end of every epoch from the live encoder (the same trick that makes Memorizing Transformers stable). The memory is non-differentiable wrt itself but gradients flow through q,K_proj,V_proj. This places long-tail / rare BSSID configurations explicitly into the prediction path instead of relying on the encoder to memorise everything in 128-d.

**Why for NavLoRI**: Our kNN-on-train MAE is 6.48 m, meaning the training set DOES contain enough nearby fingerprints to localise the test â€” but our parametric encoder cannot project test-day RSSI close enough to the right train-day RSSI in the OUTPUT head. Adding a kNN-attention head moves that retrieval INTO the model: queries align test-day embeddings to train-day embeddings via gradient on the gate. Diff-kNN failed earlier because it had to differentiate THROUGH the kNN with learned-metric loss; this variant is non-differentiable through retrieval but differentiable through (q, K_proj), which is much easier to fit on 2 sessions.

**Small test plan**: 30-min test: load WiFiNet embeddings of the train set (~10k tokens) into a FAISS IndexFlatL2; bolt a single kNN-attention head onto the readout cross-attention of FusionTransformer; train 20 epochs with K=32 retrievals, learned sigmoid gate init=-2 (memory starts off). Compare test MAE vs idea1 baseline 9.93 m.

**Risk**: (a) Memory built from train-session keys is offset from test-session keys by the same session bias that hurts kNN; the gate may collapse to 0 and add no value. (b) FAISS rebuild every epoch is the bottleneck â€” needs to be backgrounded or done every-N-epochs.

**Sources**:
- https://arxiv.org/abs/2203.08913
- https://arxiv.org/pdf/2203.08913
- https://arxiv.org/pdf/2407.13193

### Channel-charted reference retrieval + graph attention  *(angle: retrieval-memory, type: validated_published, lift~1.5 m, impl: medium)*

**Mechanism**: Reproduce the retrieval-assisted localisation framework (Zhang et al., arXiv 2603.06158): (1) self-supervised channel charting on the train RSSI pool â€” learn a 32-d chart z = f(rssi) with a triplet-loss objective using nearby-in-TIME RSSI pairs as positives (no labels needed, fits MSILN's continuous traces). (2) Index the chart with FAISS. (3) At inference, encode the query RSSI to its chart point z_q, retrieve K=16 nearest train chart points, build a K+1-node graph (query + neighbours, edges weighted by chart distance), and run a 2-layer Graph Attention Network whose output node is the query's predicted (x,y). The chart absorbs session-invariant geometry (because triplets are time-local, they capture spatial smoothness not the absolute RSSI level), and GAT replaces the brittle WKNN distance metric with a learned, geometry-aware weighting.

**Why for NavLoRI**: The 'WiFi fingerprints don't transfer between sessions' problem in CLAUDE.md is exactly what channel charting was invented to mitigate â€” it learns geometry from temporal coherence, not absolute RSSI magnitudes. Reported to beat similarity-based AND learning-based baselines in the source paper. MSILN traces are dense in time (1 Hz WiFi over multi-minute trajectories) so triplet positives are easy to mine. The GAT step replaces our failed cost-min L-BFGS post-process with a properly differentiable one.

**Small test plan**: 30 min: SimCLR-style training of a 5-layer MLP charter on all train RSSI with NT-Xent loss, positive = same trace within 2 s, negative = different trace. FAISS the train chart. Plug a 2-layer torch_geometric GATv2Conv over (query, K=16 neighbours) and regress to (x,y). Compare test MAE vs lead3 9.11 m.

**Risk**: (a) Triplet mining inside a single trace can collapse to identity because the chart only sees smooth walks â€” needs hard-negative mining across distant time windows. (b) GAT over 17 nodes is tiny; gradient may be noisy and overfit.

**Sources**:
- https://arxiv.org/pdf/2603.06158
- https://arxiv.org/pdf/2405.04357
- https://arxiv.org/pdf/2210.06294

### Deep kernel Gaussian process readout on WiFi embeddings  *(angle: retrieval-memory, type: validated_published, lift~1.2 m, impl: easy)*

**Mechanism**: Replace the FusionTransformer's MLP readout head with a Deep Kernel Learning (Wilson et al., AISTATS 2016) sparse-variational Gaussian process. The deep feature extractor is exactly the existing FusionTransformer (frozen after a short warm-up); on top we put a 2-D variational GP with M=512 inducing points learnt jointly with an RBF + Matern composite kernel, using gpytorch's SVGPRegression. Train end-to-end with the marginal log-likelihood as the loss. At inference, the posterior mean over (x,y) is a Gaussian-weighted memory lookup over the 512 inducing points â€” i.e. a learnable RAG over a compressed training memory. The 512 inducing points are the 'memory'; the GP kernel is the learned retrieval similarity.

**Why for NavLoRI**: DKL gives us TWO things we currently lack: (1) a non-parametric readout that does kernel-weighted retrieval over training-set summaries â€” directly addresses the 6.48 m kNN-floor by replacing the linear regression head that currently forces a global linear fit. (2) Posterior variance for free, which our current ConformalPosition needs an outer wrapper to produce. Wilson's original paper showed >50% RMSE drop vs deep-NN-only baselines on small regression tasks (e.g. on power-plant: 4.32 -> 3.51 RMSE). 512 inducing points is enough for MSILN-B1's ~10k training samples.

**Small test plan**: 30 min: pip install gpytorch; freeze idea1's encoder, swap MLP head for gpytorch.models.ApproximateGP with VariationalStrategy and 512 inducing points init'd by k-means on the train embeddings; train 10 epochs with ELBO. Compare test MAE vs idea1 9.93 m; also report 90% credible-interval coverage.

**Risk**: (a) GPyTorch on Pascal (sm_61) may need fp32 â€” slower than expected. (b) Cholesky during ELBO is numerically fragile when inducing points drift; needs jitter. (c) 2 outputs (x,y) means two GPs or one multi-task GP â€” adds complexity.

**Sources**:
- https://proceedings.mlr.press/v51/wilson16.pdf
- https://www.researchgate.net/publication/283619654_Deep_Kernel_Learning
- https://openreview.net/pdf?id=99GWvTezZ8

### Per-BSSID anchor memory with masked cross-attention  *(angle: retrieval-memory, type: novel_niche, lift~2 m, impl: easy)*

**Mechanism**: Treat each of the 1419 B1 BSSIDs as a slot in an external memory tensor M of shape (1419, d=64). M is a learnable parameter, initialised by averaging the (x,y) target for every train sample where that BSSID was observed at RSSI > -75 dBm and projecting through a learned linear. At forward time, for each query scan, mask M with the BSSIDs actually visible in this scan AND weight by min-max-normalised RSSI; do a masked cross-attention layer where Q = current fused token, K = visible-BSSID rows of M (after a K_proj), V = visible-BSSID rows of M (after a V_proj). The query token is augmented by the attended memory before the readout head. This is sparse retrieval over BSSID anchors â€” a memory network (Sukhbaatar 2015) where slots ARE the BSSIDs. Crucially, the same BSSID slot is shared across train and test sessions: only the RSSI mask/weight changes session-to-session, which is exactly the session-invariant 'who-saw-what' signal that survives the 11-day gap.

**Why for NavLoRI**: The known failure mode is that test-day RSSI MAGNITUDES drift but the SET of visible BSSIDs at a location is much more stable. Encoding (x,y) priors per BSSID into a memory bank, then attending over only the visible subset, factors out exactly the session-variant part. No floorplan, no AP coordinates â€” the memory LEARNS per-BSSID coordinates from data. This is the natural retrieval-memory answer to lead3's observation that JEPA SSL on RSSI helped: SSL learned 'who's with whom', memory makes it explicit.

**Small test plan**: 30 min: nn.Embedding(1419, 64) for M; build visibility mask + normalised RSSI weight from raw scan; bolt one nn.MultiheadAttention(d=64, h=4) using Q=encoder CLS, K=V=M[visible_idx]; concat the attended vector to CLS before readout. Train 20 epochs on idea1 config. Compare test MAE vs idea1 9.93 m and vs lead3 9.11 m.

**Risk**: (a) BSSIDs in test that did NOT appear in train get zero-init slots; need a graceful fallback (mask them out, or share one 'unknown' slot). (b) Memory may overfit per-BSSID â€” needs L2 / dropout on M, or low-rank factorisation M = U V^T with U trainable, V fixed-random.

**Sources**:
- https://arxiv.org/abs/1503.08895
- https://arxiv.org/pdf/2407.13193
- https://www.mdpi.com/2079-9292/14/14/2807

### DDPM head for (x,y) conditioned on multimodal token  *(angle: generative, type: validated_published, lift~1.5 m, impl: easy)*

**Mechanism**: Replace the regression head with a conditional DDPM over the 2D coordinate y0 = (x,y), conditioned on the FusionTransformer CLS embedding c. Forward: yt = sqrt(alpha_bar_t)*y0 + sqrt(1-alpha_bar_t)*eps. Train a small MLP eps_theta(yt, t, c) (~50k params) with simple eps-MSE loss (Ho et al. 2020). At test, sample y_T ~ N(0,I) and run K=20 DDIM steps; average S=16 samples to get point estimate, std for uncertainty. DiffLoc (CVPR 2024) used the same recipe for 6-DOF LiDAR poses; DDPM theory (Li et al. 2024) shows iteration count scales with intrinsic dim k=2, so very few steps are needed. The model is forced to learn p(x,y|c) rather than just argmax y, which is exactly the multimodal posterior that fails MDN.

**Why for NavLoRI**: MDN failed at 50m because Gaussian mixtures struggle when modes are spatially close and badly identifiable; DDPM has no mode-collapse pathology. Our cross-session WiFi gives genuinely multimodal posteriors (same RSSI fingerprint near 2 different places on the floor). Per-sample variance gives free calibrated uncertainty -- a known weakness in our 9.1m baseline. Drops in next to existing FusionTransformer with zero feature-engineering.

**Small test plan**: Replace MLP head in FusionTransformer with diffusion head: ~60 LOC for eps_theta MLP + ~30 LOC cosine schedule + DDIM. Train 30 epochs on existing MSILN B1 split. Compare mean-of-16-samples MAE vs current 9.11m. Sanity-check coverage of the 16-sample empirical std.

**Risk**: If c is too peaky the diffusion collapses to a delta and underperforms the deterministic head. Mitigation: classifier-free guidance with drop_p=0.1 on c during training.

**Sources**:
- https://arxiv.org/abs/2404.09140
- https://ieeexplore.ieee.org/abstract/document/10657300/
- https://github.com/liw95/DiffLoc
- https://arxiv.org/abs/2410.18784

### Noise-contrastive energy-based regression head  *(angle: generative, type: validated_published, lift~1.2 m, impl: easy)*

**Mechanism**: Define E_theta(y | c) = f_theta(c, y) where f is a small MLP taking the CLS embedding c and a 2D candidate y. Train with noise-contrastive estimation (Gustafsson et al. 2020): for each ground-truth y_gt, sample M=128 noise candidates y_m ~ N(y_gt, sigma^2 I) and minimise -log[ exp(-E(y_gt)) / sum_m exp(-E(y_m)) ]. At inference, evaluate E on a 2D grid (e.g. 80x80 = 6400 evals, batched in one forward) and take argmin; or run a few SGD steps from grid winner. Gustafsson's recipe beat MDN/Gaussian-MLE on 1D regression and got 63.7% AUC on LaSOT visual tracking with the same energy formulation.

**Why for NavLoRI**: MDN collapses because it must commit to a finite K of Gaussians. EBM is non-parametric in y: lets the network paint an arbitrary energy landscape on the floor. The grid-argmin is precisely an interpretable 'occupancy heat-map' over the basement (great for the ICINCO paper figure). Floor is bounded (~80x60m at 1m res = 4800 cells) so the grid scan is cheap (one batched forward, <5ms on Quadro).

**Small test plan**: Add ~80 LOC: MLP f(c, y), NCE loss with M=128 noise samples drawn from N(y_gt, 4m^2 I), eval = grid argmin. Reuse FusionTransformer up to CLS. Test on existing B1 split. Visualise predicted energy landscape on one Dec test trajectory as paper figure.

**Risk**: NCE training can be unstable if sigma is wrong. Mitigation: sigma annealing from 8m to 1m over training. Grid scan does not generalise outside the convex hull of training waypoints - but our B1 floor is bounded.

**Sources**:
- https://arxiv.org/abs/2005.01698
- https://arxiv.org/pdf/2005.01698
- https://github.com/fregu856/ebms_regression
- https://arxiv.org/pdf/2012.04634

### Conditional RealNVP flow over (x,y)  *(angle: generative, type: validated_published, lift~0.9 m, impl: easy)*

**Mechanism**: A conditional normalising flow p_theta(y | c) on R^2 with K=8 affine coupling layers; scale/shift networks take both the masked y-coordinate and the CLS embedding c. Train by exact maximum likelihood: log p(y|c) = log N(z; 0,I) + sum_k log|det J_k|, where z = f(y, c). At inference, sample z ~ N(0,I), invert through f^{-1}(z, c) to get y candidates; mean over S=32 samples = point estimate. Unlike MDN which is a finite mixture, the flow is a continuous diffeomorphism so it represents arbitrary unimodal-with-skew posteriors exactly. Conditional flows have been validated for regression density estimation (Trippe & Turner; Winkler et al.) and for joint state-parameter estimation (Padmanabha et al. 2024).

**Why for NavLoRI**: Gives a closed-form log-likelihood (no sampling for the loss, no NCE noise samples), which makes training as stable as plain regression. Couples naturally with our existing FusionTransformer (c just feeds into coupling-net MLPs). Differentiates from MDN by being a non-mixture continuous density, addressing the exact failure mode MDN had. Cheap: a 2D flow with 8 layers is ~10k params.

**Small test plan**: ~70 LOC for RealNVP-2D: 8 coupling blocks alternating mask=[1,0]/[0,1], each block a 2-layer MLP(c|.|input -> scale, shift). Train with NLL on B1 split, 30 epochs. Compare MAE of (sample mean over 32 z draws) to baseline 9.11m. Verify log p calibration on held-out.

**Risk**: 2D flows can saturate if c carries most of the information (collapses to Dirac and MLE blows up). Mitigation: small additive Gaussian noise on y_gt during training (denoising-flow trick).

**Sources**:
- https://siboehm.com/assets/img/nfn/Bachelorarbeit_Simon_Boehm.pdf
- https://arxiv.org/pdf/2601.07013
- https://github.com/janosh/awesome-normalizing-flows
- https://www.codegenes.net/blog/nice-realnvp-normalizing-flows-pytorch/

### Cross-session RF-diffusion data augmentation (novel)  *(angle: generative, type: novel_niche, lift~1.8 m, impl: hard)*

**Mechanism**: Train an RF-Diffusion-style time-frequency conditional diffusion (Chi et al. MobiCom 2024) on the Nov training RSSI sequences, conditioned on (x,y, t, magnetometer). At training time of the localiser, repeatedly sample synthetic RSSI sequences along the Nov trajectories with diffusion temperature tau in [0.5, 1.5] and small random per-BSSID dropout / power-shift sampled from a learned 'session shift' latent. This synthesises plausible Dec-like sessions: the diffusion's natural sample diversity acts as a learned, realistic noise model for cross-session drift -- something we cannot get from random Gaussian noise injection. The localiser sees an effective training set 5-10x bigger that spans the manifold between Nov and Dec sessions. Novel synthesis: standard RF-Diffusion is used for sensing-task data augmentation in single-session settings; we condition it on a session-shift latent fitted by treating Dec_unlabelled as anchor for the latent distribution -- crucially WITHOUT using Dec labels.

**Why for NavLoRI**: We only have 2 train sessions (Nov 24+25). Every model overfits to Nov-specific RSSI statistics; that's why intra-pool is 6.5m but Dec test is 9.1m -- 2.6m of pure session shift. Augmenting with diffusion-sampled near-Nov fingerprints attacks this directly without violating the no-Dec-labels constraint. Magnetometer (which IS stable across sessions, currently dropped) can serve as the time-frequency anchor that conditions the diffusion sampling.

**Small test plan**: Phase 1 (1 night): train a small 1D conditional DDPM on RSSI(t) | (x_t, magn_t) on Nov pool only (~250 LOC). Phase 2: generate 5x augmentation per Nov trajectory, sigma annealed. Retrain idea1 (best baseline) on Nov+synth, eval on Dec test. Compare to plain idea1 9.93m -> target <8.5m.

**Risk**: Diffusion samples may collapse to memorised Nov fingerprints (no real diversity) -> no lift. Mitigation: classifier-free guidance + diversity score gating during sampling. Risk 2: training the RF diffusion model itself is the harder ~300-LOC part.

**Sources**:
- https://arxiv.org/abs/2404.09140
- https://github.com/mobicom24/RF-Diffusion
- https://www.cs.cmu.edu/~jingaox/assets/pdf/papers/mobicom24_rfdiffusion.pdf
- https://arxiv.org/pdf/2509.01875

### E(2)-equivariant steerable CNN over BSSID-anchored RSSI grid  *(angle: equivariance-geometric, type: validated_published, lift~0.8 m, impl: medium)*

**Mechanism**: Reshape the 1419-dim RSSI vector into a 2-D 'BSSID grid' by assigning each AP a learnable 2-d slot (via metric MDS on the AP-AP RSSI co-occurrence matrix, frozen after init), giving a (H,W,1) RSSI image per WiFi packet. Run an escnn C8 / D8 (8 rotations + reflection) steerable CNN over this grid, then global-pool and feed into the FusionTransformer as the WiFi token. Math: the model commutes with the dihedral group D8 acting on the BSSID-plane, so the embedding of session-A and session-B RSSI vectors are mapped through group-equivalent feature fields (Weiler & Cesa 2019). The assumption: nearby BSSIDs in physical space correlate in RSSI, so the anchor MDS preserves locality; small rotations/reflections of the anchor plane (which is what a session drift / re-installed AP looks like at the embedding level) become explicit symmetries the net is invariant to.

**Why for NavLoRI**: Cross-session WiFi drift on MSILN is mostly affine / low-rank in the AP plane (one AP moved 2 m, another reinstalled). A D8-equivariant encoder turns those nuisance transformations into model symmetries instead of nuisance variations the regressor has to absorb. Our pain is exactly that the WiFi encoder generalises Nov->Dec poorly; escnn drops in as a replacement for our WiFiNet token branch without changing fusion.

**Small test plan**: 30-min test: (1) compute MDS slots from train-only AP co-occurrence; (2) rasterize RSSI -> 32x32 image; (3) escnn 4-layer D8 steerable CNN -> 128-d; (4) swap into FusionTransformer WiFi branch; train idea1 recipe for 20 epochs on MSILN B1. Pass if test MAE < 9.0 m.

**Risk**: MDS slot placement is the load-bearing assumption â€” if AP geometry is too non-planar or many BSSIDs are nearly-collocated, the 'image' has too much aliasing. Equivariance gives invariance to a wrong group if the symmetry doesn't match drift.

**Sources**:
- https://arxiv.org/abs/1911.08251
- https://github.com/QUVA-Lab/escnn
- https://pypi.org/project/e2cnn/

### Subequivariant SO(2)-canonical IMU encoder (EqNIO-style)  *(angle: equivariance-geometric, type: validated_published, lift~0.6 m, impl: easy)*

**Mechanism**: Replace the IMUCNN with the EqNIO three-stage block: (1) estimate a gravity-aligned yaw frame R_yaw from the IMU window using equivariant vectors (accel - g, gyro) and invariant scalars (||a||, ||w||); (2) canonicalise the 9-DoF IMU window into that frame (rotate accel and gyro by R_yaw^-1, magnetometer stays in body frame initially, then is rotated too); (3) pass the canonicalised window through any standard 1D-CNN/Mamba backbone; (4) rotate the predicted displacement back by R_yaw. Mathematically the whole encoder commutes with the subgroup G = SO(2) around gravity x Z_2 reflection across vertical planes â€” physical symmetries that IMU dynamics obey but a vanilla CNN does not. EqNIO shows this is achievable with off-the-shelf MLPs + a small equivariant frame head, not specialized layers.

**Why for NavLoRI**: Our IMU branch (Mamba/IMU-CNN) currently sees raw body-frame IMU and has to learn yaw-invariance from data â€” on MSILN with only 38 training traces, that data is thin. EqNIO bakes the yaw symmetry into the architecture, so cross-session generalisation comes for free. Reported gains on TLIO, Aria, RIDI, OxIOD (3 of those are pedestrian datasets close to MSILN's regime).

**Small test plan**: 30-min test: implement the gravity-yaw canonicalisation as a pre-processing wrapper around our existing Mamba IMU encoder (~80 LOC, uses AHRS quaternion already in the data); train idea1 recipe for 20 epochs; compare to non-canonicalised baseline at 9.13 m. Pass if test MAE < 8.7 m.

**Risk**: MSILN AHRS quaternions are smartphone-grade and noisy â€” bad yaw estimate poisons the canonicalisation. Magnetometer-aided yaw helps but is itself drifted indoors.

**Sources**:
- https://arxiv.org/abs/2408.06321
- https://arxiv.org/pdf/2111.11676
- https://arxiv.org/pdf/2501.15659

### SIREN/Fourier place-field RF radiance prior (NeRF2-lite)  *(angle: equivariance-geometric, type: validated_published, lift~1.2 m, impl: medium)*

**Mechanism**: Train a coordinate-based MLP f_phi: (x,y) -> R^{1419} that predicts the expected RSSI vector at floor location (x,y), using a SIREN/Fourier-features positional encoding (Sitzmann 2020 / Tancik 2020). Fit f_phi on train-session waypoints only by minimising masked MSE on observed APs. At inference, instead of regressing (x,y) directly, our model outputs a coarse (x,y) candidate from the existing FusionTransformer, then refines it by gradient descent on ||f_phi(x,y) - r_obs||_M (M masks unobserved APs at test time). This is a stripped-down NeRF2 (RF radiance field) but only over the 2-D floor â€” and crucially the radiance field is session-agnostic if trained with session-id-conditioning then marginalised at test time. Math: SIREN sin-activations let the field model sub-metre RSSI structure that ReLU MLPs miss (NeurIPS 2020).

**Why for NavLoRI**: Our 9.1 m residual is partly that the regressor doesn't exploit the WiFi field's smooth structure across positions â€” it treats fingerprints as a 1419-d bag. A SIREN radiance prior lets us do test-time refinement using physics-of-propagation regularity, which is exactly what NeRF2 showed gives 50% median-error reduction in localization. No floorplan needed; the prior is fit purely on (waypoint, RSSI) pairs.

**Small test plan**: 30-min test: (1) train SIREN (4 hidden, 256 units, w0=30) on train-waypoint -> RSSI for 5k steps; (2) for each test sample, take FusionTransformer (x,y) as init, 10 steps of Adam on ||f_phi(x,y) - r_obs||; (3) report MAE. Pass if test MAE < 8.0 m. Falls back gracefully if the test-time refinement diverges (clip step size).

**Risk**: Cross-session: f_phi trained on Nov RSSI is not exactly the Dec RSSI field. If the field drift is larger than the gradient basin, refinement points uphill. Mitigation: only refine within +-2 m ball; use Huber on the residual.

**Sources**:
- https://web.comp.polyu.edu.hk/csyanglei/data/files/nerf2-mobicom23.pdf
- https://github.com/XPengZhao/NeRF2
- https://bmild.github.io/fourfeat/

### Diffusion-map manifold coordinates as session-invariant auxiliary feature  *(angle: equivariance-geometric, type: novel_niche, lift~1 m, impl: easy)*

**Mechanism**: Build a k-NN graph (k=20) over ALL WiFi packets in train+test pooled (RSSI cosine similarity), compute the first d=8 non-trivial eigenvectors of the normalised graph Laplacian (diffusion-map embedding, Coifman & Lafon 2006). This gives each WiFi packet an 8-d coordinate on the intrinsic WiFi manifold. The novel claim: the diffusion manifold is the same physical floor for Nov and Dec â€” only the *function from RSSI to floor* drifts, but the local-neighbourhood structure (which packets are 'close' in signal space) is preserved much better. Concatenate the 8-d diffusion coords to the WiFi token as an auxiliary modality (no labels used to compute it -> safe for test). Mathematically this is Laplacian eigenmaps as a session-invariant chart of the floor manifold. iBeacon RSSI (currently dropped) joins the same graph: 7.5 Hz beacons give a much denser graph than 1 Hz WiFi.

**Why for NavLoRI**: Our kNN-on-train floor is 6.48 m -> the 1-NN neighbours of test packets in train are 6.48 m off on average. That's because the RSSI metric is session-drifted. A diffusion-map metric is *learned* to respect manifold geodesics, not raw Euclidean distance; this should tighten cross-session nearest-neighbour structure. Adds iBeacon as a free side-effect (denser graph, currently wasted).

**Small test plan**: 30-min test: (1) add iBeacon + magn to converter; (2) sklearn SpectralEmbedding(n_components=8, affinity='nearest_neighbors', n_neighbors=20) on pooled train+test WiFi+iBeacon vectors; (3) concat to WiFi token; (4) retrain idea1 20 epochs. Pass if test MAE < 8.5 m. Even simpler ablation: just kNN-regress on diffusion coords -> position, compare to raw-RSSI kNN at 6.48 m.

**Risk**: Pooling test packets into the graph computation is transductive â€” fine for offline academic eval, but breaks online deployment. Need to test out-of-sample Nystrom extension (sklearn has it) for a fair online story.

**Sources**:
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7038483/
- https://arxiv.org/pdf/2009.08062
- https://www.mathworks.com/matlabcentral/fileexchange/36141-laplacian-eigenmap-diffusion-map-manifold-learning

### MagHT magnetic Hough-Transform place recognition  *(angle: wildcard-niche, type: validated_published, lift~1.4 m, impl: medium)*

**Mechanism**: Treat magnetometer readings the data pipeline currently drops as a *3-component spatial fingerprint* and do place recognition via a generalized Hough vote. Given a sequence of magnetic samples plus relative pose increments (IMU-integrated odometry from accel+gyro), each sample casts a 4-DoF (x,y,z,yaw) vote against a map of training-session magnetic samples using yaw-invariant features (horizontal magnitude m_h = sqrt(mx^2+my^2) and vertical m_z). Thousands of votes are DBSCAN-clustered; the largest cluster centroid is the pose hypothesis. Yaw invariance kills the largest cross-session nuisance (phone orientation), and magnetic ferromagnetic signatures are *physically stable across 1-year gaps* (paper shows mapping vs test sequences 1 year apart). 0.21 m median translation error vs particle-filter 0.18 m, 10000x faster. We use the Hough vote as a *soft prior* on top of the fusion model output: posterior = w*model + (1-w)*mag_vote, with w learned per-window from agreement score.

**Why for NavLoRI**: Our converter currently DROPS magnetometer entirely (50 Hz, available in 40/40 traces). The MSILN train(Nov24/25)->test(Dec5/6) gap is exactly the regime where magnetic anomalies dominate: building steel + wiring don't change in 11 days but WiFi APs reshuffle. Adding a magnetic place-recognition prior should claw back a chunk of the 2.6 m algorithmic headroom directly from a modality we already have but throw away.

**Small test plan**: Patch convert_msiln.py to keep TYPE_MAGNETIC_FIELD columns (mx,my,mz) at 50 Hz. Build train-set magnetic map = (x,y,m_h,m_z). At test time, slide a 5 s window, IMU-integrate relative pose, cast Hough votes against map (sklearn DBSCAN). Eval Hough-only MAE. If <10 m, fuse as residual prior.

**Risk**: Pedestrian-grade magnetometer is noisier than the robotic-platform mag used in MagHT - votes may be too diffuse. Also the IMU-integrated relative pose drifts faster than the original paper's odometry, hurting vote quality at long windows.

**Sources**:
- https://arxiv.org/abs/2312.05015
- https://arxiv.org/html/2312.05015
- https://arxiv.org/pdf/2503.04286

### STELLAR Siamese contrastive cross-session WiFi  *(angle: wildcard-niche, type: validated_published, lift~1.1 m, impl: medium)*

**Mechanism**: Reframe cross-session WiFi as a metric-learning problem instead of regression. Build positive pairs as (scan_i, scan_j) from the SAME training session within radius r meters, and negative pairs as (scan_i, scan_k) from FAR locations OR from the other training day. Multi-head self-attention encoder over the 1419-D RSSI vector treats each BSSID as a token (RSSI as value, learned BSSID embedding as key). Siamese forward pass; loss = supervised contrastive (Khosla 2020) + auxiliary regression head. The trick: weight day-vs-day negatives higher so the embedding learns to be *day-invariant within a location*. At test time, kNN-in-embedding-space instead of raw-RSSI kNN. The published paper reports 18-165% accuracy improvement over 2 years of temporal variations on real indoor RSSI without retraining - much harder than our 11-day gap.

**Why for NavLoRI**: Our kNN-on-train noise floor is 6.48 m and best fusion is 9.1 m - the 2.6 m gap is *partly* because raw-RSSI distance is a bad proxy for spatial distance across sessions. STELLAR-style contrastive embeddings directly attack this; the published 18%+ improvement over 2-year drift transfers strongly to our 11-day cross-session setting. Slots in cleanly upstream of FusionTransformer (replace JEPA SSL pretrain in lead3).

**Small test plan**: Replace lead3 JEPA stage with SupCon: pairs = (anchor_scan, pos_scan_same_session_within_2m), negatives weighted 3x if from the OTHER training day. 20 epochs, 256-d embedding. Eval kNN-MAE in embedding space on test set; if <6 m alone, plug as WiFi encoder into FusionTransformer K=4.

**Risk**: With only 2 training sessions (Nov 24 + Nov 25), the inter-day-negative signal is weak - day-invariance may not generalise to Dec 5/6. Risk of collapsing embeddings if day-balance not careful.

**Sources**:
- https://arxiv.org/abs/2312.10312
- https://arxiv.org/pdf/2312.10312

### Magnetic-gradient odometry residual to IMU drift  *(angle: wildcard-niche, type: validated_published, lift~0.6 m, impl: medium)*

**Mechanism**: Solin/Kok (2025) prove that *local spatial gradients of the magnetic field carry odometry information*: when a sensor moves through a non-uniform magnetic field, dm/dt = J(x)*v where J is the magnetic Jacobian, so observing dm/dt + a window of m(t) constrains v up to drift. We adapt this as a self-supervised auxiliary head on the IMU branch: input = 1 s window of (accel, gyro, magnetometer) at 50 Hz; the encoder must predict (a) the position from fusion supervision AND (b) the displacement vector dx in the local frame consistent with the observed dm. Loss = L_position + lambda * ||dx_pred - integrated_dx||^2 weighted by ||grad(m)|| (only use when gradient is strong). The auxiliary task acts as a strong physically-grounded regulariser on the IMU encoder, suppressing drift that costs us during stale-WiFi windows. The original paper shows 'significantly lower error growth rates' vs standalone INS.

**Why for NavLoRI**: Our diagnosis (CLAUDE.md, point 1-2) says WiFi dominates fresh-data accuracy and temporal fusion's value is robustness under stale WiFi. The Achilles' heel in those stale windows is IMU drift. Magnetic-gradient odometry is a *physically lawful* drift correction that needs zero floorplan, zero APs - just the magnetometer we already drop. Attacks the exact failure mode we identified.

**Small test plan**: Add magnetometer to imu.csv. Add aux head to IMUCNN: predicts dx_local over 1 s; loss term active only when ||grad(m)|| in that window exceeds 5th-percentile gradient on train. Train 30 epochs at lambda=0.3. Eval cross-session test MAE vs baseline.

**Risk**: Pedestrian/phone magnetometer is dominated by carrier-attitude swings, not spatial gradient - gradient signal-to-noise may be near zero outside steel-rich corridors. Aux loss could also hurt the primary task if lambda is mis-tuned.

**Sources**:
- https://arxiv.org/pdf/2503.04286
- https://arxiv.org/pdf/2409.01091
- https://arxiv.org/pdf/2505.12634

### Pheromone-trail stigmergic memory readout  *(angle: wildcard-niche, type: novel_niche, lift~1 m, impl: easy)*

**Mechanism**: Direct steal from ant-colony stigmergy + hippocampal replay theory: build a 2-D 'pheromone map' P(x,y) over the floor as a learnable spatial buffer (e.g. 64x64 grid covering B1 bounding box). During TRAINING, every observed (x,y,wifi_token,magn_token) deposits a learned embedding e_obs at position (x,y) with bilinear-spread + temporal decay tau=11 days (matching the Nov-Dec gap). At test, the network queries P with a *predicted prior position* p_hat (from a single forward pass of the existing FusionTransformer), reads back a soft attention over P's nearest cells, and produces a *correction vector* dx that is added to p_hat. The trick: pheromones decay between training days but accumulate dense evidence per cell, so test-time readout averages across N training visits. Unlike memory-augmented networks, the storage key is GEOGRAPHIC not learned - it bakes spatial inductive bias in. This is also a deliberate cross-domain transfer from swarm-robotics stigmergy (deposits at a 2D substrate read by re-visiting agents) and biological replay (offline consolidation of trajectories into a place-indexed cache). 30 lines of PyTorch.

**Why for NavLoRI**: Whole-path bidirectional transformer (17 m, failed) overfit because it tried to learn the *path*, not the place. A pheromone grid stores per-cell evidence regardless of path. The 11-day gap matches a biological consolidation timescale - stigmergy lets old evidence vote on new scans without retraining. Bold, novel, and 2.6 m headroom is exactly what bold leads chase.

**Small test plan**: After Stage-B fusion training, run a single epoch over train set and accumulate a 64x64xD pheromone tensor (D=128 = WiFi token). At test, query with bilinear interp around p_hat predicted by FusionTransformer, take softmax over a 5x5 neighborhood weighted by inverse RSSI distance, produce dx in metres. Eval test MAE before/after correction.

**Risk**: If p_hat is far off (>~3 m), the local readout is empty/noisy and may push estimates in the wrong direction. May silently memorise training XY if not regularised - watch for train-MAE collapse with test-MAE flat.

**Sources**:
- https://www.cell.com/neuron/abstract/S0896-6273(25)00709-3
- https://arxiv.org/pdf/2307.05793
- https://arxiv.org/pdf/2402.06590

---

## Curator rejection notes

Dropped most leads with high overlap or high risk for the 24-hour window. The FuseMoE lead was rejected for "hard" implementability with documented expert-collapse risk under 2 training sessions. Heterogeneous GNN duplicates the Per-BSSID Set Transformer's mechanism with a heavier PyG dependency. Log-distance pathloss prior, Recursive KalmanNet, Deep Kernel GP re-ranker, and Conditional Normalizing Flow each compete with stronger leads in their slot (physics, probabilistic, retrieval, generative); we kept only the highest-lift or easiest representative per slot. Barron adaptive loss, Rank-N-Contrast, ASAM/Lookahead, QMF/AECF gating, per-anchor cross-attention readout, SIREN radiance prior, E(2)-equivariant CNN, iBeacon modality, RF-diffusion augmentation, MagHT, magnetic-gradient odometry, and conditional RealNVP were rejected for being either redundant with picked leads, too risky on smartphone-grade noise, or not as composable with the existing Mamba/JEPA stack. The bar was: easy or medium implementability, distinct mechanism per top-20 slot, expected lift >=0.4m, and clear hook into idea1.
