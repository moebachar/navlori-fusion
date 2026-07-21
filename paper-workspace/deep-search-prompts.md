# Deep-Search Prompts — Related Work (ICINCO 2026)

Prompts to hand to the deep-search agent(s) for the Related Work literature
gap-fill. Phase 1 (Zotero/NotebookLM inventory) showed groups ①②③ are
citation-ready from the library; the spend below is narrow and high-value:
the **method pillars** and **experiment benchmarks** missing from Zotero,
plus a **novelty/competitor sweep**.

**How to use:** prepend the *Shared preamble* to each numbered prompt before
giving it to an agent. Run them independently (parallel is fine). Bring the
reports back here; I will de-duplicate against Zotero, verify, then load the
corpus into per-group NotebookLM notebooks (Phase 3.5) for the iterative
querying phase.

---

## Shared preamble (prepend to every prompt)

> Context. I am writing a double-blind ICINCO 2026 paper on a
> **continuous-time set-transformer** that fuses **asynchronous WiFi RSSI
> (~1 Hz) and IMU (~30 Hz)** streams for indoor **(x, y)** localization. Its
> contributions are: (i) a **continuous-time encoding** of each observation's
> elapsed time Δt, so unequal/asynchronous rates need no resampling;
> (ii) a **single set-transformer** whose one self-attention block performs
> **both cross-modal and cross-time fusion** — no per-modality branches,
> permutation-invariant over modalities; (iii) **async-robustness** via
> modality- and instant-dropout giving **graceful degradation** under missing
> or stale sensors, demonstrated by **real-world cross-session** generalization.
> I am building the Related Work section.
>
> Output for EVERY reference you return: (1) full BibTeX-ready metadata —
> full author names, exact title, venue/journal/conference + publisher, year,
> volume/pages if any, and DOI and/or arXiv ID; (2) a 2–3 sentence method
> summary; (3) one sentence on how it relates to or differs from the paper
> above, and which group it fits — {WiFi fingerprinting | inertial/IMU |
> multimodal fusion | attention/transformer/set | continuous-time/async}.
> Verify each item is a real publication and the metadata matches the
> canonical version (prefer the published version; note if only a preprint
> exists). If a paper I name does not exist or differs from my description,
> say so explicitly. **Do not fabricate references, DOIs, or venues.**

---

## Prompt 1 — Architecture foundations (set + attention)

Retrieve and verify the foundational references this paper builds on, and
surface closely-related permutation-invariant fusion work.
Must-find (verify exact metadata): (a) Vaswani et al., "Attention Is All You
Need" (2017); (b) Lee et al., "Set Transformer: A Framework for
Attention-based Permutation-Invariant Neural Networks" (ICML 2019).
Then exhaustively search for works that apply **set-transformers or
permutation-invariant attention to multi-sensor / multimodal fusion** (any
domain, especially localization or state estimation). Return the two pillars
plus up to ~6 most-relevant set/permutation-invariant-fusion works.

## Prompt 2 — Continuous-time / irregular sampling / time encoding

This is the pillar of our "continuous-time" contribution — be exhaustive.
Must-find (verify): (a) Shukla & Marlin, "Multi-Time Attention Networks for
Irregularly Sampled Time Series" (mTAN, ICLR 2021); (b) Chen et al., "Neural
Ordinary Differential Equations" (NeurIPS 2018); (c) Rubanova et al., "Latent
ODEs for Irregularly-Sampled Time Series" (NeurIPS 2019); (d) Kazemi et al.,
"Time2Vec: Learning a Vector Representation of Time" (2019).
Then exhaustively find recent (2020–2025) models for **irregularly-sampled /
asynchronous / multi-rate time series** and **continuous-time / time-aware
encodings**, especially any applied to sensor fusion or localization. For each,
state precisely HOW it handles irregular/async time (ODE solver vs learned
time embedding vs interpolation/attention) versus our approach (a learned
sinusoidal encoding of real-valued Δt added to each token). Return the 4
pillars + up to ~8 most-relevant.

## Prompt 3 — WiFi localization benchmarks & baselines we use

Retrieve and verify the canonical citations for the WiFi datasets/baselines in
our experiments. Must-find (verify exact metadata): (a) the **UJIIndoorLoc**
dataset paper (Torres-Sospedra et al., IPIN 2014) — canonical citation;
(b) **CNNLoc** — the CNN-based WiFi indoor-localization method (find exact
paper, authors, venue, year); (c) a citable reference for the kNN/RSSI
fingerprinting approach implemented by the open-source **wlan_localization**
project (GitHub: sharan-naribole) — its underlying method's canonical paper or
the repo's own stated reference; (d) the classic RSSI kNN fingerprinting
reference (e.g., RADAR, Bahl & Padmanabhan, INFOCOM 2000) as the classical
baseline. Return each with verified metadata.

## Prompt 4 — Learned inertial navigation

Confirm/retrieve learned-inertial-navigation references. (a) Verify **RoNIN**
(Herath, Yan, Furukawa) — confirm canonical venue/year (ICRA 2020 vs the 2019
arXiv); (b) find/verify **IONet** (Chen et al., AAAI 2018); (c) find/verify
**TLIO** (Liu et al., 2020) if it exists. Then briefly survey current SOTA in
**deep inertial odometry / pedestrian dead-reckoning (2020–2025)** to confirm
whether a stronger learned-inertial baseline than RoNIN ResNet1D should be
acknowledged. Return verified metadata + one line each on relation to our IMU
encoder (a lightweight 1-D CNN, ~0.05 M params).

## Prompt 5 — Closest competitors / novelty defense  (HIGHEST PRIORITY)

Find our CLOSEST prior works so we can position against them and confirm
novelty. Exhaustively search for **deep-learning methods that fuse WiFi (RSSI
or CSI) with IMU/inertial for indoor localization** AND have ANY of: (i)
attention/transformer fusion; (ii) continuous-time or asynchronous/multi-rate
handling without resampling; (iii) robustness to missing/stale modalities /
graceful degradation; (iv) cross-session / cross-day real-world generalization
evaluation. For EACH close work, state explicitly which of (i)–(iv) it has and
lacks, whether it uses a **single unified fusion block vs per-modality
branches**, and whether it handles **real-valued time gaps**. We already hold
these (go BEYOND them; you may confirm them): Yu 2022 "Multi-Modal Recurrent
Fusion", Antsfeld 2020, WIO-EKF 2024, WiMU 2025, AFT-VO 2022, A-KIT 2024.
Deliver a ranked list of the 8–12 nearest competitors, each with a crisp "how
it differs from a continuous-time unified set-transformer with modality/instant
dropout."

## Prompt 6 — Recent transformer / deep indoor-localization frontier (recency sweep)

Ensure Related Work cites the current frontier. Exhaustively find **2023–2025
transformer-based or deep multimodal indoor-localization** papers NOT in this
exclude list (already held): Ott 2024 (Radio Foundation Models), Bhatia 2025
(decoder-only transformer), Zhang 2022 (TIPS), SwinULoc 2026, Abdullah 2025
(RIS transformer), Tiku 2022 (multi-head attention), Ai 2022 (DRVAT),
Zhang 2023 (CSI attention-ResCNN), and the surveys Liu 2025 / Wang & Ahmad
2025 / Martín-Frechina 2025 / Feng 2022. Return up to ~10 additional recent,
high-relevance works with verified metadata, prioritizing venue quality and
direct relevance to WiFi/IMU localization with attention or async fusion.

---

## After the reports come back (my next steps, for reference)

1. De-duplicate every returned reference against the Zotero inventory; drop
   anything already held.
2. Verify metadata (canonical version, DOI/arXiv, diacritics) before anything
   enters `refs.bib`.
3. Map each kept reference to its group ①–⑤; flag the 8–12 competitors from
   Prompt 5 against gap-criteria (i)–(iv).
4. Phase 3.5 — create the 5 per-group NotebookLM notebooks, load Zotero
   holdings + verified new finds as sources.
5. Phase 4 — the excessive iterative-query loop over those notebooks.
