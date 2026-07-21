# ICINCO 2024 — Observed Style & Conventions

Calibration notes for writing the ICINCO 2026 submission. These are
**observations of accepted ICINCO 2024 papers**, not a template to copy.
The authoritative formatting rules live in the `paper-format` skill and
`conference-rules.md`; this file records what accepted authors actually
*did*, so our prose lands at the right register.

**Evidence base.** Derived from iterative NotebookLM queries against the
"ICINCO 2024 proceedings" notebook (2 volumes, ~100+ papers). The
sentence-level rhetorical observations sample these papers in detail
(page = Volume/start):
- Triplet Neural Networks for Visual Localization (V2 p125)
- Uncertainty-Aware DNN for Multi-Modal Camera Localization (V2 p80)
- A Modular Multimodal Multi-Object Tracking (V2 p336)
- Multimodal 6D Detection of Industrial Pallets (V2 p345)
- Characteristics-Based LCM Indoor Positioning (V1 p301)
- Autonomous Forklift Navigation (V2 p327)
- Drone Warehouse Localization (V2 p357)
- RoboMorph (V2 p149)
- Multi-Step Simulation Time Series (V1 p651)

The 9-paper rhetorical sample skews toward robotics/localization papers
(the cohort our paper sits in). Numbers below are stated as fractions of
that sample unless noted "(notebook-wide)".

---

## 1. Structural norms

### Section ordering — highly regular
All 6 papers structurally analysed follow the same IMRaD skeleton:

```
1 INTRODUCTION  →  2 RELATED WORK / PREVIOUS WORK / STATE OF THE ART
              →  3 METHOD(OLOGY) / PROPOSED-X  →  4 EXPERIMENTS / RESULTS
              →  5 CONCLUSIONS
```

- **Related Work comes BEFORE Methodology in 6/6 sampled papers.** It is
  numbered as Section 2 in 5/6; the forklift paper instead folds it into
  the Introduction as subsection "1.2 Navigation Strategies". Title varies:
  "RELATED WORK" (most common), "PREVIOUS WORK" (Triplet), "STATE OF THE
  ART" (Drone). **Put our Related Work before Method.**
- **Top-level section headings are ALL-CAPS and numbered** (`1 INTRODUCTION`,
  `2 RELATED WORK`), matching the SCITEPRESS template. Subsections are
  numbered titlecase (`3.1 Proposed Architecture`, `4.2 Baseline System`).

### Section count
- **5 top-level sections is the mode** (Intro, Related Work, Method,
  Experiments, Conclusions). Range observed 5–7. The Triplet paper runs to
  7 by splitting Method into "3 TRIPLET NETWORK" + "4 VISUAL LOCALIZATION"
  and adding "6 COMPARISON WITH OTHER WORKS" before Conclusions.
- **Plan for 5–6 numbered sections.** Our scope.md's 8-section plan
  (Intro / Related / Method / Per-leg validation / Experiments / Ablations
  / Discussion / Conclusion) is on the high end — consider merging
  Ablations into Experiments and Discussion into Conclusions to land at
  ~6, which matches the cohort.

### Length distribution
- The proceedings are split into explicit **"FULL PAPERS"** and **"SHORT
  PAPERS"** subsections (notebook-wide). Observed page spans in the sample:
  Triplet 8pp, LCM 8pp, Drone 8pp, Forklift 9pp, MOT 9pp, Camera Loc 11pp,
  Pallets 8pp. **Most robotics/localization papers run 8–11 pp**, clustering
  at 8–9. Note ICINCO 2026 reviews on **character count (10k–50k excl.
  spaces)**, not pages — these page counts are only a rough guide.

### Results vs Discussion — MERGED
- **6/6 sampled papers merge Results and Discussion.** There is no separate
  "Discussion" section; interpretation happens inline within
  "EXPERIMENTS" / "EXPERIMENTAL RESULTS" / "EXPERIMENTAL VALIDATION", with
  final synthesis pushed into "CONCLUSIONS". **A standalone Discussion
  section would be atypical for the venue** — our scope.md §7
  Discussion/Limitations can be folded into Conclusions or kept short.

### Limitations — NO dedicated section
- **0/6 sampled papers have a standalone Limitations section.** Limitations
  surface either (a) hedged inline in Results, or (b) as future-work in the
  Conclusion (see §5 below). Our honest-limitations content (smoothness
  debt, C2 gap, MSILN path-130) is therefore *above* venue norm in candour
  — present it inside Discussion/Conclusion prose rather than a flagged
  "Limitations" header, to read as confident self-assessment, not apology.

---

## 2. Methodology norms

### Notation: equations + diagrams, mixed
- **Numbered equations are common but not universal.** 4/6 sampled papers
  use numbered display equations (Triplet: triplet/circle-loss eqs;
  Camera Loc: Eq (5)–(10) for evidential loss; LCM: Eq (1)–(9); Pallets:
  Eq (4)–(5)). 2/6 (Forklift, MOT) are **almost equation-free**, carrying
  the method in block/pipeline diagrams and prose.
- **No symbol/notation tables observed** in the sample — symbols are
  defined inline at first use (e.g. Triplet defines `D⃗a,p`, `s_n^j` in
  running text). **A compact notation table is acceptable but not the norm;
  if used, keep it small.**
- **Pattern:** define a loss or distance formally with 1–4 numbered
  equations, and carry the architecture in exactly one pipeline figure
  (see below). Our Method section should mirror this — formalise the token
  construction + attention + readout with a handful of numbered equations,
  not a wall of math.

### System architecture presentation — ONE pipeline figure
- **The dominant pattern is a single left-to-right pipeline/block diagram**
  labelled "Figure 1: Overview/Pipeline of …" placed early in the Method
  section. Examples: MOT "Figure 1: Overview of the object detection and
  tracking baseline pipeline"; Pallets "Figure 1: Pipeline of the proposed
  framework using RGB and Depth". Subsystem decomposition happens *within*
  that one figure (data flowing through labelled blocks), not as several
  separate architecture figures. **Build one clear pipeline figure for our
  set-transformer (tokens → self-attention → PositionQuery readout).**

### Hyperparameter reporting — INLINE PROSE
- **6/6 sampled papers report hyperparameters inline in prose**, not in a
  dedicated table and not in an appendix. Representative phrasing:
  - Camera Loc: "we trained all models from scratch for a total of 400
    epochs, by fixing a learning rate of 1e−4, … ADAM optimizer and a batch
    size of 24."
  - Pallets: "2000 epochs, a batch size of 16, … input image resolution of
    640×640 … momentum value of 0.937 and a weight decay of 0.0005."
  - Triplet: "10 epochs, with an epoch length of 25000 triplet samples …
    Stochastic Gradient Descent (SGD)."
- Hardware is named inline too ("NVIDIA GeForce RTX 3090 GPU with 24 GB";
  "a single NVidia GTX1080ti"). **Report our AdamW/OneCycleLR/90-epoch/
  B=128/K=4 settings inline in a "Training Details" subsection.** A
  reproducibility table is fine but would be slightly above venue norm.

---

## 3. Experiments norms

### Baseline count — LOW (0–3)
- Sampled baseline counts: Triplet **3** (Gist, HOG, AlexNet); Camera Loc
  **3** variants (CMRNet-no-iter, +MCD, +DE); MOT **1** (its own unimodal
  3D tracker); Pallets **1** (DenseFusion baseline); Forklift **1**
  (A* vs TEB); LCM **0** external baselines (evaluates itself across 5
  datasets). **Median ≈ 1–2 external comparison methods.**
- **Implication for us:** comparing our transformer against
  wlan_localization (+ WiFi-kNN) on MSILN, and WiFi-Net vs
  wlan_localization on UJI, and IMUCNN vs RoNIN ResNet1D — i.e. **one
  named SOTA per leg** — already *meets or exceeds* the venue's typical
  comparison density. ICINCO reviewers do ask "needs comparative
  evaluation?", so keep at least one external SOTA per claim, but we are
  not under-baselined by ICINCO standards.

### Metric reporting — BOTH tables and plots
- **5/6 sampled papers use both tables and plots; 1/6 (MOT) is
  table-only.** Tables carry the headline numbers (geometric error,
  translation/rotation error, HOTA sub-metrics); plots carry curves
  (Recall@K, calibration curves, accuracy-vs-threshold, error-vs-distance
  histograms). Captions are sentence-case, end with a period, and tables
  caption ABOVE / figures caption BELOW (per template).
- **Our plan fits:** main-results tables + staleness/K-sweep curves is the
  standard shape. Lead each table with the headline metric column.

### Ablations — informal, common; formal "Ablation" rare
- **Parameter/sensitivity studies are near-universal** (Triplet sweeps loss
  function, triplet-threshold, batch size; LCM sweeps the Distance Scale
  Factor; MOT sweeps `m_th`/`min_hits`/`max_age`). But only **1/6 (Camera
  Loc) labels a table "Ablation study"** explicitly. The rest present the
  same idea as "Experiment 2/3" or "fine-tuning" subsections.
- **Implication:** our K-axis sweep + modality-dropout-rate sweep are
  exactly the venue's idiom. Labelling them "Ablations" (as scope.md does)
  is fine and slightly more rigorous than the median paper.

### Statistical tests / confidence intervals — MOSTLY ABSENT
- **No formal statistical significance tests (t-test, etc.) in any of the 6
  sampled papers.** Reporting of variance is light: Camera Loc reports
  **mean ± std** in its results and ablation tables ("0.65 ± 0.45"); the
  others report point estimates only. **Confidence intervals are not a
  venue expectation.**
- **Implication:** reporting mean ± std (we have seeds) would put us at or
  slightly above the venue norm; full significance testing is not expected
  and not necessary to satisfy reviewers. Our R²=0.995 staleness-slope fit
  is already more quantitative rigour than the cohort median.

---

## 4. Related Work norms

- **Length:** Related Work is typically **one section of ~3–6 paragraphs**,
  well under a page in the two-column format (≈0.5–1 column in the sample).
  Our scope.md budgets 1 page for §2 — at the upper end but acceptable.
- **Grouping: methodological / thematic, not chronological.** Camera Loc
  opens "we can divide existing methods into two categories: camera pose
  regression … and place recognition …" then treats each group. MOT splits
  "tracking-by-detection" vs "joint detection and tracking". LCM groups
  clustering families ("hierarchical and partitional … like k-means").
  **This matches our paper-content skill's anti-shopping-list rule — group
  by method family (WiFi fingerprinting / inertial / multimodal-attention /
  async filtering).**
- **Paragraph template in practice:** an opening sentence naming the group
  and what it does, 1–2 sentences on the most relevant members with their
  approach, and a closing sentence on the gap or on which member they
  build from. Camera Loc's "An example is CMRNet … which performs direct
  regression … Its ultimate goal is to …" is a clean exemplar.
- **Citations render APA author-year** (Author, year), consistent with the
  required `apalike` style — never numeric.

---

## 5. Rhetorical patterns (sentence-level)

### 5.1 Introduction openings
Two dominant opening moves (notebook-wide tendency, all from the sample):

**(a) Broad domain/importance statement** — the most common:
- MOT: "As industries embrace the era of automation, catalyzed by the
  principles of Industry 4.0, the demand for AMRs … has intensified
  greatly."
- Drone: "Ensuring accurate product tracking within industrial
  environments is important for real-time inventory management and
  operational efficiency."
- Triplet: "Vision systems are a very suitable option to tackle mobile
  robot localization."
- LCM: "The advancement in IoT technology has led to a rise in
  data-intensive applications like indoor localization …"

**(b) "Strong results exist, BUT a gap remains"** — the niche-first move:
- Camera Loc: "Although DNN-based techniques achieve outstanding results in
  camera localization …, a main challenge is still unsolved: to determine
  when such models are providing a reliable localization output …"

**For our Introduction:** open with (a) the indoor-localization /
sensor-heterogeneity importance, then pivot with a (b)-style "but existing
fusion approaches assume synchronous, always-present sensors" gap sentence.
This two-step is exactly the cohort's idiom and maps onto the CARS model in
our paper-content skill.

### 5.2 Contribution statements
- **Signal phrases are universal.** Every sampled paper uses an explicit
  "In this paper / this work" frame: "this paper presents" (Triplet),
  "we propose" (Camera Loc), "In this paper, we present" (Drone), "This
  paper proposes" (RoboMorph), "The main contributions of this work are:"
  (MOT), "This paper introduces several key contributions:" (Multi-Step).
- **Format split ≈ 50/50.** ~3/6 state contributions as **running prose**
  (Triplet, Camera Loc, Drone); ~3/6 use an explicit **bulleted/numbered
  list under a "Contributions" sub-heading** (MOT "1.x Contributions",
  Multi-Step "1.1 Contributions", Forklift "1.3 Contributions").
- **Recommendation:** use an explicit numbered/bulleted contributions list
  under a "Contributions" subheading at the end of the Introduction.
  Reviewers (and the paper-content skill's CARS guidance) look for it, and
  it is well within venue norm. 2–4 items.

### 5.3 Limitations & hedging
- **Limitations are hedged and scattered, never a section** (consistent
  with §1). Modes observed:
  - *Hedged inline in Results:* Triplet — "the trained network **may have
    experienced some** overfitting to the training condition." RoboMorph —
    "challenges persist in the z-coordinate, **likely due to** the
    intrinsic complexity of the problem."
  - *As future work in the Conclusion:* Drone — "Future perspectives
    involve further refining the algorithm …" Camera Loc states no explicit
    weakness at all.
  - *Explicit but cushioned in Conclusion:* Multi-Step — "While including
    exogenous variables has clear benefits, it also presents challenges.
    These variables may not always be … available in real-time …" then
    immediately "**Despite these challenges**, the method significantly
    improves …".
- **Hedging vocabulary:** "may have", "likely due to", "can introduce",
  "potentially impacting", "while effective, …". The register is **cautious
  about own weaknesses, assertive about own gains** — claims of improvement
  are stated flatly ("improve substantially", "significant improvements",
  "outstanding performance"); weaknesses are always softened.
- **Calibration for us:** our scope.md commits to *unhedged* honest
  limitations (smoothness debt falsified, C2 +94% gap, MSILN path-130).
  That is more candid than the venue median. Keep the candour (it is
  defensible and reviewers reward it), but adopt the venue's "concede →
  **Despite this** → restate value" sentence shape so honesty reads as
  confidence. E.g. "Our fusion inherits an architecture-invariant
  smoothness debt (r < 0.10); despite this, absolute-position accuracy is
  competitive, and we identify auxiliary velocity loss as the fix."

### 5.4 Section-preview sentence — near-universal
- **Almost every paper ends the Introduction with an explicit roadmap
  sentence:** "This manuscript is structured as follows. Section 2 reviews
  … Section 3 details … Section 7 we discuss …" (Triplet); "The remainder
  of this paper is organized as follows: Section 2 reviews … Section 5
  summarizes …" (Drone); same in LCM and Forklift. **Include a one-sentence
  roadmap at the end of our Introduction** — it is an ICINCO convention.

### 5.5 Conclusion shape
- **Consistent 3-move conclusion:** (1) restate what was proposed
  ("We proposed an application of …", "A complete navigation architecture
  was designed …"), (2) restate the key result/finding, (3) one or two
  future-work sentences ("Future work will focus on …"). Conclusions are
  short (one paragraph to half a column). **Match this: recap contribution
  + headline number + journal-direction future work, kept tight.**

### 5.6 Transitions & connectives (observed phrases)
- Result→explanation: "This happens because …", "That leads to the
  conclusion that …", "From an analysis of the data, these clusters
  correspond to …".
- Concession: "However, …" (ubiquitous), "Despite the fact that …",
  "While effective, …", "Although … a main challenge is still unsolved".
- Addition/sequence: "Furthermore, …", "Moreover, …", "Additionally, …",
  "Then, …", "Next, …".
- Method choice justification: "The reason for this is …", "since it has
  demonstrated to …", "due to its simplicity and low computational cost".
- The register is plain, declarative scientific English; sentences are
  moderate length; first person plural ("we propose", "we have used") is
  standard and expected.

---

## 6. Quick checklist for our draft (calibrated to the venue)

- [ ] 5–6 numbered ALL-CAPS sections; Related Work **before** Method.
- [ ] One pipeline figure for the architecture (Figure 1, early in Method).
- [ ] A handful of numbered equations for tokenization/attention/readout;
      define symbols inline (notation table optional, keep small).
- [ ] Hyperparameters inline in a "Training Details" subsection.
- [ ] ≥1 external SOTA baseline per claim; results in tables + curves;
      report mean ± std (we have seeds) — no significance test needed.
- [ ] Ablations labelled as such (K-sweep, dropout-rate) — at/above norm.
- [ ] Results and Discussion merged; no standalone "Limitations" header —
      fold honest limitations into Discussion/Conclusion with the
      "concede → despite this → restate value" shape.
- [ ] Introduction: broad-importance opener → "but" gap → numbered
      Contributions list → one-sentence roadmap.
- [ ] Conclusion: recap + headline number + tight future-work.
- [ ] APA author-year citations throughout (never numeric).

## Caveat

The structural/numeric observations are robust (drawn from full
section-by-section reads of 6 robotics/localization papers); the
"X/6" fractions describe that targeted sample, which deliberately matches
our paper's cohort rather than the whole 100+-paper proceedings, so they
are indicative of the relevant sub-community, not a census of every ICINCO
2024 paper. Direct sentence quotations are reproduced from the proceedings
PDFs via NotebookLM and should be treated as paraphrasable evidence of
register, not as text to reuse.
