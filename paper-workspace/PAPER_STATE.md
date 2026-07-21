# PAPER STATE

> **STATUS (2026-07-21, author-confirmed):** the paper WAS SUBMITTED on
> 2026-06-26, ICINCO 2026 second round, **DOUBLE-BLIND** (`\blindtrue` in
> `paper/main.tex` since 2026-07-03 — this resolves the C1 conflict logged
> below: double-blind is the final ruling; earlier single-blind records are
> void). Notification: 2026-07-31. Camera-ready: 2026-09-08 (Angers,
> 26–28 Oct 2026). NOTE: entries below stop at 2026-06-26; the manuscript
> received further Overleaf edits through 2026-07-03 (`paper/` is the ground
> truth). **Camera-ready critical path:** fold the M1 ablation, M2 mean±std,
> and M5 real-data robustness tables from `revision/PAPER_INSERTS.md` into
> `paper/sections/06_results.tex`; then `\blindfalse`, SCITEPRESS
> declarations + AI-usage disclosure (source: `ai-usage.md`), `main.bbl`.

Track: Regular Paper (ICINCO 2026, second round).
Submitted: 26 June 2026 (double-blind).
Char band: 10,000-50,000 excl. spaces. Current per-section counts (raw .tex incl.
LaTeX markup, from the task wc command): 04_methodology.tex = 11,907 (refined
2026-06-17 from 12,121); 05_experiments.tex = 3,559 nospace / 547 words (shrunk
2026-06-17 from 3,878 / 606, -9.7%); 06_results.tex = 8,483 nospace / 1,215 words
(shrunk 2026-06-17 from 9,239 / 1,346, -9.7%). Earlier ~7,234 figure was a
LaTeX-stripped proxy for the Methodology prose only — the 11,907 is the raw-source
count. Full-paper count still pending a real compile (no detex/latexmk on this
machine).
Working title (from scope.md): Async-Robust Multi-Modal Indoor
Localization via a Continuous-Time Set-Transformer.

## Section structure (venue-calibrated from style-icinco.md)
5-6 numbered ALL-CAPS sections. Related Work BEFORE Methodology.
Results and Discussion MERGED. NO standalone Limitations section -
limitations are hedged inline or folded into Conclusion with the
"concede -> despite this -> restate value" shape.

### Methodology subsection order (2026-06-01 restructure)
Director-validated. NOW APPLIED to BOTH draft.tex AND the MAIN paper file
paper/sections/03_methodology.tex (which main.tex \input's):
3.1 Overview (global low-symbol map; renamed from "Pipeline Overview";
   CONCEPT-ONLY -- no token equation, no $\phi$ math; carries Figure 1)
-> 3.2 Problem Formulation (pure formalism)
-> 3.3 Per-Modality Encoders (WiFi-Net, IMUCNN; now also carries the
   universal-token equation eq:token + continuous-time $\phi$ math eq:time,
   relocated out of Overview as the token-construction lead-in)
-> 3.4 Set-Transformer Fusion and Position Readout (eq:attn, eq:readout)
-> 3.5 Training and Async-Robustness.
(Overview <-> Problem Formulation swapped vs the prior draft. main.tex
sections/03_methodology.tex SYNCED to this order on 2026-06-01 as full prose;
Training subsection's hardcoded "Section~3.1" cross-ref re-pointed to
"Section~3.2" to match the new order. Equation document order:
eq:token, eq:time, eq:attn, eq:readout. All \label/\ref verified by reading;
only cross-ref into the section is \ref{fig:pipeline}. Figure 1 +
scope-warning comment + [[VERIFY: figure-1 asset]] left untouched.)

### Methodology rigor + symbolization (2026-06-01 director-approved pass)
TAKE 1 added to §3.4: Algorithm~1 (\label{alg:forward}, algorithm2e) for the
end-to-end forward pass (token assembly -> never-masked CLS -> L pre-norm
self-attn layers w/ padding mask -> single PositionQuery cross-attn -> 2-layer
MLP), single-PositionQuery only. New equations: eq:wifi_assign +
eq:wifi_token (WiFi-Net soft assignment + anchor-weighted token w/ residual
MLP), eq:imu (LN(W*GAP(ConvStack))); the former eq:attn was expanded into
eq:layer (two pre-norm residual sub-layers MHSA+FFN) and eq:mhsa (per-head
softmax + concat through W_O). NB: the eq:attn label was retired — no \ref by
number existed (verified), so nothing breaks. New equation document order:
eq:token, eq:time (3.3) ; eq:wifi_assign, eq:wifi_token (WiFi-Net) ; eq:imu
(IMUCNN) ; eq:layer, eq:mhsa, eq:readout (3.4).
TAKE 2 (symbolization): collision-free symbol set used in §3.3-§3.5 prose
instead of inline numbers — D, M, K (instants), N_a (WiFi anchors; RENAMED
from lowercase k to disambiguate from K), T (IMU window), F (IMU features),
(C_1,C_2,C_3) (conv channels), L (self-attn layers; chosen to avoid clashing
with lengths), H (heads), r (FFN ratio, width rD), n (sinusoids, periods
[tau_min,tau_max]). NEW Table tab:arch (architecture config, end of §3.4)
lists every symbol+value (D=128, M=2, N_a=64, T=32, F=9, (32,64,128), L=6,
H=4, r=4, n=32, periods [0.05,120] s, ~0.05 M params/encoder); prose cites it.
TRAINING CONFIG MOVED out of §3.5 into a NEW Table tab:train placed in
main.tex INSIDE \subsection{Exp setup} (the ONLY main.tex edit this pass).
§3.5 now reads symbolically (AdamW/OneCycleLR/Huber/grad-clip/seed
methodological; dropout mechanism kept with rates p_m, p_i + rescue pass) and
\ref's tab:train. The §3.5 \textcolor{red}{...exp setup} marker REMOVED (the
move fulfills it); all other red notes (§3.1 overview, Figure 1 caption,
main.tex title block) preserved. All numeric values unchanged in meaning
(presentation only); all % src comments preserved (36 in 03_methodology.tex +
relocated training src block in main.tex); all \label/\ref resolve. Figure 1
block untouched. NOT COMPILED — no LaTeX engine; verified by reading.

## Section status
- [x] Abstract                          (2026-06-25: FULL CLARITY REWRITE per external co-author verdict "confusing, too much info without explanation, long sentences"; director wants understandable/polished/catchy. 198 words (<=200, confirmed wc -w + python split). New venue opener (broad-importance hook "Knowing where a device is inside a building..." -> WiFi+RSSI/inertial recipe -> async/missing/stale gap -> resample+branch status quo -> "We instead present..."). Long token sentence SPLIT into 3 short ones; jargon glossed in plain words -- "permutation-invariant self-attention block" -> "a single attention block ... independent of token order" (term REMOVED), "modality- and instant-dropout" -> "we randomly hide whole sensors and single time steps during training" (coined names REMOVED); "no differential-equation solver" kept as brief contrast. Model named "continuous-time transformer" (NOT set-transformer), mechanism = "unordered set". "robot" NOT added (rejected the reviewer's pencilled "robot localization" first line). (RSSI) acronym def kept on first use; NO number/cite added. RESULTS CLAIMS UNCHANGED IN MEANING (supervisor-approved): (a) improves over open WiFi fingerprinting reference under cross-session drift; (b) stays competitive with learned-fusion + dead-reckoning references; (c) degrades smoothly under missing/stale inputs; validation scope (each encoder on its benchmark, fused model on sim+real incl. cross-session unseen-days) preserved. [[VERIFY-TIGER]] STILL OPEN: the approved (b)/(a) wording may over-claim vs external reviewer Min1 + revision M2 (cross-session test margin within seed noise; k-NN weak) -- re-calibrate when the revision/"tiger" results are folded in. This pass preserved the claims as instructed; it did not re-open the calibration.)
- [x] 1. Introduction                    (2026-06-25: FUNNEL REWRITE de-centering robotics -- broad indoor-localization opener (pedestrian/robot/asset, robot NOT lead) -> WiFi+IMU narrowing (IMU acronym defined here) -> device-agnostic continuous-time TRANSFORMER (prose drops "set-transformer"; platform-agnostic = phone/robot/wearable, still exactly 2 modalities) -> unchanged roadmap. Figure 1 float + fig:pipeline + 6 section refs unmoved/resolve; no cite added. PRIOR: 2026-06-23 supervisor rewordings accepted+de-colored; global-reference-system framing, consolidated WiFi/IMU para, "In this paper, we address..." + clean roadmap; Figure 1 float unmoved)
- [x] 2. Related Work                    (2026-06-23: supervisor first-three-paragraph rewordings accepted+de-colored; British spelling kept; tab:properties + cites untouched)
- [x] 3. Methodology                     (Day 1 DRAFTED 2026-06-01)
- [x] 3->4 split                          (2026-06-23: supervisor asked §3 to hold ONLY the general problem and move "from eq.4 to end" into §4. Done: §3 now = tuple + eq:spaces + eq:obsset-gen (O') + eq:objective-gen (general f(O',t_q)) + a generic cross-session/missing-stale conditions paragraph (no eqs). MOVED into §4 Overview: eq:obsset (the Delta-t set O), eq:perm, eq:decomp. DROPPED eq:objective (simplified, redundant) and eq:fit (cross-session ERM, already concrete in Experiments); re-pointed §4 loss ref eq:objective -> eq:objective-gen. Verified: no dup labels, no dangling refs (eq:objective/eq:fit gone everywhere), envs balanced, all 19 cites resolve. §3=3 eqs, §4=14 eqs.)
- [~] 4. Experiments (incl. results+discussion)  (Day 2 2026-06-01: Experimental Setup DONE as prose in main.tex; Per-Leg Encoder Validation + End-to-End Fusion DONE as prose + tab:perleg/tab:fusion in sections/04_experiments.tex. STILL TODO: Ablations & robustness, Discussion & limitations; fusion figure pending director; Webots val/test reconciliation pending director)
- [ ] 5. Conclusion (incl. honest-limits + future work)  (Day 5)

## 7-day plan (venue-adjusted)
- Day 1: Methodology draft. Repo survey + token construction + attention block + readout + training details (hyperparams inline).
- Day 2: Experiments. Setup, baselines (>= 1 per claim), tables+plots, ablations as labelled subsection, results-and-discussion merged.
- Day 3: Related Work draft (method-family grouping per style-icinco.md). Zotero filter + repo cross-check + deep-search prompts.
- Day 4: Introduction (broad-importance opener -> but-gap -> numbered contributions -> roadmap). Ingest deep-search results into refs.bib + finish Related Work.
- Day 5: Conclusion (recap + headline number + future-work). Fold honest limitations using "concede -> despite this -> restate value" shape. Abstract last.
- Day 6: One pipeline figure + figure/table-reference sweep + char-count pass + double-blind sweep.
- Day 7: Hostile-reviewer self-pass against ICINCO reviewer criteria. Resolve [[VERIFY]] markers. Final compile. Style-anti-ai self-check on full draft.

## Style references (read every session)
- paper-workspace/scope.md           (what to publish / hold back)
- paper-workspace/conference-rules.md (ICINCO hard rules)
- paper-workspace/style-icinco.md    (venue register: structure, voice, hedging)
- paper-workspace/style-anti-ai.md   (universal anti-LLM-tone rules)
- paper-workspace/icinco-2024-relevant.md (citation pool for ICINCO 2024)

## Open questions / [[VERIFY]] / [[DECIDE]]
- [[VERIFY: full-model param count]] 04_methodology.tex closing says "about 1.5 M
  parameters"; the measured full 2-mod (K=4, depth=6, heads=4, query) model is
  1,380,035 (~1.38 M, RESULT_06 + "Resolved facts" below). "~1.5 M" rounds up
  generously. Kept verbatim during the 2026-06-17 refine pass per that task's hard
  constraint; consider tightening to "~1.4 M" in a future authorized edit.
- [[VERIFY: K]] §3.5 reports K=4 (per scope.md sec 2 + approved plan), but the
  committed configs/stage_c/fusion.yaml:71 sets temporal.n_instants=8. Resolve
  before Experiments: either re-run/confirm at K=4 or change the reported value.
- [[VERIFY: dropout vs RESULT_05]] The plan asked to cross-check
  modality_dropout=0.4 / instant_dropout=0.45 against RESULT_05. RESULT_05 is
  the RoNIN-C2 result and does NOT carry these values. Cross-checked instead
  against RESULT_06 (provided), which confirms "0.4 / 0.45 (config defaults
  preserve the run-1 audit fix)". fusion.yaml:59-60 is the cited ground truth.
- [[VERIFY: WiFi-Net param budget]] scope.md sec 3 + plan claim ~0.075 M params
  for WiFi-Net; measured count is 49,217 (n_aps=117) / 49,921 (n_aps=128 PCA)
  = ~0.05 M. Drafted with ~0.05 M (measured). scope.md figure looks stale.

## Resolved facts (code-grounded, used in §3)
- WiFi-Net anchors: k=64 (src/pipeline/encoders/wifi.py:48). ~0.05 M params.
- IMUCNN: 9-channel body-frame input, 32-step window, channels 32->64->128
  (imu.py:55-62; dataset.py:27-31; simulation_2mod.yaml:23). ~0.05 M params.
  NOTE scope.md says "6-channel" — code says 9 (body frame); world-frame M4
  variant is 5. Conference 2-mod config uses the body frame (9). scope.md stale.
- Transformer trunk: depth=6, heads=4, ff_mult=4, pre-norm, GELU, embed=128,
  readout=query (fusion.yaml:24-45). NOTE transformer.py *defaults* are depth=4
  / heads=8, but the config (ground truth) overrides to 6 / 4.
- Full model (2-mod, K=4, depth=6, heads=4, query) = 1,380,035 params (~1.38 M),
  matching RESULT_06.

## Blockers
- No LaTeX toolchain (latexmk/pdflatex) and no detex on this machine. Could not
  compile or run the official char-count command; used a Python LaTeX-strip
  proxy instead. Compile + official char count deferred to a machine with the
  toolchain (or Overleaf).

## Double-blind / scope risks noticed
- RESOLVED (title): main.tex/draft.tex now carry the director-approved title
  "Continuous-Time Set-Transformers for Asynchronous WiFi--IMU Indoor
  Localization" — no project-name leak remains.
- RESOLVED (blind status): director CONFIRMED ICINCO/INSTICC is single-blind
  (authors visible), so \blindfalse stays. Contract + paper-format/paper-content
  skills assume double-blind; this is a director-confirmed venue policy, not a
  tentative deviation. \ifblind toggle kept so the build can still flip to
  anonymized in one line if ever needed.
- SCOPE LEAK (figure) -- now OWNED BY DIRECTOR: figures/figure_1.jpg is the
  OUT-OF-SCOPE decomposed dual-head readout (Q_absolute + Q_motion + gate g)
  contradicting §3.4's single PositionQuery; also implies K=8 not K=4. The
  director is producing the in-scope single-PositionQuery diagram separately
  with a design tool; figure_1.jpg, its \includegraphics, caption, scope-warning
  comment, and [[VERIFY: figure-1 asset]] marker are left untouched pending the
  director's replacement asset. STILL PENDING the director's asset.

## ORCID / template-asset status (2026-06-01 director-decisions pass)
- RESOLVED (ORCID): \orcidAuthor{<id>} now WIRED immediately after each author
  name in main.tex/draft.tex per the SCITEPRESS template pattern (Example.tex):
  Mohamed Bachar 0009-0001-6222-3096; Ilyass Abouelaziz 0000-0001-7208-2509;
  Yuehua Ding 0000-0002-6196-3068. The OFFICIAL orcid.eps from the SCITEPRESS
  author kit (paper-workspace/SCITEPRESS_Conference_Latex/orcid.eps) was copied
  to paper/orcid.eps (no icon fabricated), so \orcidAuthor (SCITEPRESS.sty:163,
  \epsfig{file=orcid.eps}) now has its asset. Not yet visually confirmed -- no
  LaTeX engine on this machine to compile.
- RESOLVED (author names): director supplied full first names. Author block now
  reads "Mohamed Bachar, Ilyass Abouelaziz and Yuehua Ding"
  (was "Mohamed Bachar, I.~Abouelaziz and Y.~Ding"). Stale "pending first names"
  comment lines removed from both .tex files.

## Last session
2026-06-26 (ENGLISH-REGISTER / "oral-English" cleanup, GREEN-marked, paper-writer
persona): senior author Yuehua flagged "oral/spoken English" in the BODY (red comment
in the abstract block lists his examples: so->Consequently/Thus, very different->different,
line up in time->align in time, free of resampling->without resampling, go missing->are
missing, far smaller->significantly smaller). Director asked ME to apply the fixes and
mark each change in GREEN (\textcolor{green}{...}) so Yuehua can see what changed; RED is
his colour and was left untouched. SCOPE: body only -- the 6 section files
(01_introduction..06_results) + the \section{Conclusion} prose in main.tex; abstract,
title, keywords, preamble NOT touched (those are Yuehua's, already edited). 13 GREEN fixes
total: 01_introduction 5 (barely penetrate->scarcely penetrate; "so an indoor device has
to"->"Consequently, an indoor device must" [sentence split]; far higher->significantly
higher; "may drop out"->"may become unavailable"; stays accurate->remains accurate);
02_related_work 1 (typo+awkward "was intuitivly presented in Time2Vec and mTAN"->"builds on
Time2Vec and mTAN"); 03_problem_statement 1 (far higher->significantly higher);
06_results 3 ("does so on a far smaller budget"->"which it incurs on a significantly
smaller budget"; holds position->maintains position; stays above->remains above); main.tex
Conclusion 3 (stays competitive->remains competitive; far smaller->significantly smaller;
much larger->significantly larger). 04_methodology + 05_experiments: NO extreme informal
expressions found -- all their "so" uses are tightly-bound mid-sentence result subordinators
(standard scientific register), left intentionally per the "don't robotically convert every
'so'" guidance. Tightly-bound "so" clauses in intro/results/problem also left. Register
intentionally favours formal academic English this pass (Yuehua's explicit request), kept
natural not robotic. INTEGRITY: NO number/equation/label/\cite/\ref/table-cell/figure/caption
metric changed -- register/wording only (grep-confirmed the flagged extreme phrases are gone
from all section files; the only residual "far smaller"/"line up" match is INSIDE Yuehua's
red comment in the abstract, correctly preserved). All existing \textcolor{red}{...} comments
and \sout{...} preserved exactly (grep-listed: figure-1 caption red, 2x figure caption "le
text est tres petit" red, main.tex title/affiliation/keywords/abstract red + 2x \sout in
keywords). 13 \textcolor{green}{ opens, all closed (2 span a line break: "significantly\n
smaller}" and "significantly\nlarger}" in main.tex Conclusion -- verified by reading). Only
GREEN added; no RED added/removed/recoloured. Edited ONLY the 6 section files + the
Conclusion block of main.tex (+ this file, ai-usage.md). Could NOT compile -- no LaTeX engine
on this machine; verified by reading + grep.

## Session before this
2026-06-25 (ABSTRACT clarity rewrite, paper-writer persona): external co-author (not
deeply involved) reviewed the abstract -- verdict "confusing, too much information
without explanation, long sentences"; director wants understandable + polished +
genuinely catchy (abstract-screening grade), academic register, no hype. Rewrote ONLY
the paper/main.tex `\abstract{}` block. FIXES: (1) short sentences -- the long
"We present a continuous-time set-transformer that represents every observation as one
token ... without resampling or a differential-equation solver" run-on split into 3
short sentences (set framing / per-token contents / single-block fusion); one idea per
sentence throughout. (2) jargon glossed in plain words -- "permutation-invariant
self-attention block" -> "A single attention block fuses across sensors and time,
independent of token order" (bare term "permutation-invariant" REMOVED from the
abstract); "modality- and instant-dropout" -> "During training we randomly hide whole
sensors and single time steps, so the model learns to cope when inputs go missing or
stale" (coined compound names REMOVED); "differential-equation solver" kept as a brief
contrast ("with no resampling and no differential-equation solver"). (3) catchy venue
opener (broad-importance -> but-gap -> what-we-do) -- hook "Knowing where a device is
inside a building underpins navigation and many location-aware services." then the
WiFi+RSSI/inertial recipe, the async/missing/stale tension, the resample-and-branch
status quo, then "We instead present...". (4) DEVICE-AGNOSTIC/general framing kept;
"robot" deliberately NOT added (the external reviewer's pencilled "robot localization"
first-line edit was NOT taken; grep-confirmed no "robot"). (5) model named
"continuous-time transformer" (NOT "set-transformer", matching the rewritten Intro);
mechanism still "treats the observations as an unordered set". RESULTS CLAIMS UNCHANGED
IN MEANING (supervisor-approved): improves over an open WiFi fingerprinting reference
under cross-session drift; stays competitive with learned-fusion + dead-reckoning
references; degrades smoothly under missing/stale inputs; validation scope (each encoder
on its benchmark; fused model on sim+real incl. a cross-session collection tested on
unseen days) preserved. NO number/percentage/citation added; "(RSSI)" acronym definition
kept on first use. Anti-AI self-check: zero banned vocab (grep); tricolons = 2 across the
whole abstract but both load-bearing/irreducible (the token's three fields; the three
approved result claims) with the decorative opener tricolon CUT; one "We"-starter only;
varied openings; no "In this paper, we propose..." opener; no method-hype adjective.
WORD COUNT 198 (<=200), confirmed two ways (wc -w and python str.split()). INTEGRITY:
`\abstract{}` brace-balanced, edit confined to the abstract body; preamble / `\title` /
`\ifblind` / author block / `\keywords` / `\input` untouched; zero `\color`/`\textcolor`.
Edited ONLY paper/main.tex (+ this file, ai-usage.md). Could NOT compile -- no LaTeX
engine on this machine; verified by reading + grep + word-count scripts. Next: figure_1
asset still pending director; [[VERIFY-TIGER]] results-claim calibration still pending
(preserved as instructed, not re-opened this pass).

## Session before this
2026-06-25 (Introduction FUNNEL rewrite, de-center robotics, paper-writer persona):
director intent -- the paper should read as about indoor localization in general
(method is device-agnostic: WiFi scans + IMU windows, which phones/robots/wearables
all carry; real-data validation is pedestrian/smartphone, not robot), matching a
broadened title direction. Rewrote paper/sections/01_introduction.tex as a 4-paragraph
broad->narrow funnel, same overall length. PARA 1 (broad): replaced the
"Mobile robots increasingly operate indoors..." opener entirely with a general
indoor-localization opener -- "Many agents that move through buildings must know
where they are." -- and three varied use cases with robotics NOT leading (pedestrian
smartphone nav; mobile robot warehouse logistics / facility inspection; tracked
asset); GPS-fails-indoors + no-global-reference + noisy + environment-shifts +
hardware/site-survey-bounded difficulty; ends "reliable indoor self-localization
remains an open problem." PARA 2 (narrow to WiFi+IMU): "Two sensors are carried by
nearly all of these devices: a WiFi radio and an inertial measurement unit (IMU)."
-- IMU acronym DEFINED here on first use; WiFi = sparse absolute reference; IMU =
dense relative motion at higher rate; complementary/heterogeneous/asynchronous, no
common clock, drop-out/stale; "Fusing these imperfect, asynchronous signals into one
position estimate is the core problem this work targets." PARA 3 (method,
device-agnostic): "In this paper, we propose a continuous-time TRANSFORMER
(Figure~\ref{fig:pipeline}) that fuses asynchronous WiFi and IMU observations into a
single (x,y) estimate ... missing or stale." + platform-agnostic claim ("not tied to
a particular platform ... needs only WiFi scans and inertial windows, the same inputs
a phone, a robot, or a wearable already produces ... any device that carries these two
sensors") + "We validate it on real cross-session data." PARA 4 (roadmap): unchanged.
HONESTY GUARDRAILS HELD: robotics MENTIONED (use case #2 in para1, "a robot" in para3)
but NOT central; method fuses EXACTLY two modalities (WiFi+IMU) -- "device-agnostic"
is platform-only (phone/robot/wearable), NO general-multimodal-beyond-WiFi+IMU implied;
prose uses "continuous-time transformer" (NOT "set-transformer") per the broadened title
direction; the Figure 1 caption (which still says "set-transformer") left untouched per
"keep the figure float exactly as is". NO fabricated citation -- intro is cite-free
before and after (grep-confirmed: 0 \cite). Anti-AI self-check: zero banned vocab (grep),
tricolons = exactly one in para1 (pedestrian/robot/asset) + one in para3
(phone/robot/wearable) = <=1 per paragraph, no recap-after-figure (para2 opens on "Two
sensors are carried...", not on the figure), one "We"-starter only (para3 closing), one
idea per sentence, varied openings. INTEGRITY (reading + grep; no LaTeX engine):
braces 16/16; \begin{figure}/\end{figure} balanced + fig:pipeline float UNMOVED; all 7
\ref keys present and resolve (fig:pipeline + sec:related/problem/method/experiments/
results/conclusion -- all six section \label's confirmed to exist via repo grep); zero
\color/\textcolor. Edited ONLY paper/sections/01_introduction.tex (+ this file,
ai-usage.md). Could NOT compile -- no LaTeX engine on this machine; verified by reading +
grep. Next: figure_1 asset still pending director; results-number revision still pending.

## Session before this
2026-06-23 (Pass B: hyphen de-compounding + dense-sentence lightening, paper-writer
persona): director ask -- "simplify when possible but stay fancy in the scientific
language." Pass B of a two-pass job (Pass A did Methodology + Results). TWO goals over
paper/main.tex (abstract + Conclusion), 01_introduction.tex, 02_related_work.tex,
03_problem_statement.tex, 05_experiments.tex: (1) reduce hyphenated "x-y" compounds,
(2) lighten the densest sentences while keeping a polished scientific register. No
number, percentage, metric value, dataset count, hyperparameter, or claim changed; no
`\label`/`\cite`/`\ref`/`\eqref`/equation/table/figure/`\includegraphics` added, removed,
or renamed; `\title` untouched; no `\color`/`\textcolor`.
HYPHEN REWORDINGS, main.tex (abstract): "received-signal-strength-indicator" ->
"received signal strength indicator"; "WiFi-fingerprinting" -> "WiFi fingerprinting".
main.tex (Conclusion): "modality-specific benchmarks" -> "single-modality benchmarks";
"sub-metre accuracy" -> "accuracy below one metre". KEPT (claim wording / standard):
learned-fusion, dead-reckoning, cross-session, cross-subject, continuous-time,
set-transformer, modality-/instant-dropout, robustness-aware, WiFi--IMU,
permutation-invariant, self-attention, session-invariant, over-smoothing.
HYPHEN REWORDINGS, 01_introduction.tex: none needed (all hyphens are keep-terms:
search-and-rescue, error-prone, self-localization, continuous-time, set-transformer,
self-attention, cross-session). 02_related_work.tex: "modality-specific branches" ->
"a separate branch per modality"; "missing-sensor robustness" -> "robustness to a
missing sensor"; "cross-session generalisation" -> "generalisation across sessions";
"Whole-modality dropout" -> "Dropping an entire modality at training time";
"missing-channel robustness" -> "robustness to an absent channel"; "per-modality
encoders" -> "an encoder per modality"; "per-instant dropout" -> "instant dropout"
(x2); "cross-session robustness" -> "robustness across sessions"; "whole-modality
dropout" (closing) -> "the earlier whole-modality scheme"; "cross-session evaluation"
-> "evaluation across sessions". KEPT: single-modality, multi-rate, ten-day, hand-set,
filter-based, cross-user, permutation-invariant, real-valued, continuous-time,
cross-modal, cross-time, self-attention, camera-only, ODE-in-attention.
03_problem_statement.tex: "multi-modal" -> "multimodal"; "held-out test trajectory" ->
"test trajectory ... held out from training"; "per-observation encoding" -> "an
encoding of each observation"; "modality-specific encoders" -> "one encoder per
modality". KEPT: Self-localization, non-positive, cross-session (protocol), set-level,
permutation-equivariant, self-attention, position-query.
05_experiments.tex: "sub-metre accuracy" -> "accuracy below one metre";
"real-world test" -> reordered to "real cross-session test" (real-world hyphen
dropped); "WiFi-fingerprinting benchmark" -> "WiFi fingerprinting benchmark";
"single-encoder" (x2) -> "encoder-only"; "learned-inertial regressor" -> "learned
regressor over inertial windows"; "dead-reckoning" (noun mod) -> "dead reckoning";
"cross-session comparison" -> "comparison across sessions"; "set-transformer" (prose)
-> "set transformer". KEPT: co-recorded, within-session, $k$-nearest-neighbour,
distance-weighted, end-to-end, IMU-only, ground-truth, four-layer, learned-fusion,
PDR-from-start, root-mean-square, Umeyama-aligned, rigid-body. TABLE CELLS LEFT
BYTE-IDENTICAL ("Real (cross-session)", "Real (held-out paths)" in tab:datasets
untouched).
DENSE SENTENCES LIGHTENED: (01_intro) the "error-prone ... which bounds accuracy
together with the hardware and the site survey that calibrates it" trailing clause
recast into a separate sentence ("Accuracy is further bounded by the sensing hardware
and by the site survey that calibrates it."); the "challenge we address" / "address
this challenge" repetition varied (first instance -> "the core problem this work
targets", keeping the supervisor-approved "In this paper, we address this challenge by
proposing" signal phrase). (02_related) the WIO-EKF/deep-variants multi-clause sentence
split at the WIO-EKF branch; the three-clause "Whole-modality dropout ...; ...; and
SeFT and STraTS ..." semicolon chain broken into three sentences; the 40+-word "Beyond
that domain shift, the proposed method adds ..." sentence split into a lead + two short
sentences; the closing colon+semicolon chain unstacked into four sentences. (03_problem)
the long "Referencing every observation ... drops $t_q$ ... and simplifies the set to"
sentence split (the $\Delta t_i$ definition + its sign condition is now its own
sentence; "This drops $t_q$..." starts a new one); "asynchronous" de-duplicated in the
first two sentences (kept once). (05_experiments) light touch only -- the reference-
method bullets de-jargoned ("learned-inertial regressor" -> "learned regressor over
inertial windows") with no number touched.
Anti-AI self-check: zero banned vocab (grep), <=1 tricolon/paragraph, no recap-after-
figure, varied openings, one idea per sentence.
INTEGRITY (reading + grep; no LaTeX engine on this machine): brace/begin-end balance
unchanged vs baseline (main 47/47 1/1; intro 16/16 1/1; related 24/24 2/2; problem
116/116 13/13; experiments 54/54 5/5). Every `\label`/`\ref`/`\eqref`/`\cite`/
`\includegraphics` key identical (verified via ripgrep enumeration: related = 13 cite
groups + tab:properties; intro = fig:pipeline + 6 section refs; problem = 9 labels + 3
eqref + 3 ref; experiments = sec:experiments + tab:datasets + 8 cite + eq:mae/eq:ate +
sec:method; main = sec:conclusion + bibliography). All 05_experiments protected numbers
identical count; tab:datasets tabular block BYTE-IDENTICAL (diff clean). Zero
`\color`/`\textcolor`. `\title` unchanged. Abstract = 189 words (<=200; was 200 before
the two abstract de-hyphenations split compound words into separate tokens). Abstract +
Conclusion CLAIMS unchanged in meaning -- named methods (WiFi fingerprinting reference,
learned-fusion, dead-reckoning, open WiFi reference) and comparative verbs ("improves
over", "remains competitive with", "improves on", "stays competitive", "degrades
smoothly"/"degrading gracefully") intact; no claim added/removed/softened. Edited ONLY
the five section/main files (+ this file, ai-usage.md). Could NOT compile -- no LaTeX
engine on this machine; verified by reading + grep. Next: figure_1 asset still pending
director; results-number revision still pending (untouched).

## Pass A (same job, earlier)
2026-06-23 (hyphen de-compounding + dense-sentence lightening, paper-writer persona):
director ask -- "simplify when possible but stay fancy in the scientific language."
TWO goals over paper/sections/04_methodology.tex and paper/sections/06_results.tex ONLY:
(1) reduce hyphenated "x-y" compounds, (2) lighten the densest sentences while keeping a
polished scientific register. No number, percentage, metric value, or claim changed; no
`\label`/`\cite`/`\ref`/`\eqref`/equation/table/figure/`\includegraphics` added, removed,
or renamed; `\title` untouched; no `\color`/`\textcolor`.
HYPHEN REWORDINGS, 04_methodology.tex: `set-transformer` -> "set transformer" (prose; the
\subsection{Set-Transformer Fusion...} heading LEFT as-is per task allowance); subsection
heading "Universal Tokenization and Modality-Specific Encoders" -> "...and Encoders for
Each Modality"; "no modality-specific branches" -> "no branch per modality"; "a
modality-specific encoder realizes $g$" -> "An encoder for each modality realizes $g$";
"the modality-specific encoder output" -> "the output of the encoder for that modality";
"without modality-specific wiring" -> "without wiring specific to its modality";
"place-discriminative distribution" -> "distribution that discriminates place";
"reference-point matrix" -> "matrix of reference points"; "the assignment-weighted
aggregation of the reference-point embeddings ..., refined by a residual layer-normalized
MLP" -> "The weights $\mathbf{w}$ aggregate the embeddings of the reference points ... into
a pooled vector, which a residual MLP with layer normalization then refines into the token"
(also unstacked); fig:wifinet caption "the learned reference-point embeddings" -> "the
learned embeddings of those reference points"; "dense short-term motion ... the
relative-motion complement" -> "dense motion over a short temporal window, supplying the
relative motion that complements WiFi's absolute reference"; "capture short-term dead
reckoning" -> "capture dead reckoning over short intervals". KEPT (standard ML): self-
attention, feed-forward, multi-head, pre-norm, real-valued, permutation-equivariant/
invariant, cross-modal, cross-time, scaled-dot-product, layer-normalized (now reworded
away once), continuous-time, ground-truth(mod), NaN-safe, WiFi-Net, IMU-CNN. self-attention
kept at its 4 load-bearing uses (first use, property statement, MHSA acronym def, MHSA
expansion) -- not padded repetition.
HYPHEN REWORDINGS, 06_results.tex: subsection + caption "Modality-Specific Encoder
Evaluation" -> "Encoder Evaluation by Modality"; "fingerprint-to-position mapping" ->
"mapping from fingerprint to position"; "$32$ held-out sequences" -> "$32$ sequences held
out from training"; "an in-domain-competitive encoder paying a cross-subject generalization
cost at a far smaller budget" -> "IMUCNN is competitive within its own domain but pays a
cross-subject generalization cost, and it does so on a far smaller budget" (split + de-
jargoned); "one held-out sequence" -> "one such sequence"; "five best held-out sequences"
-> "five best sequences"; "less one-sided per-sequence picture" -> "more even picture
sequence by sequence"; "the set-transformer fuses" -> "the set transformer fuses"; "cross-
session real-world collection" -> "real collection across sessions"; "the two-modality
model" -> "the model fusing the two modalities"; "This sub-metre figure" -> "This figure,
below one metre"; "real-world numbers" -> "real numbers"; "real, cross-session evaluation"
-> "real evaluation across sessions"; "On the held-out test session" -> "On the test
session, held out for evaluation"; "Out-performing wlan_localization" -> "Beating wlan_
localization"; "full per-sample test error distribution" -> "full distribution of test
error per sample"; "per-path error against PDR-from-start" -> "error for each path against
PDR-from-start"; tab:fusion caption "two-modality reference; ... is cross-session" ->
"reference fusing two modalities; ... runs across sessions"; subfig captions "Per-sample
error distribution" -> "Error distribution per sample", "Per-path test MAE ... A per-path
illustration" -> "Test MAE for each of the two paths ... An illustration", parent caption
"cross-session fusion ... per-sample ... per-path MAE" -> "fusion across sessions ... per
sample ... MAE per path"; webots caption "sub-metre tracking" -> "tracking below one
metre"; "a within-distribution held-out-path split ... no session-drift advantage" -> "a
split that holds out paths within the same distribution ... gains nothing from session
drift"; "the cross-session MSILN test" -> "the MSILN test across sessions"; iwfine caption
"per-path test MAE" -> "test MAE for each of four paths"; "reaching the IMU-only level" ->
"reaching the level of IMU alone". KEPT: cross-subject (not a target), distance-weighted,
WiFi-dense, real-time, WiFi--IMU, end-to-end, PDR-from-start (proper name), WiFi-Net,
IMUCNN; one fig:ronin_traj caption "held-out sequence" left as a self-contained split-name.
DENSE SENTENCES LIGHTENED (04): the permutation-equivariant comma-splice "Self-attention is
permutation-equivariant by construction, the aggregation the encoding ... is built for." ->
"Self-attention is permutation-equivariant by construction. This is exactly the aggregation
that the token encoding of Section~\ref{subsec:tok} is built for." (split); the $\phi$
sinusoid run-on -> broken into two sentences at the projection; the readout closing
semicolon splice -> "...as Equation~\eqref{eq:perm} requires. Retaining CLS in the key set
keeps this last attention NaN-safe."; the loss-pointer stacked subject -> "The optimizer
and the threshold $\delta$ are given in Section~\ref{sec:experiments}, along with the
modality and instant dropout that confer robustness...". DENSE (06): the in-domain-
competitive jargon stack split + de-jargoned (above); the MSILN "the fusion error falls,
because fusing ... boundary" run-on split into two sentences; the within-distribution split
sentence unstacked. Anti-AI self-check: zero banned vocab (grep), <=1 tricolon/paragraph,
no recap-after-figure, varied openings, one idea per sentence.
INTEGRITY: all 30 protected 06 numbers identical count before/after
(8.69/15.17/42.7/5.14/9.72/89.2/0.05/4.6/95/0.61/10.90/28.31/12.49/52.69/16.67/16.88/21.26/
65.15/6.37/2.31/1.06/0.84/48.2/7.62/4.77/0.146/786/28/0.86/3.50). 04 braces 298->299/299
(balanced; one added math-free balanced pair), begin/end 17/17; 06 braces 92/92, begin/end
12/12. Zero `\color` in either. Every `\label`/`\ref`/`\eqref`/`\cite`/`\includegraphics`/
`\subref` key identical to before (grep-listed: 04 = 42 keys, 06 = 23 keys + 3 subref).
Edited ONLY the two section files (+ this file, ai-usage.md). Could NOT compile -- no LaTeX
engine on this machine; verified by reading + grep. Next: Conclusion polish if directed;
figure_1 asset still pending director; results-number revision still pending (untouched).

## Session before this
2026-06-23 (project supervisor style onto Experiments + Results, paper-writer persona):
LIGHT consistency + clarity + grammar pass over sections/05_experiments.tex and
sections/06_results.tex, projecting the same editing style the supervisor applied to
the front matter (abstract->methodology): observe acronyms on first use, tie into the
problem-statement notation where natural, clean phrasing. He did NOT touch these two
sections; this pass brings them into line. No number, percentage, table cell, claim,
`\label`, `\cite`, equation, table, figure, or `\includegraphics` changed; no
`\color`/`\textcolor` introduced. FOUR small edits total:
- EXPERIMENTS (terminology): observed the `$k$-NN` shorthand at its definition --
  "$k$-nearest-neighbour ($k$-NN) RSSI fingerprinting" in the wlan_localization
  reference-method bullet -- so the `$k$-NN` used later in Results is now introduced
  on first use, matching the supervisor's acronym style. (RSSI and IMU were already
  consistent throughout both sections; no straggler "RSS" exists -- grep-confirmed.
  Metric names "raw ATE"/"MAE"/"Umeyama-aligned ATE" already standardized.)
- RESULTS (grammar): fixed a copular mismatch -- "The gap is an in-domain-competitive
  encoder paying..." -> "The gap reflects an in-domain-competitive encoder paying..."
  (a gap is not an encoder). Every word/number kept.
- RESULTS (notation tie-in, the single light touch): the Robustness opener now ties the
  two ablations to the problem-statement degradation conditions -- "...suppressed or
  aged on demand, realizing the missing ($a_i=0$) and stale (large $|\Delta t_i|$)
  conditions defined in Section~\ref{sec:problem}." Uses the established symbols $a_i$
  and $\Delta t_i$ verbatim; `\ref{sec:problem}` resolves to the Problem Statement label.
- RESULTS (register): "So the LSTM is therefore..." sentence-opener "So" was casual ->
  "The LSTM is therefore strong within its own distribution but not across sessions..."
Experiments needed only the one terminology touch; no notation tie-in forced there
(the $K=4$ in the hyperparameter list already reads in the problem-statement sense).
Anti-AI self-check: zero banned vocab, <=1 tricolon/paragraph, no recap-after-figure,
no header-restate, varied openings. Protected-number grep: all 30 values
(8.69/15.17/42.7/5.14/9.72/89.2/0.05/4.6/95/0.61/10.90/28.31/12.49/52.69/16.67/16.88/
21.26/65.15/6.37/2.31/1.06/0.84/48.2/7.62/4.77/0.146/786/28/0.86/3.50) identical count
before vs after. Env balance unchanged (05_experiments begin/end 5/5, braces 54/54;
06_results begin/end 12/12, braces 91->92/92 = the one added balanced `\ref` pair).
Zero `\color` in either file. All `\label`/`\cite`/`\ref`/`\includegraphics` keys
identical to before (grep-listed). Edited ONLY the two section files (+ this file,
ai-usage.md). Could NOT compile -- no LaTeX engine on this machine; verified by reading
+ grep. Next: Conclusion polish if directed; figure_1 asset still pending director;
separate results-number revision still pending (untouched here, by instruction).

## Earlier session (accept supervisor edits)
2026-06-23 (accept supervisor edits — front matter de-color, paper-writer persona):
The supervisor (co-author Yuehua Ding) reviewed in Overleaf and left `{\color{blue}...}`
spans; his commented copy was extracted at _nav3_extract/. We ACCEPTED his edits =
integrated his rewordings into the working files, DE-COLORED, and rectified his
non-native grammar to clean scientific English. No number, citation, equation,
label, table, or figure float changed; no `\color`/`\textcolor` introduced.
- ABSTRACT (main.tex `\abstract{}`): took ONLY his SETUP-half rewordings (first-use
  "WiFi received-signal-strength-indicator (RSSI) fingerprints"; "sampled
  asynchronously at very different rates"; asynchrony => "so at any instant some
  information is missing or stale"; elapsed time glossed "from acquisition to the
  localization query"; token carries sensor data + its modality + learned elapsed-time
  encoding; "evaluate ... on both simulated and real data"). DID NOT adopt his
  over-claiming results half — the calibrated honest second half is PRESERVED verbatim
  in meaning. Trimmed filler to hold the cap: EXACTLY 200/200 words (whitespace count
  over the brace-matched body). His clumsy phrasings fixed ("resulting that",
  "without resampling nor", "(WiFi or IMU )").
- INTRODUCTION (01_introduction.tex): difficulty framed around the lack of an effective
  global reference system (noisy measurements; ambiguous, complex environment, grammar
  fixed); consolidated WiFi/IMU complementarity paragraph; IMU acronym defined here at
  first use; "In this paper, we address this challenge by proposing ..."; roadmap fixed
  to "The remainder of this paper is organized as follows:". Figure~\ref{fig:pipeline}
  float kept EXACTLY where it was.
- RELATED WORK (02_related_work.tex): his first-three-paragraph rewordings applied
  ("WiFi RSSI fingerprints recover absolute indoor position ... labelled survey";
  "On UJIIndoorLoc, a 1-NN reference method reports a 7.9 m error"; "Learned inertial
  navigation provides dense relative estimates."; "attention entered the field through
  iMoT, which uses accelerometer and gyroscope channels"; "with no absolute reference,
  inertial tracking drifts; combining ... is the remedy."; "The fusion of these two
  heterogeneous streams is well studied, but ..."). His grammar fixed ("can provide a
  type of dense estimates", "the inertial tracking drifts exist"). British spelling
  kept (labelled/generalisation/localisation); NOT switched to US. tab:properties and
  all 13 `\cite` groups untouched; 7.9 m intact.
- PROBLEM STATEMENT (03_problem_statement.tex): LIGHT polish only — de-duplicated the
  IMU acronym (now defined once, in the Intro) and smoothed one transition. NO equation,
  `\label`, or structure changed (eq:spaces, eq:obsset-gen, eq:objective-gen, eq:obsset,
  eq:objective, eq:fit, eq:decomp, eq:perm all verified identical; the general->simplified
  arc intact).
Anti-AI self-check: zero banned vocab (grep), <=1 tricolon/paragraph, varied paragraph
openings, no recap-after-figure, no header-restate. Balance verified by script
(main {47/47, 1/1}; intro {16/16, 1/1}; related {24/24, 2/2}; problem {116/116, 13/13});
no `\color`/`\textcolor`; no duplicate `\label` across the four. Could NOT compile — no
LaTeX engine; verified by reading + grep + word-count. Edited ONLY main.tex,
01_introduction.tex, 02_related_work.tex, 03_problem_statement.tex (+ this file,
ai-usage.md). Note: `\usepackage{xcolor}` in main.tex preamble intentionally kept (it is
a package load, not a color span; supervisor's Overleaf review workflow uses it).
Next: Conclusion polish if directed; figure_1 asset still pending director.

## Earlier sessions
2026-06-18 (reader-aids pass, paper-writer persona): ONE iteration adding THREE
reviewer-requested reader aids. No new scientific number; every addition restates
a fact already in the paper. No equation, existing table/figure/number, abstract,
or `\ifblind`/author block touched. Edited ONLY 02_related_work.tex and
03_problem_statement.tex (+ this file, ai-usage.md).
- ADD 1 (Min2, 02_related_work.tex): new full-width Table tab:properties scoring
  7 methods (iMoT, AFT-VO, ContiFormer, SeFT/STraTS, WIO-EKF, Deep concat fusion,
  Ours) against P1 (continuous-time tokens w/o resampling+ODE), P2 (one perm-inv
  set transformer, cross-modal+cross-time in one self-attn block), P3 (learned
  missing/stale robustness, cross-session-validated). 3-state marks
  \checkmark/$\sim$/-- defined in caption. Lead sentence \ref's it; FOLDED the
  former per-method enumeration prose (iMoT/AFT-VO/SeFT/WIO-EKF subset sentence)
  into the table (no recap-after-table); kept the "Ours completes the combination"
  closing claim. Every row carries a % src to the section's own prose. Marks
  VERIFIED against the section text (WIO-EKF P3=partial per ten-day session gap +
  filter; Deep concat P3=partial per hand-set null vector; ContiFormer P1=partial
  ODE-in-attention, P2=-- single-stream).
- ADD 2 (Min4, 03_problem_statement.tex): new full-width Table tab:notation,
  two Symbol|Meaning pairs per row (18 rows, 36 symbols) covering every symbol the
  task listed across Problem Statement + Methodology. Each row % src'd to its
  defining equation/prose. \ref'd once near the top ("Table~\ref{tab:notation}
  collects the notation."). Forward \eqref's into 04_methodology (eq:token, eq:time,
  eq:wifi_assign, eq:loss) all resolve. Bare $\tau$ (query offset, eq:readout)
  deliberately EXCLUDED to avoid clash with $\boldsymbol{\tau}$ (period vector,
  eq:time); only the period range $[\tau_{\min},\tau_{\max}]$ is listed.
- ADD 3 (Min6, 03_problem_statement.tex): made staleness quantitative at first
  use — "stale once $|\Delta t_i|$ exceeds the modality's nominal update interval,
  about one second for the wifi stream against roughly 30 Hz for the IMU", + a
  sentence that the Section~\ref{sec:results} staleness sweep ages the wifi token
  from fresh to fully stale across the $K$ held instants. No invented seconds-grid
  or slope. % src points to the CLAUDE.md sensor-rate table (wifi ~31 sim steps x
  32 ms ~ 1 s; IMU ~30 Hz).
Anti-AI self-check: zero banned vocab, <=1 tricolon/paragraph, no
recap-after-table, one idea per sentence. Brace/environment balance verified by
script: 02_related_work {30/30} table* 1/1 tabular 1/1; 03_problem_statement
{141/141} table* 1/1 tabular 1/1. Column counts per row verified (properties 5,
notation 4). Both new labels have exactly one \ref each, no name collision (grep).
No [[VERIFY]] raised. Could NOT compile — no LaTeX engine; verified by reading +
grep. Next: Related Work / Introduction / Conclusion + figure_1 asset still
pending director.

2026-06-18 (orchestrator LaTeX-hygiene pass, NOT paper-writer — mechanical only):
addressing the two external reviews (content review .docx + REVIEW_formatting_and_
AI-traceability.md). Mechanical edits only, no prose composition:
- M3/AI-traceability: stripped the UTF-8 BOM from main.tex (was \xef\xbb\xbf;
  now starts with \documentclass). The only tooling fingerprint the formatting
  reviewer flagged.
- Ed5/style (acronym consistency): paper defined "received-signal-strength (RSS)"
  in 02_related_work but then used "RSSI" everywhere else (problem statement,
  methodology, experiments). Unified: 02_related_work now defines "received signal
  strength indicator (RSSI)" and "WiFi RSSI scans"; abstract keeps the spelled-out
  pre-acronym form (length-capped at 199/200, no acronym added there). No bare
  "RSS" remains (grep-confirmed).
- Ed5 (metric naming): grounded "raw ATE" at its definition in 05_experiments
  ("...position error, reported without alignment as the raw ATE,"); results +
  tab:perleg already used "raw ATE" uniformly, only the definition lagged.
DECISIONS (deliberately NOT changed, flagged to director):
- H2 xcolor: formatting reviewer says remove (unused). KEPT — supervisor's review
  workflow adds red/blue \textcolor comments in Overleaf; removing xcolor would
  break the next colored comment. Remove only at final submission packaging.
- C1 \blindtrue: formatting reviewer says ICINCO 2026 is DOUBLE-blind and
  \blindfalse is submission-blocking; PAPER_STATE earlier recorded a director
  confirmation of SINGLE-blind. CONFLICT — left \blindfalse, director must resolve.
- H1 file cleanup (draft.tex, figure_1.pdf, *.svg, SCITEPRESS.eps; add main.bbl):
  submission-zip-time action, not done now (sources kept in repo for editing).
- Min3/Ed4 equation rendering (Eq.15 sqrt(d_h) etc.): source is correct
  (\sqrt{d_h} clean; +B is the additive padding bias, intentionally outside the
  root). The reviewer's "extra symbol under the root" is a PDF-to-text extraction
  artifact, not a source bug. No edit.
- Min5 figure captions: all results captions already name dataset + split (+ metric
  where one applies). No edit needed.
Edited: paper/main.tex, paper/sections/02_related_work.tex,
paper/sections/05_experiments.tex (+ this file). Verified by reading + grep; no
LaTeX engine to compile.

2026-06-18 (director revisions to the two reader-aid tables, orchestrator):
- tab:notation (Min4) REMOVED entirely per director — notation stays defined
  inline in the paragraphs where each symbol is introduced, not dumped in a table.
  Deleted the table* block + its "Table~\ref{tab:notation} collects the notation."
  sentence from 03_problem_statement.tex; transition reads clean; no dangling ref
  (verified: tab:notation referenced nowhere). Inline symbol definitions were
  already present (the table had only restated them).
- tab:properties (Min2) made SINGLE-COLUMN to fit: table* -> table; removed the
  ~\cite{...} (author/year) from each method row (all 6 methods remain cited in
  the section prose, so refs.bib coverage unchanged and all 19 cite keys still
  resolve); header "Modalities / domain" -> "Domain" to save width. Marks/rows
  unchanged. Env balance + all refs/cites re-verified by script.
Edited: paper/sections/02_related_work.tex, paper/sections/03_problem_statement.tex
(+ this file). No LaTeX engine; verified by reading + grep.
- tab:datasets (Sec 5, "table 3") made SINGLE-COLUMN to fit: table* -> table;
  text columns Dataset/Modalities/Setting/Unit are now wrapping p{} columns
  (p{1.05}/p{0.8}/p{1.15}/p{0.78} cm), numeric columns (#APs/Train/Val/Test) kept
  as auto-width c so "19937" never overflows; \footnotesize + \tabcolsep 2pt;
  "Modalities" header shortened to "Modal." to avoid an unbreakable wide word in
  the narrow column. All numbers/rows unchanged. Estimated total width ~7.8 cm
  (< ~8.35 cm column) but NOT compiler-checked — watch for an overfull-hbox in
  Overleaf; fallback is merging Train/Val/Test into one "tr/val/test" column.
  Env balance + tab:datasets ref re-verified. Edited: 05_experiments.tex.
- CORRECTION (director, table-number mix-up): after the notation table was
  removed the numbering shifted (now 1=properties, 2=datasets, 3=perleg,
  4=fusion). Director meant "Table 3" = tab:perleg, not the datasets table.
  Reverted tab:datasets back to full-width table* (original 8-col l/c layout,
  no footnotesize); made tab:perleg (Results, encoder eval) SINGLE-COLUMN
  instead: table* -> table, p{} wrapping for Benchmark/Metric/Rel.-change
  columns, \tabcolsep 4pt, kept 9pt font (fits ~7.5 cm). Renamed the long
  header "wlan\_localization / ResNet1D" -> "Reference" and moved the
  per-row reference mapping into the caption; "Rel.\ change (\%)" -> "Rel.\
  change" (values already carry %). All numbers/rows unchanged. Current float
  types: 05_experiments table* x1 (datasets); 06_results table* x1 (fusion) +
  table x1 (perleg). Env balance + all refs/cites re-verified. Edited:
  05_experiments.tex, 06_results.tex.
- FIX (director: "text overflow border" on tab:perleg): the p{} fixed-width
  columns forced "UJIIndoorLoc" past its cell rule. Switched tab:perleg to
  auto-width |l|l|c|c|c| (cells size to content, so nothing crosses a border)
  + \small + \tabcolsep 3pt to keep the whole table inside the ~8.35 cm column
  (est ~7.9 cm). Full dataset names kept; header/data unchanged. Edited:
  06_results.tex.
- Figure 3 (fig:imucnn, IMU-CNN schematic) tweaks: placement [!hb] -> [t]
  (top of column; SCITEPRESS-preferred, ICINCO-compliant), caption shortened
  from the conv/GAP/MLP recap (already in the IMU-CNN paragraph + eqs) to
  "IMU-CNN encoder: a $T\times F$ inertial window mapped to one token."
  Director also asked to pin it to page-1 col-2: declined/not done -- a float
  cannot appear before its source page, and the figure is defined in Methodology
  (sec 4); relocating its source to the intro would detach it from its
  explanation. Not an ICINCO rule violation, but poor practice. Edited:
  04_methodology.tex.
- CORRECTION (director): meant Figure 1 (fig:pipeline, the overview), not
  Figure 3; and page 1 = page 1 of the whole paper. Reverted fig:imucnn fully
  (back to [!hb] + original conv/GAP/MLP caption). Moved fig:pipeline OUT of the
  Methodology Overview (was a full-width figure*) INTO 01_introduction as a
  single-column figure[t] placed right after intro paragraph 1, so the [t] float
  defers to the TOP OF COLUMN 2 on page 1 (col-1 top is taken by para 1).
  \includegraphics width 0.7\textwidth -> \columnwidth. Caption shortened to a
  one-line teaser ("each observation becomes one token, a single self-attention
  block fuses the set, and a learnable position query reads out (x,y)"). Added a
  page-1 cross-ref "(Figure~\ref{fig:pipeline})" in intro para 3. fig:pipeline
  stays Figure 1 (first in doc order), 1 label, now \ref'd 3x (intro + 2 in
  methodology), all resolve; no dup labels; envs balanced. CAVEAT: a near-square
  overview diagram (source ~22 cm) scaled to one column (~8.35 cm) may render
  small; fallback is a full-width figure* teaser across the top of page 1.
  ICINCO-compliant (overview figure on p.1 is standard). Edited:
  01_introduction.tex, 04_methodology.tex.

## Session before that
2026-06-18 (claim calibration, paper-writer persona): ONE focused iteration
recasting THREE claims to match the measured margins after two external reviews
(content + formatting/AI-traceability). No number, table, figure, label,
equation, or the `\ifblind`/author block touched.
- EDIT 1 (abstract verbs, paper/main.tex `\abstract{}`): rewrote the closing so
  it leads on the conceptual contribution + graceful degradation; states dead
  reckoning honestly ("improves on dead reckoning on the test session and matches
  it on validation" — test 10.90 vs PDR 12.49 m, val 16.67 vs 16.88 m tied);
  states the model "outperforms a learned WiFi--IMU fusion that fails to transfer
  across sessions" (IMUWiFine LSTM 52.69/65.15); demotes the wlan_localization
  k-NN gap (-61.5%) to "only a sanity check", not the headline. Abstract
  recounted at 199 words (was 184; 200-word ceiling) via the brace-matched
  word-count script — trimmed iteratively to stay under the cap.
- EDIT 2 (k-NN reframe, 06_results.tex End-to-End Fusion): added ONE sentence
  after the IMUWiFine negative-result sentence noting that out-performing the
  distance-weighted WiFi k-NN is expected for any motion-incorporating method, so
  the -61.5% figure is a sanity check; the informative comparisons are
  PDR-from-start and the IMUWiFine LSTM. No number/caveat deleted. The per-ENCODER
  UJIIndoorLoc 8.69/15.17/-42.7 result (different claim) left untouched.
- EDIT 3 (STraTS delineation, 02_related_work.tex): added ONE sentence after the
  SeFT/STraTS "but for clinical data" note pinning the delta beyond domain shift —
  (i) per-modality encoders mapping WiFi RSS scans + inertial windows into the
  shared token (STraTS assumes scalar clinical observations); (ii) modality +
  per-instant dropout for cross-session robustness (STraTS does not address). No
  hype word.
Anti-AI self-check: zero banned vocab (grep-confirmed), <=1 tricolon/paragraph,
no recap-after-figure, one idea per sentence (EDIT-3 two-part delta kept as one
sentence per the task's "ONE crisp sentence" instruction). Brace/environment
balance verified unchanged by script: main.tex {47/47} begin/end 1/1; 06_results
{89/89} 12/12; 02_related_work {20/20} 0/0. No [[VERIFY]] raised (every fact
already traceable in 06_results.tex; no new number introduced). Edited ONLY
main.tex, 06_results.tex, 02_related_work.tex (+ this file, ai-usage.md). Could
NOT compile — no LaTeX engine on this machine; verified by reading + grep.
Next: Related Work / Introduction / Conclusion + figure_1 asset still pending
director.

## Two sessions before that
2026-06-17 (Experiments + Results shrink ~10% each, paper-writer persona):
supervisor flagged both sections as too large; cut ~10% from EACH by removing
redundancy/over-explanation only, no facts/numbers/citations/tables/figures/
caveats deleted. paper/sections/05_experiments.tex 606 -> 547 words (3878 ->
3559 nospace, -9.7%); paper/sections/06_results.tex 1346 -> 1215 words (9239 ->
8483 nospace, -9.7%). Cites unchanged (Experiments 8, Results 2). All labels
(sec:experiments/sec:results, eq:mae/eq:ate, tab:datasets/perleg/fusion,
fig:ronin_traj/ronin_perseq/webots_scatter/msiln*/iwfine_showcase), all 7
\includegraphics, and all protected numbers verified intact by grep
(staleness chain 0.61->1.13->1.48->1.84->2.05/3.50, 8.69/15.17/-42.7,
9.72/5.14/+89.2, 0.05M/4.6M/95x, 10.90/28.31/12.49/52.69/16.67/16.88/21.26/
65.15, 6.37/2.31/1.06/0.84/+48.2/7.62, 4.77ms/0.146ms, 786/~28%). Redundancies
cut: (Exp) trimmed Datasets opener that recapped the table + deleted the
split-unit paragraph the caption already states; condensed reference-method
"It is the X reference" tails. (Results) cross-session story told once (kept the
wlan-doubling sentence, folded the IMUWiFine "fails to generalize" + curve recap
into one clause); removed recap-after-figure on ronin_traj/msiln_cdf/msiln_perpath/
iwfine_showcase; trimmed the RoNIN per-seq aggregate-gap repeat; dropped the
modality-dropout closing recap; compressed Limitations to the concession (kept
all numbers). Honest caveats intact (synthesized-WiFi-optimistic, cross-session
protocol, WiFi-dense-path test split). Edited ONLY the two section files (+ this
file, ai-usage.md). Could NOT compile — no LaTeX engine on this machine; verified
by reading + wc/grep. Next: Related Work / Introduction / Conclusion + figure_1
asset still pending director.

## Session before (earlier 2026-06-17)
2026-06-17 (Methodology refine-and-verify, paper-writer persona): owned the
prose of paper/sections/04_methodology.tex after the orchestrator's revision that
made functions f,g,h and the training loss visible (supervisor comments 1+2).
This was a prose-refine pass, NOT a rewrite — every structural/mathematical change
was preserved. Both comments stay satisfied: Overview names $f=h\circ g$
(Eq.~decomp); tokenization opener "realizes the encoding $g$"; fusion opener
"realizes the fusion $h$"; Algorithm 1 caption states
$\hat{\mathbf p}=h(g(\mathcal O))=f(\mathcal O)$ with `encoding g`/`fusion h`/
`readout of h`/`training` step comments; loss is Eq.~loss (Huber surrogate tied
to Eq.~objective) AND an algorithm line. Applied all three orchestrator-flagged
concision cuts: (1) merged the IMU shift-invariance triple to two distinct claims
(conv = equivariant, GAP = invariant); (2) trimmed WiFi-Net's editorial sentence
to one non-redundant fact (reference points learned, not tied to physical APs);
(3) folded the Overview's "WiFi sparse / IMU dense" overlap into a pointer to
Section~\ref{sec:problem}. Also split two-idea sentences (fusion opener,
MHSA/padding, loss list) and de-duplicated the readout CLS-safety sentence. Every
claim re-verified against model code (transformer.py, wifi.py, imu.py, fusion.yaml).
Char count 12,121 -> 11,907 excl. spaces (-214). All 15 labels + five
problem-statement cross-refs resolve; environments balanced (eq 11/11, cases 1/1,
gathered 1/1, aligned 1/1, algo 1/1); $\gamma$ kept for WiFi temperature; displayed
eqs kept column-safe (loss gathered+cases, layer aligned untouched). One [[VERIFY]]
raised: "~1.5 M params" vs measured ~1.38 M (kept verbatim per task constraint).
Edited ONLY 04_methodology.tex (+ this file, ai-usage.md). Could NOT compile — no
LaTeX engine on this machine; verified by reading + grep balance. Next: per the
7-day plan, Related Work / Introduction / Conclusion + figure_1 asset still pending
director.

## Earlier session (2026-06-01 results subsections)
2026-06-01 (Experiments results subsections, paper-writer persona): appended
the Per-Leg Encoder Validation and End-to-End Fusion subsections as full
publication PROSE after Experimental Setup in paper/sections/04_experiments.tex
(only this file edited). Per-Leg: WiFi-Net UJI val MAE 8.58 m vs
wlan_localization 15.17 m (-43.4%); IMUCNN RoNIN canonical raw ATE 9.72 m vs
ResNet1D 5.14 m (+89.2%), honestly framed (in-domain-competitive, cross-subject
cost at a ~95x-smaller budget; Umeyama deferred to limitations); created
Table tab:perleg and \ref'd it. End-to-End: Webots sim 2-mod K=4 val 0.395 /
test 0.375 m (live notebook, under the 0.5 m bar, no public SOTA on the sim);
MSILN site1/B1 cross-session headline val 15.22 / test 10.89 m vs
wlan_localization 21.26 / 28.31 m (-28.4% val, -61.5% test), compared ONLY
against wlan_localization (WiFi-kNN absent from prose); created Table tab:fusion
and \ref'd it. Webots val/test reconciliation kept as a COMMENT-ONLY % TODO at
the value (not in prose); NO figure float added (a % TODO marks the undecided
UJI-scatter-vs-MSILN-trajectory asset, no dangling \ref). Every numeric claim
carries a % src; no [[VERIFY]] needed. tab:perleg/tab:fusion/tab:datasets/
tab:train/tab:arch \ref/\label all resolve (verified by reading). Could NOT
compile — no LaTeX engine on this machine. Next: Ablations & robustness +
Discussion & limitations subsections; the Webots and figure decisions await
the director.

## Two sessions before
2026-06-01 (Experiments writing STARTED — Experimental Setup subsection,
paper-writer persona): wrote the FIRST Experiments subsection as full
publication PROSE inside paper/main.tex \subsection{Exp setup}, renamed to
\subsection{Experimental Setup}. Covers: (1) the four datasets and their roles
+ splits in §3 register — Webots sim (controlled lab/ablations; sim-WiFi-
synthesized caveat), MSILN site1/B1 (cross-session real-world headline),
UJIIndoorLoc (WiFi per-leg, 520 APs), RoNIN canonical (IMU per-leg, 32 unseen
sequences); only in-scope WiFi+IMU advertised for Webots/MSILN. (2) ADDED a
datasets summary Table tab:datasets (table*) with counts VERIFIED by running
src.pipeline.data.dataset_stats in the venv: webots 11/3/3 paths; msiln 94/34/5
paths; uji 19937/1111 samples + 520 APs; ronin 73/-/32 sequences. Split-unit
mixing (paths/samples/sequences) stated in the caption + flagged [[VERIFY]] in
a comment; MSILN AP count not exposed by dataset_stats -> left blank. (3)
Baselines = wlan_localization (WiFi) + RoNIN ResNet1D (IMU); WiFi-kNN DROPPED
entirely (director decision) in BOTH main.tex prose AND the draft.tex baselines
bullet. (4) Implementation prose \ref's tab:train (this subsection) + tab:arch
(Methodology); hardware Quadro P4000; seed 42; PyTorch. (5) Metric policy: MAE
+ raw ATE; Umeyama-aligned ATE only in the later limitations discussion. The
existing tab:train table + its % src:/% TODO reconcile-K comments left intact;
the \textcolor{red} Experiments roadmap note + all other scratch subsections +
preamble + title block + \ifblind + \input untouched. All \ref (tab:train,
tab:arch, tab:datasets) verified to resolve by reading. Did NOT resolve the
other open decisions (Webots number, K-sweep, RMSE/Umeyama placement) — those
belong to later subsections. Could NOT compile — no LaTeX engine on this
machine; verified by reading. Next: remaining Experiments subsections (Results
and discussions / cross comparison / fair comparison with SOTA / analysis); the
open decisions in draft.tex's OPEN DECISIONS block still need director input.

## Earlier this day
2026-06-01 (methodology rigor + symbolization pass, paper-writer persona):
implemented the director-approved TAKE 1 + TAKE 2 in
paper/sections/03_methodology.tex with one authorized edit to main.tex's
Exp-setup subsection. TAKE 1: added Algorithm 1 (alg:forward, algorithm2e,
§3.4) for the forward pass; added equations eq:wifi_assign + eq:wifi_token
(WiFi-Net), eq:imu (IMUCNN); expanded the single attention eq into eq:layer
(pre-norm residual MHSA+FFN sub-layers) + eq:mhsa (multi-head). TAKE 2:
defined a collision-free symbol set (D, M, K, N_a [renamed from k], T, F,
(C_1,C_2,C_3), L, H, r, n) and rewrote §3.3-§3.5 prose to read in symbols;
added the architecture config Table tab:arch at end of §3.4; moved the
training-hyperparameter dump out of §3.5 into a NEW training-config Table
tab:train inside main.tex \subsection{Exp setup} (the only main.tex edit);
§3.5 now reads symbolically and \ref's tab:train; removed the §3.5
\textcolor{red} exp-setup note (move fulfills it), preserved all other red
notes. All numbers unchanged in meaning; all % src comments + all \label/\ref
intact and resolving; Figure 1 block untouched. Could NOT compile — no LaTeX
engine on this machine; verified by reading. Next (pending director go): Day 2
Experiments; ingest director's figure_1 asset.

## Prior session
2026-06-01 (methodology-restructure pass, paper-writer persona): brought
paper/sections/03_methodology.tex (the MAIN file main.tex \input's) into the
director-validated structure from draft.tex, as full publication PROSE. Swapped
+ renamed the first two subsections to Overview (3.1) -> Problem Formulation
(3.2); rewrote Overview as a concept-only global map (no token/$\phi$ equations);
rewrote Problem Formulation as pure formalism (cut motivational framing);
relocated eq:token + eq:time (with their `% src:` comments) out of Overview into
Per-Modality Encoders (3.3) as the token lead-in; re-pointed the Training
subsection's stale "Section~3.1" cross-ref to "Section~3.2". Fusion/Readout +
Training prose otherwise unchanged. Figure 1 float, caption, scope-warning
comment, and [[VERIFY: figure-1 asset]] marker left EXACTLY as-is. Verified all
\label/\ref by reading (only cross-ref into the section is \ref{fig:pipeline};
no equation \ref'd by number; equation order eq:token,eq:time,eq:attn,eq:readout).
Could NOT compile -- no LaTeX/detex engine on this machine. Updated ai-usage.md.
Next (pending director go): Day 2 Experiments; ingest director's figure_1 asset.

## Earlier session
2026-06-01 (director-decisions pass, paper-writer persona): applied four director
decisions to the title block. FORMAT-ONLY changes (no new prose): (1) copied the
OFFICIAL orcid.eps from the SCITEPRESS author kit to paper/orcid.eps and WIRED
\orcidAuthor{<id>} after each author name in main.tex/draft.tex per the template
(Example.tex) pattern; (2) resolved full author names -- "Ilyass Abouelaziz",
"Yuehua Ding"; (3) updated the \blindfalse comment to director-CONFIRMED
single-blind, \ifblind toggle preserved; (4) left figure_1.jpg fully untouched
(director owns the in-scope single-PositionQuery diagram). Removed stale
"pending first names" / "missing orcid.eps" comment lines. Could NOT compile or
visually confirm the icon (no LaTeX/detex on this machine). Updated ai-usage.md.
Next (pending director go): Day 2 Experiments; ingest director's figure_1 asset.
