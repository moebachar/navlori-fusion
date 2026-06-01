---
name: paper-content
description: Decide and draft the substantive content of the ICINCO 2026 paper, section by section. Use whenever writing, structuring, or revising prose, or building the related-work corpus. Hybrid of ICINCO official rules, observed ICINCO 2024 cohort norms, and community scientific-writing conventions (IMRaD, CARS, related-work synthesis).
---

# Writing content for the ICINCO 2026 paper

## 0. Non-negotiable ICINCO rules (from the official Guidelines page)
- **Double-blind.** No author names, affiliations, "our group/lab", grant numbers, self-revealing acknowledgements, or the project public name anywhere in the draft. Phrase prior self-work in the third person ("Prior work [cite] showed..."), never "we previously showed".
- **Length is measured in CHARACTERS excluding white spaces** (NOT pages, at review):
  - Regular Paper: 10,000-50,000 characters.
  - Position Paper: 8,000-40,000 characters.
  Submissions outside the band may not be reviewed. Track the running count every session (the paper-format skill has the command).
- **Self-citations < 20%** of total references, and avoid an excessive co-author count.
- **AI-use disclosure:** any AI-generated text must be disclosed in Acknowledgements and the AI system cited. Keep a running list of AI-assisted passages in paper-workspace/ai-usage.md so the disclosure can be written at the end. (The acknowledgements/identity stay anonymized for review; disclosure text is drafted but the identifying parts are added only in the camera-ready.)
- Decide Regular vs Position with the user before drafting -- it sets the length band and the framing (completed+validated research -> Regular; work-in-progress/opinion -> Position).

## 1. Source hierarchy for every claim

Before drafting any prose, read these files in addition to scope.md and conference-rules.md:
- paper-workspace/style-icinco.md -- venue register (structure, voice, hedging shape)
- paper-workspace/style-anti-ai.md -- universal anti-LLM-tone rules (banned vocab, tricolon rules, self-check protocol)
- paper-workspace/icinco-2024-relevant.md -- curated ICINCO 2024 citation pool

Source hierarchy (in priority order) for every claim:
1. **The repo** (code, notebooks, run logs, metrics CSVs) -- ground truth for all numbers, settings, and results. Never invent a number. If a value isn't traceable to a repo file, write `[[VERIFY: where?]]` and list it at session end.
2. **scope.md** -- what we are allowed to publish in this paper (the rest is paper #2). If asked to include out-of-scope material, refuse and flag it.
3. **Zotero / local corpus + icinco-2024-relevant.md** -- for related work and positioning.
4. **conference-rules.md** -- the distilled ICINCO constraints.

When citing a result in prose, drop a LaTeX comment with the source path next to it, e.g. `% src: notebooks/stageB_eval.ipynb cell 14 / runs/2026-05-12/metrics.csv`.

## 2. Global structure (venue-calibrated)

ICINCO 2024 papers in our cohort follow this ordering in 6/6 sampled:

```
1 INTRODUCTION -> 2 RELATED WORK -> 3 METHODOLOGY -> 4 EXPERIMENTS -> 5 CONCLUSION
```

Important venue-specific findings from style-icinco.md (override the IMRaD defaults you may know):
- **Related Work comes BEFORE Methodology** in 6/6 sampled papers.
- **Results and Discussion are MERGED** (no standalone Discussion section).
- **No standalone Limitations section** -- limitations are hedged inline or appear in the Conclusion using the "concede -> despite this -> restate value" sentence shape (see style-icinco.md sec 5.3).
- **One pipeline figure** for the architecture (placed early in Methodology), not multiple decomposition diagrams.
- **5-6 numbered ALL-CAPS sections** is the mode.

Write sections in DEPENDENCY order, not document order: Methodology + Experiments first (you know them cold and they anchor everything), then Related Work, then Introduction, then Conclusion (with limitations folded in), then Abstract LAST.

## 3. Per-section playbooks

### Abstract -- write LAST, after a full draft exists
A miniature of the whole paper, standing on its own, ~150-200 words (ICINCO caps the template abstract at 200; aim 150-200). One paragraph following the IMRaD-abstract moves: context+motivation -> gap -> what we do (method in one line) -> key QUANTITATIVE result (a real number, not "improves performance") -> significance. No citations, no undefined acronyms. Include searchable keywords for indexing.

### Introduction -- funnel + CARS "moves" + venue opener
Follow Swales' CARS model (Create A Research Space), in the venue idiom:
1. **Establish the territory:** broad real-world/industrial motivation; why the problem matters. The ICINCO cohort opens with a broad importance statement (style-icinco.md sec 5.1, pattern (a)). (1 paragraph)
2. **Establish the niche:** review the current state briefly and indicate the GAP. The cohort pivot is "X achieves outstanding results, BUT a key challenge remains..." (style-icinco.md sec 5.1, pattern (b)). (1 paragraph)
3. **Occupy the niche:** state what this paper does, then a numbered **Contributions** list under an explicit "Contributions" subheading (~50% of the cohort do this, the others use prose -- pick the explicit list for clarity), then a one-sentence paper-structure roadmap. (1-2 paragraphs)

Funnel: general -> specific. End the Introduction with a roadmap sentence ("The remainder of this paper is organized as follows: ...") -- this is near-universal in the cohort (style-icinco.md sec 5.4).

### Related Work -- synthesis, not a "shopping list"
The single most common reviewer complaint is the "Paper A did X. Paper B did Y." list. Avoid it. Rules:
- **Method-family grouping** is the venue idiom (style-icinco.md sec 4). The Triplet paper opens "There exist multiple families of approaches based on absolute pose regression... and place recognition..." then treats each group. Use that pattern.
- **Funnel:** broad at the start, focused at the end.
- **Paragraph template:** (a) opening sentence stating why these works form a group, citing all of them; (b) 1-2 sentences each on the few most relevant, naming the approach and finding; (c) a closing sentence on what the group collectively MISSES and how this paper differs/extends. Strengths AND limitations across papers.
- Every cited work must exist in refs.bib. Every claim about a paper must be checkable.
- Defend any "first/novel" claim here -- never assert novelty without the group that surrounds it.
- **Citation pool ordering:** start from icinco-2024-relevant.md (venue-local cites help "Relevance" reviewer score), then bring in Zotero entries for the actual baselines and SOTA. icinco-2024-relevant.md warns there is no direct WiFi-RSSI / IMU dead-reckoning SOTA in the proceedings -- those baselines stay sourced from the repo/Zotero.

### Methodology (one pipeline figure, equations + prose)
Formal problem statement first (notation; symbol table only if it actually helps -- venue norm is inline symbol definition).
- **One pipeline figure** placed early in this section: "Figure 1: Overview/Pipeline of [the system]". Subsystem decomposition happens *within* that one figure (data flowing through labelled blocks), not as several separate architecture figures.
- **Numbered equations sparingly** (4/6 cohort papers use them, 2/6 are almost equation-free). A handful of equations for the load-bearing definitions (token construction, attention, loss); not a wall of math.
- **Hyperparameters inline in a "Training Details" prose paragraph** (6/6 cohort papers do this). Optimizer, schedule, batch size, epochs, hardware -- all in running prose, not a table. Example phrasing: "Models were trained for 90 epochs with AdamW (lr 1e-4, batch size 128) on a single NVIDIA RTX 3090."
- Describe components in the order data flows through the system. Reproducibility: state every hyperparameter, dataset split, and design choice a reader would need to reimplement -- pull these verbatim from the repo, do not paraphrase numbers.

### Experiments (results + discussion merged)
Setup block: dataset(s), metrics (define them), baselines (justify the choice), hardware. Then results tables -> ablations -> brief inline observations.
- **At least one external SOTA baseline per claim** -- venue median is 1-2. Our planned setup (wlan_localization on MSILN, RoNIN on IMU) meets/exceeds this.
- **Tables for headline numbers, plots for trends.** 5/6 cohort papers use both.
- **Ablations as a labelled subsection** within Experiments -- above venue norm but matches our scope.
- Every number, table, and figure traces to a repo file (cite the path in a `%` comment).
- **Results and Discussion are merged** -- interpret inline rather than in a separate section. Lead each result with the headline metric. Compare against baselines fairly.

ICINCO reviewers are explicitly asked: "needs more experimental results?", "needs comparative evaluation?" -- so include baselines and at least one comparison.

### Conclusion (incl. honest limitations + future work)
Cohort shape (style-icinco.md sec 5.5): three moves in one short section.
1. Restate what was proposed.
2. Restate the key result/finding (with a number).
3. Honest limitations + future-work, using the "concede -> despite this -> restate value" sentence shape from style-icinco.md sec 5.3. Future work may *hint* at the second paper's direction WITHOUT revealing scope-restricted specifics.

Conclusions are short (one paragraph to half a column). Don't pad.

## 4. Mapping to ICINCO's reviewer checklist (self-review before "done")
Reviewers rate Relevance, Originality, Technical Quality, Significance, Presentation, and answer: Abstract+Intro adequate? More experiments needed? Comparative evaluation present? Critical discussion improved? Figures adequate? Conclusions/future work convincing? References up-to-date+appropriate? Formatting OK? English OK?
Run a hostile-reviewer self-pass against this list before declaring any section final.

## 5. Related-work corpus workflow (the heavy lift)
1. **Use the curated ICINCO 2024 pool first:** paper-workspace/icinco-2024-relevant.md already has 26 candidates with scope tags. Pick from `[in-scope:paper-1]` entries (11 papers) per the cohort grouping plan.
2. **Pull the Zotero library:** query the Zotero MCP, export to paper-workspace/zotero-export.json (title, authors, year, abstract, key, tags).
3. **Cross-check the repo:** grep code/notebooks/reports for arXiv IDs, DOIs, BibTeX blocks, and author names. Any paper *used in the work but absent from Zotero* -> paper-workspace/missing-citations.md for the user to add (or add via Zotero MCP if write is supported). Any paper in Zotero that's irrelevant to scope.md -> exclude.
4. **Find gaps:** for each theme in scope.md with thin coverage, write a focused query into paper-workspace/deep-search-prompts/NN-theme.md for the user's existing deep-search agent. Each prompt states: the theme, what we already have, what's missing, inclusion/exclusion criteria, and desired output (BibTeX + 2-line summary).
5. **Ingest** returned papers into refs.bib + the local corpus, then draft per the Related Work playbook above.

## 6. Style rules (deferred to dedicated files)
- Venue voice, hedging shape, contribution phrasing, intro opener patterns -> **style-icinco.md** (read it; it has real quotations from the cohort).
- Banned vocabulary, tricolon rules, recap-after-figure ban, self-check protocol -> **style-anti-ai.md** (run the self-check protocol before declaring any paragraph done; log violations to paper-workspace/style-violations.md).
- Plain, precise scientific English; first person plural ("we propose") is venue-standard.
- Define every acronym on first use.
- Present every figure/table in text BEFORE it appears.
- No claim without a source (repo for results, citation for prior art).
- Mark every uncertain value `[[VERIFY]]`; never paper over a gap with a plausible number.

## Sources combined to build this skill
- ICINCO 2026 Guidelines (official): double-blind, 10k-50k / 8k-40k character bands, <20% self-citation, AI-disclosure rule, reviewer rating criteria & question list.
- ICINCO 2024 proceedings cohort observations (paper-workspace/style-icinco.md): section ordering, merged Results+Discussion, no Limitations section, one-pipeline-figure norm, inline hyperparameters, method-family grouping, contribution-statement signal phrases, "concede -> despite this -> restate value" sentence shape.
- Swales CARS model -- introduction "moves" (establish territory / niche / occupy).
- IMRaD convention (writing centers: GMU, Stanford, UBC) -- section roles; abstract written last.
- Related-work synthesis guidance (Nacke; "Slow Searching" formula) -- theme/method grouping, funnel shape, anti-"shopping-list", group-level limitation sentences.