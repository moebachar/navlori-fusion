---
name: paper-content
description: Decide and draft the substantive content of the ICINCO 2026 paper, section by section. Use whenever writing, structuring, or revising prose, or building the related-work corpus. Hybrid of ICINCO's official rules and community scientific-writing conventions (IMRaD, CARS, related-work synthesis).
---

# Writing content for the ICINCO 2026 paper

## 0. Non-negotiable ICINCO rules (from the official Guidelines page)
- **Double-blind.** No author names, affiliations, "our group/lab", grant numbers,
  self-revealing acknowledgements, or the project's public name anywhere in the draft.
  Phrase prior self-work in the third person ("Prior work [cite] showed…"), never
  "we previously showed".
- **Length is measured in CHARACTERS excluding white spaces** (NOT pages, at review):
  - Regular Paper: **10,000–50,000** characters.
  - Position Paper: **8,000–40,000** characters.
  Submissions outside the band may not be reviewed. Track the running count every
  session (the paper-format skill has the command).
- **Self-citations < 20%** of total references, and avoid an excessive co-author count.
- **AI-use disclosure:** any AI-generated text must be disclosed in Acknowledgements
  and the AI system cited. Keep a running list of AI-assisted passages in
  paper-workspace/ai-usage.md so the disclosure can be written at the end. (The
  acknowledgements/identity stay anonymized for review; disclosure text is drafted
  but the identifying parts are added only in the camera-ready.)
- Decide Regular vs Position with the user before drafting — it sets the length band
  and the framing (completed+validated research → Regular; work-in-progress/opinion
  → Position).

## 1. Source hierarchy for every claim
1. **The repo** (code, notebooks, run logs, metrics CSVs) — ground truth for all
   numbers, settings, and results. Never invent a number. If a value isn't traceable
   to a repo file, write `[[VERIFY: where?]]` and list it at session end.
2. **scope.md** — what we are allowed to publish in this paper (the rest is paper #2).
   If asked to include out-of-scope material, refuse and flag it.
3. **Zotero / local corpus** — for related work and positioning.
4. **conference-rules.md** — the distilled ICINCO constraints.

When citing a result in prose, drop a LaTeX comment with the source path next to it,
e.g. `% src: notebooks/stageB_eval.ipynb cell 14 / runs/2026-05-12/metrics.csv`.

## 2. Global structure — IMRaD (Introduction, Methods, Results, Discussion)
This is the standard empirical-paper skeleton expected by reviewers in control/
robotics/CS venues. Adapt to: Abstract → Introduction → Related Work → Methodology
→ Experiments/Results → Discussion → Conclusion (+ Acknowledgements, References,
optional Appendix). Write sections in DEPENDENCY order, not document order:
Methodology + Results first (you know them cold and they anchor everything), then
Related Work, then Introduction, then Conclusion/Limitations, then Abstract LAST.

## 3. Per-section playbooks

### Abstract — write LAST, after a full draft exists
A miniature of the whole paper, standing on its own, ~150–200 words (ICINCO caps the
template abstract at 200; aim 150–200). One paragraph following the IMRaD-abstract
moves: context+motivation → gap → what we do (method in one line) → key QUANTITATIVE
result (a real number, not "improves performance") → significance. No citations, no
undefined acronyms. Include searchable keywords for indexing.

### Introduction — funnel + CARS "moves"
Follow Swales' CARS model (Create A Research Space):
1. **Establish the territory:** broad real-world/industrial motivation; why the
   problem matters. (1 paragraph)
2. **Establish the niche:** review the current state briefly and indicate the GAP —
   what existing approaches fail to do for *this* problem. (1 paragraph)
3. **Occupy the niche:** state what this paper does, then an explicit, numbered
   **contributions** list (2–4 items), then a one-line paper-structure preview.
   (1–2 paragraphs)
Funnel shape: general → specific. End with the numbered contributions; reviewers look
for them.

### Related Work / State of the Art — synthesis, not a "shopping list"
The single most common reviewer complaint is the "Paper A did X. Paper B did Y."
list. Avoid it. Rules:
- **Funnel:** broad at the start, focused at the end.
- **Group by THEME or METHOD family**, never one-paragraph-per-paper. For an
  engineering/CS paper, methodological grouping works well (e.g. classical filtering
  vs. learning-based fusion vs. neural-ODE/asynchronous approaches — map to scope.md).
- **Paragraph template:** (a) opening sentence stating why these works form a group,
  citing all of them; (b) 1–2 sentences each on the few most relevant, naming the
  approach and finding; (c) a closing sentence on what the group collectively MISSES
  and how this paper differs/extends. Strengths AND limitations across papers.
- Every cited work must exist in refs.bib. Every claim about a paper must be checkable.
- Defend any "first/novel" claim here — never assert novelty without the group that
  surrounds it.

### Methodology
Formal problem statement first (notation; a notation table if it helps). Then describe
components in the order data flows through the system. Exactly one system-architecture
figure that the reader can follow. Reproducibility: state every hyperparameter,
dataset split, and design choice a reader would need to reimplement — pull these
verbatim from the repo, do not paraphrase numbers.

### Experiments / Results
Setup block: dataset(s), metrics (define them), baselines (justify the choice),
hardware. Then results tables → ablations → brief inline observations. Lead each
result with the headline metric. Every number, table, and figure traces to a repo
file (cite the path in a % comment). Report results here; INTERPRET them in Discussion.
ICINCO reviewers are explicitly asked: "needs more experimental results?",
"needs comparative evaluation?" — so include baselines and at least one comparison.

### Discussion
Interpret, don't re-report. What do the results mean, where do they hold/break, why.
Connect back to the gap from the Introduction.

### Conclusion + Limitations + Future Work
Restate contribution and the key quantitative result. An HONEST limitations paragraph
(reviewers reward it; ICINCO asks "conclusions/future work convincing?"). Future work
may *hint* at the second paper's direction WITHOUT revealing scope-restricted specifics.

## 4. Mapping to ICINCO's reviewer checklist (self-review before "done")
Reviewers rate Relevance, Originality, Technical Quality, Significance, Presentation,
and answer: Abstract+Intro adequate? More experiments needed? Comparative evaluation
present? Critical discussion improved? Figures adequate? Conclusions/future work
convincing? References up-to-date+appropriate? Formatting OK? English OK?
Run a hostile-reviewer self-pass against this list before declaring any section final.

## 5. Related-work corpus workflow (the heavy lift)
1. **Pull the library:** query the Zotero MCP, export to
   paper-workspace/zotero-export.json (title, authors, year, abstract, key, tags).
2. **Cross-check the repo:** grep code/notebooks/reports for arXiv IDs, DOIs, BibTeX
   blocks, and author names. Any paper *used in the work but absent from Zotero* →
   paper-workspace/missing-citations.md for the user to add (or add via Zotero MCP if
   write is supported). Any paper in Zotero that's irrelevant to scope.md → exclude.
3. **Find gaps:** for each theme in scope.md with thin coverage, write a focused query
   into paper-workspace/deep-search-prompts/NN-theme.md for the user's existing
   deep-search agent. Each prompt states: the theme, what we already have, what's
   missing, inclusion/exclusion criteria, and desired output (BibTeX + 2-line summary).
4. **Ingest** returned papers into refs.bib + the local corpus, then draft per the
   Related Work playbook above. (If a local RAG corpus is used instead of NotebookLM,
   chunk the PDFs and keep a simple index the agent can grep/query.)

## 6. Style rules
- Plain, precise scientific English; short sentences; define every acronym on first use.
- Present every figure/table in text BEFORE it appears.
- No claim without a source (repo for results, citation for prior art).
- Mark every uncertain value `[[VERIFY]]`; never paper over a gap with a plausible number.

## Sources combined to build this skill
- ICINCO 2026 Guidelines (official): double-blind, 10k–50k / 8k–40k character bands,
  <20% self-citation, AI-disclosure rule, reviewer rating criteria & question list.
- Swales CARS model — introduction "moves" (establish territory / niche / occupy).
- IMRaD convention (writing centers: GMU, Stanford, UBC) — section roles; abstract
  written last; report-vs-interpret split between Results and Discussion.
- Related-work synthesis guidance (Nacke; "Slow Searching" formula; NotebookLM
  related-work method) — theme/method grouping, funnel shape, anti-"shopping-list",
  group-level limitation sentences.
