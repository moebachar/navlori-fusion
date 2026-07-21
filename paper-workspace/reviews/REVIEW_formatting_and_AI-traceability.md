# Review — Formatting/Template Compliance & AI‑Traceability

**Paper:** *Continuous‑Time Set‑Transformers for Asynchronous WiFi–IMU Indoor Localization*
**Target venue:** ICINCO 2026 (SCITEPRESS / INSTICC template) — 26–28 Oct 2026, Angers, France
**Scope of this review:** (1) formatting & adherence to the SCITEPRESS template; (2) whether the document reads as AI‑written.
**Out of scope (handled by your other reviewer):** scientific content, claims, reference *accuracy/existence*, experimental validity.

**Method used:** I compiled the actual source against the SCITEPRESS class, rendered all 10 pages, and ran scans for layout errors, character/word counts, citation coverage, hidden characters, and stylistic markers. Findings below are evidence‑based, not impressionistic.

---

## Bottom line

- **Formatting:** Strong. It compiles clean (0 overfull boxes, no undefined references/citations, clean BibTeX) and respects the template across title block, two‑column body, headings, tables, figures, equations, algorithm, and the APA reference style. There are **a few concrete fixes**, one of which is **submission‑blocking** (anonymization for double‑blind).
- **AI‑traceability:** The prose is **already very clean of the usual machine‑writing tells** — a human reviewer is unlikely to read it as AI‑generated. The only items are minor optional polish, plus one **policy fact you must decide on knowingly** (ICINCO requires an AI‑use disclosure in the acknowledgements of the camera‑ready).

---

# PART 1 — Formatting & Template Compliance

## 1.1 Verified compliant (checked in the rendered PDF)

| Area | Status |
|---|---|
| Document class / two‑column body + one‑column title block | ✅ correct structure |
| Title — titlecased, bold, centered | ✅ |
| Authors — first‑name‑first, single shared affiliation, email | ✅ |
| Keywords — titlecased, comma‑separated, ends with period | ✅ |
| Abstract — **184 words** (limit 70–200), 9 pt | ✅ |
| Section headings ALL‑CAPS (`\uppercase`); sub/subsub titlecased | ✅ |
| Tables — caption **above**, 9 pt, end with period, wide ones use `table*` | ✅ |
| Figures — caption **below**, 9 pt, end with period, wide ones use `figure*`, vector PDFs | ✅ |
| Equations — numbered consecutively, centered | ✅ |
| Algorithm — colon separator, caption ends with period | ✅ |
| Citations APA `(Author, year)`; references via `apalike`, no numbers, alphabetical | ✅ |
| Build health — 0 overfull boxes, 0 undefined refs/cites, clean BibTeX | ✅ |
| **Length — 31,092 characters excl. whitespace** (~6,040 words) | ✅ within ICINCO's 10,000–50,000 window |

> Note on length: ICINCO gates **Regular Papers by character count (10,000–50,000, excl. whitespace)**, not by pages. You're comfortably mid‑range. The paper renders to 10 pages; that's fine under a character‑based limit. (The site's exact printed‑page figure wasn't machine‑readable on the Guidelines page — the character window is the governing constraint and you're inside it.)

## 1.2 Issues to fix — ranked

### 🔴 CRITICAL — submission‑blocking

**C1. Double‑blind anonymization is OFF.**
ICINCO 2026 review is **double‑blind**: "produce and provide the paper WITHOUT any reference to any of the authors, including the authors' personal details, the acknowledgments section… and any other reference that may disclose the authors' identity." Right now the paper is in non‑blind mode and prints **real names, "CESI LINEACT, Reims, France", emails, and ORCID IDs**.

- **Fix (one line):** in `main.tex`, change
  `\newif\ifblind \blindfalse` → `\newif\ifblind \blindtrue`
  Your `\ifblind … \else … \fi` block already produces the anonymized author block, so this is all that's needed for the review version. Flip back to `\blindfalse` only for the camera‑ready.
- **Good news:** the **body is already clean** — I scanned every section and found no institution names, author names, or "our previous work [cite]" self‑references. Once `\blindtrue` is set, the body won't leak identity.

### 🟠 HIGH — explicit template rules / things reviewers check

**H1. Remove previous‑version and unused files before zipping.**
Template, verbatim: "ONLY the files required to compile your paper should be submitted. Previous versions or examples MUST be removed." Your zip currently contains:

- `draft.tex` — old `TODO` version → **remove**
- `figures/figure_1.pdf` — **not used** (your `main.tex` includes `figure_1_latest.pdf`) → **remove**
- `figures/figure_1.svg`, `figures/figure_1_latest.svg`, `figures/figure_2.svg`, `figures/figure_3.svg` — editable sources, not used by LaTeX → **remove**
- `SCITEPRESS.eps` — not referenced by your `main.tex` (only `orcid.eps` is needed by `\orcidAuthor`) → optional remove
- **Add** `main.bbl` to the submission (so it builds without a BibTeX pass on their side), and keep `refs.bib`. Required files to keep: `main.tex`, `sections/`, used figures, `article.cls`, `SCITEPRESS.sty`, `apalike.sty`, `apalike.bst`, `orcid.eps`, `refs.bib` (+ `main.bbl`).

**H2. Package load order + an unused package.**

- The template says add packages **before** `SCITEPRESS.sty`. You load `xcolor` and `hyperref` **after** it. `hyperref`‑last is conventional and you used `[hidelinks]` (camera‑ready safe), but it *technically* contradicts the template note, and some SCITEPRESS venues prefer no `hyperref` at all. **Verify it's acceptable for ICINCO**, or drop it — here it only makes refs/cites clickable, nothing visual.
- **`xcolor` is loaded but never used anywhere** → **remove** `\usepackage{xcolor}`.

### 🟡 MEDIUM — verify / low‑risk

**M1. En‑dash in the title (`WiFi--IMU` → "WiFi–IMU").**
Template: "No formulas or special characters … allowed in the title." An en‑dash is punctuation and renders fine in the PDF, but the **PRIMORIS portal stores a plain‑text title**, where en‑dashes are sometimes flagged or mangled. Safe options: use a hyphen ("WiFi‑IMU") or "WiFi and IMU" **in the portal title field**; the PDF title can stay as‑is if you prefer.

**M2. Acknowledgements / declarations block (camera‑ready only).**
The template ships a declarations block before the references (Conflicts of Interest, Funding, Author Contributions, Data Sharing, AI Tools). Your paper omits it. **For the blind review version this omission is correct** (double‑blind forbids acknowledgements). **For the camera‑ready, add it.** See Part 2 for the AI‑Tools line specifically.

**M3. Strip the UTF‑8 BOM from `main.tex`.**
`main.tex` alone begins with a byte‑order mark (U+FEFF); all section files are clean. It didn't break this build, but it's untidy and can trip some toolchains. Re‑save `main.tex` as **UTF‑8 without BOM**.

### ⚪ LOW — informational, no output impact

- **L1.** `\noindent` appears on some section‑opening paragraphs (Problem Statement, Methodology, Experiments, Results) but not others (Introduction, Related Work, Conclusion). I confirmed this makes **no visual difference** — the SCITEPRESS `\section`/`\subsection` already suppress first‑paragraph indentation, so `\noindent` is redundant. Make uniform or ignore.
- **L2.** `refs.bib` has 69 entries but only **19 are cited**; `apalike` prints only cited ones, so the **reference list is correct and clean**. The 50 uncited entries don't appear in the PDF — harmless. Prune if you want a lean repo. (Whether some *should* be cited is a content question for your other reviewer.)
- **L3.** "Conclusion" (singular) vs the template's "Conclusions" — immaterial.

---

# PART 2 — AI‑Traceability

I'll separate two things, because they're different questions:
**(a)** does the writing *read* as machine‑generated (a quality/naturalness question I can review and improve); and
**(b)** the venue's AI‑disclosure policy (a factual compliance point that's your call).

## 2.1 (a) Does it read as AI‑written? — No; the prose is clean

Evidence from scans of all sections:

- **No cliché AI vocabulary.** Zero hits for *delve, leverage, harness, underscore, pivotal, crucial, realm, landscape, seamless, intricate, nuanced, testament, showcase, comprehensive, robustly,* etc. (The single "showcase" match is a **figure filename**, not prose.)
- **Zero em‑dashes (`---`).** LLM text leans on these heavily; you have none.
- **None of:** "not only … but also", "it is worth noting / it should be noted", "however" (0×), cleft sentences ("it is X that Y"), or hedging stacks ("serves to / aims to / in order to").
- **High sentence‑length burstiness.** Coefficient of variation **0.56**, sentences ranging **4–74 words**. Machine text tends to be flat/uniform (~0.3). This is a strong human signal.
- **Few participial summary‑tails** ("…, indicating …"): only **3** in the whole paper (normal human range, not the AI‑typical pile‑up).
- **All section files are pristine ASCII** — no smart quotes, non‑breaking spaces, or zero‑width characters. That means the text was **not pasted from a chat window or word processor** (which inject those); it was authored in a plain LaTeX editor. Another clean signal.

**Net:** the voice is consistent and specific; a human reviewer is very unlikely to flag it as AI‑written.

### Optional polish (these are *not* reliable AI tells — purely stylistic)

- **"so" as a connective appears 22×.** It's a personal verbal habit (and if anything *anti*‑AI, since models prefer "thus/therefore"). Vary a handful for formality if you like.
- **~30% of sentences open with "The"** — typical for ML papers; vary one or two for rhythm.
- **Recurring "Two X… / three Y…" framing** (Two limits, Two conditions, Two metrics, Two limitations, three properties, three blocks, five datasets). The counts are all genuine, so it's defensible; consider rephrasing one or two openers for variety.
- **The BOM on `main.tex`** (also in §1.2/M3) is the one actual tooling fingerprint — remove it.

## 2.2 (b) The policy fact you should decide on knowingly

ICINCO 2026 publishes this (verbatim): *"The use of artificial intelligence (AI)–generated text in an article should be disclosed in the acknowledgements section of any paper submitted to this conference. The sections of the paper that use AI‑generated text should include a citation to the AI system used to generate the text."* Reviews are double‑blind, and **all papers are run through plagiarism analysis** before review.

A useful distinction the policy implies:

- **Editing/polishing your own research text** (grammar, clarity, phrasing) is normally covered by a **single‑line disclosure**. The SCITEPRESS template even ships the standard wording: *"The paper used AI tools for text correction purposes only."*
- **AI‑*generated* text** (drafting sections) is what triggers the stronger section‑level citation requirement.

**Practically, by version:**

- **Review version (blind):** no acknowledgements at all (double‑blind forbids them) — nothing to add now.
- **Camera‑ready:** add the declarations block; per ICINCO, that's where any AI‑use disclosure goes, in the form that matches what you actually did.

I'm surfacing this because you asked for *all* the feedback and it bears directly on your goal. What you put in the camera‑ready acknowledgements is your decision. The naturalness review in §2.1 is just about making the writing read well, which is legitimate regardless of that decision.

---

# Pre‑submission checklist

- [ ] **Set `\blindtrue`** for the review PDF (flip back for camera‑ready) — **C1**
- [ ] Remove `draft.tex`, `figures/figure_1.pdf`, the four `figures/*.svg`, (optionally `SCITEPRESS.eps`); include `main.bbl` — **H1**
- [ ] Remove `\usepackage{xcolor}`; confirm `hyperref` is allowed (or drop it) — **H2**
- [ ] Decide on title en‑dash in the PRIMORIS title field — **M1**
- [ ] Re‑save `main.tex` as UTF‑8 **without BOM** — **M3 / §2.1**
- [ ] (Camera‑ready) add the declarations block incl. the AI‑Tools line — **M2 / §2.2**
- [ ] Confirm final character count stays within 10,000–50,000 after edits (currently 31,092 excl. whitespace) — ✅ today
- [ ] Optional: vary a few "so" connectives and "The"/"Two …" sentence openers — **§2.1**
