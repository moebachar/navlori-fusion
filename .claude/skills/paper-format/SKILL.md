---
name: paper-format
description: Produce and validate LaTeX for the ICINCO / SCITEPRESS template — preamble, title block, sections, tables, figures, equations, citations, compilation, and the character-count check. Use whenever generating, editing, or compiling .tex for this paper. Values taken verbatim from the official SCITEPRESS LaTeX template and FormatContentsForAuthors.
---

# Formatting the ICINCO / SCITEPRESS paper

The template is authoritative. Do NOT change `article.cls`, `SCITEPRESS.sty`,
`apalike.sty`, `apalike.bst`, the margins, fonts, or column geometry. If a custom
`.sty` is added, it MUST be shipped in the final submission zip.

## 1. Required files (copy into the LaTeX project, do not edit the first four)
Group 1 (never edit): `apalike.bst`, `apalike.sty`, `article.cls`, `SCITEPRESS.sty`.
Group 2 (edit/replace): `example.bib` → `refs.bib`, `example.tex` → `main.tex`,
and your figure assets (the template's `SCITEPRESS.eps` is just a demo image).

## 2. Preamble — copy exactly, add packages BEFORE SCITEPRESS.sty
```latex
\documentclass[a4paper,twoside]{article}
\usepackage{epsfig}
\usepackage{subcaption}
\usepackage{calc}
\usepackage{amssymb}\usepackage{amstext}\usepackage{amsmath}\usepackage{amsthm}
\usepackage{multicol}
\usepackage{pslatex}
\usepackage{apalike}
\usepackage{algorithm2e}
\usepackage[bottom]{footmisc}
% --- any extra packages (e.g. booktabs, graphicx, hyperref) go HERE ---
\usepackage{SCITEPRESS}   % MUST be last
```

## 3. Title block (anonymized for double-blind review)
Use a `\blind` toggle so author info exists but is suppressed for review:
```latex
\newif\ifblind \blindtrue   % set \blindfalse only for the camera-ready
\title{Title in Titlecase: Subtitle After a Colon}
\ifblind
  \author{\authorname{Anonymous Author(s)}
    \affiliation{Affiliation withheld for double-blind review}
    \email{}}
\else
  \author{\authorname{First Author\sup{1}\orcidAuthor{0000-0000-0000-0000} and Second Author\sup{2}}
    \affiliation{\sup{1}Institute, University, Address, Country}
    \affiliation{\sup{2}Department, University, Address, Country}
    \email{first@x.edu, second@y.edu}}
\fi
\keywords{Keyword One, Keyword Two, Keyword Three.}   % titlecased, ends with a period
\abstract{... at least 70 and at most 200 words, ends with a period.}
\onecolumn \maketitle \normalsize \setcounter{footnote}{0} \vfill
```
- **Title/subtitle:** titlecased; small words (is/or/then) lowercased unless first;
  subtitle separated by a colon; no formulas/special characters in the title.
- **Keywords:** ≥1, titlecased, comma-separated, sentence ends with a period, 9pt
  (handled by the style).
- **Abstract:** 70–200 words, ends with a period, 9pt (handled by the style).

## 4. Page setup (already enforced by SCITEPRESS.sty — do not override)
A4 210×297mm. Margins top 3.3cm / bottom 4.2cm / left 2.6cm / right 2.6cm.
Two-column body, each column 7.5cm, column spacing 0.8cm, body text 10pt.
**No headers, footers, running heads, or page numbers** — added electronically.
Anything outside the margins will not be printed (no figure bleed).

## 5. Sections
- `\section{...}` headings are **ALL-CAPS**: `\section{\uppercase{Introduction}}`.
- `\subsection{Titlecase Title}` and `\subsubsection{Titlecase Title}` use titlecase.
- First paragraph after any heading: **no indent** → prefix with `\noindent`.
- Number sequentially; avoid a lone single subsection inside a section.

## 6. Tables (caption ABOVE, 9pt, no bold/italic, centered)
```latex
\begin{table}[h]
\caption{One-line captions are centered; multi-line captions are justified.}\label{tab:x}
\centering
\begin{tabular}{|c|c|}\hline
Col 1 & Col 2 \\ \hline
a & b \\ \hline
\end{tabular}
\end{table}
```
- Word "Table" spelled out. Caption ends with a period. Span both columns with
  `table*` (top/bottom of page only).

## 7. Figures (caption BELOW, 9pt, centered, ≥300dpi)
```latex
\begin{figure}[!h]\centering
  {\epsfig{file=figures/arch.eps, width=7.5cm}}
  \caption{Caption below the figure, ends with a period.}\label{fig:arch}
\end{figure}
```
- Word "Figure" spelled out. Lines in line-drawings constant width, readable grids.
- Span both columns with `figure*` (top/bottom of page only).
- Every figure has `\label` and is `\ref`'d in the text BEFORE it appears.

## 8. Equations — separate line, numbered per section/contribution, right-justified
```latex
\begin{equation}\label{eq1}
  a = b + c
\end{equation}
```

## 9. Algorithms & program code
- Algorithms via `algorithm2e`; caption 9pt, colon separates "Algorithm N:" and name,
  one-line centered / multi-line justified, ends with a period.
- Inline code/commands in typewriter (Courier New). Verbatim blocks left-aligned, 9pt.

## 10. Citations & references — APA (author, year), NOT numeric
- `\bibliographystyle{apalike}` + `{\small \bibliography{refs}}`.
- Cite with `\cite{key}` → renders "(Author, year)". **Never** numeric `[1]` style.
- Every reference must be cited in the text. References 9pt (style-handled),
  citations 10pt. Keep self-citations < 20% of the reference list.
- BibTeX keys: `firstauthorYEARword`. Before every compile, grep all `\cite{...}`
  keys and confirm each exists in refs.bib.

## 11. Acknowledgements & Appendix (unnumbered, specific placement)
- Acknowledgements: `\section*{\uppercase{Acknowledgements}}` immediately BEFORE the
  references. **Omit/anonymize for review** (it can reveal identity). This is where the
  AI-use disclosure goes in the camera-ready.
- Appendix: `\section*{\uppercase{Appendix}}` directly AFTER references, not on a new
  page, unnumbered.

## 12. Compile & validate locally before any git push to Overleaf
```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
# treat as FAILURE this session: undefined references/citations, overfull \hbox > 5pt
grep -nE 'Warning|Undefined|Overfull' main.log | head -40
```

## 13. Character-count check (ICINCO counts characters EXCLUDING white spaces)
Bands: Regular 10,000–50,000 ; Position 8,000–40,000.
```bash
# approximate body character count, whitespace removed
detex main.tex | tr -d '[:space:]' | wc -m
```
Report the number every session; warn if outside the chosen band. (This is a proxy;
the official count is on the formatted PDF — keep margin from both limits.)

## 14. "Section done" gate (all must pass)
- Compiles clean; no undefined refs/citations; no overfull/underfull warnings.
- Every `\cite` key present in refs.bib; every figure/table `\ref`'d before it appears.
- Headings cased correctly (section ALL-CAPS, sub/subsub titlecase); first paragraphs
  `\noindent`. Character count reported and within band.
- Double-blind sweep: no name/affiliation/project/grant leak (`\blindtrue`).

## Submission packaging (final)
Submit a PDF for review via PRIMORIS. For camera-ready, zip ONLY the files needed to
compile (main.tex, refs.bib, the 4 Group-1 files, any custom .sty, figures). Remove
examples and previous versions from the compile directory.

## Sources combined to build this skill
- SCITEPRESS LaTeX template `Example.tex` + `SCITEPRESS.sty` (verbatim): preamble &
  package order, title/keywords/abstract macros, ALL-CAPS sections, `\noindent` rule,
  table-above / figure-below captions, `figure*`/`table*` spanning, equation numbering,
  algorithm2e, APA `\cite`, acknowledgements-before / appendix-after placement, "submit
  only compile files" rule, geometry/margins.
- `FormatContentsForAuthors.pdf` (MS Word pack): exact margins (3.3/4.2/2.6/2.6 cm),
  column width 7.5cm / spacing 0.8cm, 10pt body, 9pt captions/refs, no page numbers,
  caption sentence-case + period, page limits (12 full / 8 short, +4 paid extra).
- ICINCO 2026 Guidelines (official): character bands (10k–50k / 8k–40k), double-blind,
  <20% self-citation, AI-disclosure, PDF-via-PRIMORIS submission.
