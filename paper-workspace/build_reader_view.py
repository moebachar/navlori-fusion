#!/usr/bin/env python
r"""Build a blind review kit for the paper-reviewer agent.

The reviewer's value is that it does NOT see the authors' private context. This
script produces a self-contained kit in a directory OUTSIDE the repo, so that a
fresh `claude` session launched there never loads the project CLAUDE.md,
scope.md, PAPER_STATE.md, the RESULT_* logs, or the author skills.

Kit contents (default out dir: <repo-parent>/paper-review-kit):
  paper-reader-view.tex   every \input resolved; all `%` comments stripped
                          (they carry insider provenance like `% src: RESULT_xx`).
                          Everything reader-visible is preserved verbatim --
                          including \textcolor TODO scaffolding, which a reviewer
                          SHOULD flag as not-submission-ready.
  refs.bib                copied verbatim, so the format lens can check that every
                          \cite resolves and estimate the self-citation share.
  REVIEW_INSTRUCTIONS.md  the reviewer persona+contract, taken from
                          .claude/agents/paper-reviewer.md (YAML frontmatter
                          stripped) -- single source of truth, no duplication.
  icinco-public-rubric.md the PUBLIC ICINCO + SCITEPRESS review criteria.

No LaTeX toolchain required (pure text). Python 3.11+.

Usage:
  python paper-workspace/build_reader_view.py [--out DIR] [--root TEXFILE]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # paper-workspace/
REPO = HERE.parent                              # repo root
PAPER = REPO / "paper"
AGENT_FILE = REPO / ".claude" / "agents" / "paper-reviewer.md"

COMMENT_RE = re.compile(r"(?<!\\)%.*$")
INPUT_RE = re.compile(r"^\s*\\(?:input|include)\{([^}]+)\}\s*$")


def strip_comment(line: str) -> str:
    """Remove a LaTeX `%` comment to end of line, respecting an escaped \\%."""
    return COMMENT_RE.sub("", line)


def resolve(path: Path, seen: set[Path]) -> list[str]:
    """Recursively inline \\input/\\include, stripping `%` comments as we go."""
    path = path.resolve()
    if path in seen:
        return [f"[[CYCLE: {path.name}]]"]
    seen.add(path)
    if not path.exists():
        rel = path.relative_to(REPO) if REPO in path.parents else path
        return [f"[[MISSING INPUT: {rel} -- this would break compilation]]"]
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = strip_comment(raw)
        m = INPUT_RE.match(stripped)
        if m:
            inc = m.group(1)
            if not inc.endswith(".tex"):
                inc += ".tex"
            out += resolve(PAPER / inc, seen)
        else:
            out.append(stripped.rstrip())
    return out


def collapse_blanks(lines: list[str]) -> list[str]:
    """Collapse runs of 3+ blank lines (often left by comment-only lines) to 1."""
    out: list[str] = []
    blanks = 0
    for ln in lines:
        if ln.strip() == "":
            blanks += 1
            if blanks <= 1:
                out.append("")
        else:
            blanks = 0
            out.append(ln)
    return out


def crude_char_count(tex: str) -> int:
    """Very rough chars-excluding-spaces proxy (no detex available).

    Strips commands and LaTeX punctuation, then counts non-whitespace. The
    official ICINCO count is on the formatted PDF; this only tells you which
    side of the 10k/50k band you are on.
    """
    t = re.sub(r"\\[a-zA-Z]+\*?", "", tex)      # \commands
    t = re.sub(r"[{}\\$&~^_#]", "", t)          # latex punctuation
    t = re.sub(r"\s", "", t)                    # whitespace
    return len(t)


def instructions_from_agent() -> str:
    """Agent file minus its YAML frontmatter -> self-contained kit instructions."""
    if not AGENT_FILE.exists():
        return ("# Review instructions\n\n(.claude/agents/paper-reviewer.md not "
                "found; supply the reviewer persona manually.)\n")
    text = AGENT_FILE.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].lstrip("\n")
    return ("# Review instructions (you are the paper-reviewer; run fully "
            "blind)\n\n" + text)


PUBLIC_RUBRIC = """\
# ICINCO / SCITEPRESS public review criteria

Public information from the ICINCO author guidelines and the SCITEPRESS template.
Nothing here is project-internal -- it is what any reviewer of this venue knows.

## Rating dimensions (score each)
- Relevance
- Originality / Innovation
- Technical / Scientific Quality
- Significance / Impact
- Presentation / Readability

## Reviewer question list (answer yes/no)
- Are the abstract and introduction adequate?
- Are more experimental results needed?
- Is a comparative evaluation present and convincing?
- Is the critical discussion adequate?
- Are the figures adequate and readable?
- Are the conclusions and future work convincing?
- Are the references up to date and appropriate?
- Is the formatting correct?
- Is the English correct?

## Hard rules
- Regular Paper length: 10,000-50,000 characters EXCLUDING white spaces
  (Position Paper: 8,000-40,000). Measured on the formatted text, not pages.
- Self-citations under 20% of the reference list.
- Any AI-generated text must be disclosed in the camera-ready Acknowledgements.
- (Single-blind at this venue: author names may be visible. Do not penalize for
  visible authorship; do penalize visible TODO scaffolding or a placeholder
  abstract.)

## Format rules (SCITEPRESS)
- A4, two columns, 10pt body. Top-level sections ALL-CAPS and numbered;
  sub/subsubsections titlecase. First paragraph after a heading is not indented.
- Captions ABOVE tables, BELOW figures; 9pt; each ends with a period.
- Citations APA author-year via \\cite -> "(Author, year)"; never numeric "[1]".
- Every reference is cited; every figure/table is referenced in the text before
  it appears. No page numbers, headers, or footers.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the blind review kit.")
    ap.add_argument("--out", type=Path, default=REPO.parent / "paper-review-kit",
                    help="kit output dir (default: a sibling of the repo, OUTSIDE it)")
    ap.add_argument("--root", type=Path, default=PAPER / "main.tex",
                    help="root .tex to resolve (default: paper/main.tex)")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"ERROR: root tex not found: {root}", file=sys.stderr)
        return 1

    out_dir: Path = args.out.resolve()
    if REPO == out_dir or REPO in out_dir.parents:
        print(f"WARNING: kit dir {out_dir} is INSIDE the repo; a claude session "
              f"launched there may still load the project CLAUDE.md and break "
              f"blindness. Prefer a path outside the repo.", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)

    body = "\n".join(collapse_blanks(resolve(root, set())))
    reader_view = (
        "% READER VIEW -- generated, do not edit. `%` comments stripped, "
        "\\input resolved.\n"
        "% This approximates what a reader of the compiled PDF sees.\n\n" + body + "\n"
    )
    (out_dir / "paper-reader-view.tex").write_text(reader_view, encoding="utf-8")

    refs = PAPER / "refs.bib"
    if refs.exists():
        shutil.copy2(refs, out_dir / "refs.bib")

    (out_dir / "REVIEW_INSTRUCTIONS.md").write_text(
        instructions_from_agent(), encoding="utf-8")
    (out_dir / "icinco-public-rubric.md").write_text(PUBLIC_RUBRIC, encoding="utf-8")

    chars = crude_char_count(body)
    missing = body.count("[[MISSING INPUT")
    band = "OK" if 10_000 <= chars <= 50_000 else (
        "BELOW 10k band" if chars < 10_000 else "ABOVE 50k band")

    print("Blind review kit written to:")
    print(f"  {out_dir}")
    print()
    print(f"  paper-reader-view.tex   ({len(body.splitlines())} lines)")
    print(f"  refs.bib                ({'copied' if refs.exists() else 'MISSING'})")
    print(f"  REVIEW_INSTRUCTIONS.md")
    print(f"  icinco-public-rubric.md")
    print()
    print(f"  crude char count (excl. spaces): ~{chars:,}  [{band}]")
    print(f"  (rough proxy; the official count is on the formatted PDF)")
    if missing:
        print(f"  WARNING: {missing} unresolved \\input -- would break compilation")
    print()
    print("Run the reviewer FULLY BLIND (no project context loads):")
    print(f"  cd {out_dir}")
    print('  claude  "You are the paper-reviewer. Read REVIEW_INSTRUCTIONS.md, '
          'then review paper-reader-view.tex against icinco-public-rubric.md. '
          'Phase 1 only: verdict, no fixes."')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
