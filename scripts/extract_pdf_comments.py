"""Extract reviewer annotations from a marked-up PDF.

Workflow: the author reads the compiled paper PDF, highlights text and adds
sticky-note / highlight comments in any reader (Edge, Acrobat, Foxit), then
saves the annotated PDF. This script pulls every annotation out in reading
order, pairing each comment with the text it is anchored to, so the result
can be worked through one item at a time.

Usage:
    .venv\\Scripts\\python.exe scripts/extract_pdf_comments.py path/to/annotated.pdf
    .venv\\Scripts\\python.exe scripts/extract_pdf_comments.py path/to/annotated.pdf -o comments.md

Handles Highlight / Underline / StrikeOut / Squiggly (text-anchored markup),
plus Text (sticky note) and FreeText (typed box). For markup annotations the
underlying paper text is recovered from the annotation quad-points.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

# annotation type code -> short label (fitz.PDF_ANNOT_*)
MARKUP = {8: "Highlight", 9: "Underline", 10: "Squiggly", 11: "StrikeOut"}
NOTE = {0: "Text", 2: "FreeText", 1: "Link"}


def _quad_text(page: "fitz.Page", annot: "fitz.Annot", words: list) -> str:
    """Recover the paper text sitting under a markup annotation.

    Markup annotations store their region as quad-points (4 points per
    covered text line). A word (from ``page.get_text('words')``) belongs to
    the annotation when its centre falls inside any quad. Matching by word
    centre keeps reading order and avoids the duplication that padded clip
    rects cause on multi-line / multi-column highlights.
    """
    verts = annot.vertices or []
    quads = []
    for i in range(0, len(verts) - 3, 4):
        pts = verts[i : i + 4]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        quads.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))
    if not quads:
        quads = [annot.rect]
    picked = []
    for (x0, y0, x1, y1, w, *_rest) in words:  # words are in reading order
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(q.x0 <= cx <= q.x1 and q.y0 <= cy <= q.y1 for q in quads):
            picked.append(w)
    text = " ".join(picked).strip()
    # rejoin words split by an end-of-line hyphen ("car- ries" -> "carries");
    # leaves real hyphens ("set-transformer", "WiFi-Net") untouched (no space)
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", text)


def extract(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    items: list[dict] = []
    for pno in range(doc.page_count):
        page = doc[pno]
        annots = list(page.annots() or [])
        if not annots:
            continue
        words = page.get_text("words")  # fetched once per page, reused
        for annot in annots:
            t = annot.type[0]
            if t == 1:  # skip plain hyperlinks
                continue
            info = annot.info
            kind = MARKUP.get(t) or NOTE.get(t) or annot.type[1]
            anchored = _quad_text(page, annot, words) if t in MARKUP else ""
            items.append(
                {
                    "page": pno + 1,
                    "y": round(annot.rect.y0, 1),
                    "x": round(annot.rect.x0, 1),
                    "kind": kind,
                    "author": (info.get("title") or "").strip(),
                    "comment": (info.get("content") or "").strip(),
                    "anchored": anchored,
                }
            )
    doc.close()
    # reading order: page, then top-to-bottom, then left-to-right
    items.sort(key=lambda d: (d["page"], d["y"], d["x"]))
    return items


def render(items: list[dict]) -> str:
    if not items:
        return "No annotations found in the PDF.\n"
    out: list[str] = [f"# {len(items)} comment(s)\n"]
    cur_page = None
    for n, it in enumerate(items, 1):
        if it["page"] != cur_page:
            cur_page = it["page"]
            out.append(f"\n## Page {cur_page}\n")
        head = f"**[{n}] {it['kind']}**"
        if it["author"]:
            head += f" — {it['author']}"
        out.append(head)
        if it["anchored"]:
            out.append(f"> {it['anchored']}")
        if it["comment"]:
            out.append(f"\n**Comment:** {it['comment']}")
        else:
            out.append("\n*(highlight only, no text comment)*")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path, help="path to the annotated PDF")
    ap.add_argument("-o", "--out", type=Path, help="write markdown here instead of stdout")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: file not found: {args.pdf}", file=sys.stderr)
        return 2

    items = extract(args.pdf)
    md = render(items)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {len(items)} comment(s) -> {args.out}")
    else:
        # Windows console is cp1252; write bytes safely
        sys.stdout.buffer.write(md.encode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
