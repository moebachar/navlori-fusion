"""Make an iln20 replay world fully non-collidable except the floor.

The worlds built by build_iln20_webots_world.py use custom Solids without
boundingObject everywhere EXCEPT the corner FireExtinguisher instances — a
stock Webots PROTO that carries Physics + boundingObject internally, so the
robot can hit it (and knock it over) in tight corridors. This script replaces
every `DEF CORNER_* FireExtinguisher { ... }` block with a visual-only
look-alike Solid at the same position, then audits the file for anything else
that could collide.

Usage (PowerShell, from repo root):
    .venv\\Scripts\\python.exe scripts\\make_worlds_noncollidable.py \\
        --world src/simulation/worlds/iln20_5d27099f_F2.wbt
    # or all three at once:
    .venv\\Scripts\\python.exe scripts\\make_worlds_noncollidable.py --all
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORLDS_DIR = REPO_ROOT / "src" / "simulation" / "worlds"

EXT_HEAD_RE = re.compile(r"DEF (CORNER_\d+) FireExtinguisher \{")
TRANSLATION_RE = re.compile(
    r"translation\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)")
NAME_RE = re.compile(r'name\s+"([^"]+)"')
# Physics-capable PROTOs that could appear in these worlds. Walls/decor are
# custom Solids (no boundingObject); the robot and Floor are the only nodes
# allowed to have physics.
RISKY_PROTOS = ("FireExtinguisher", "CardboardBox", "WoodenChair", "Table",
                "Sofa", "PottedTree", "OilBarrel")


def _block_end(text: str, brace_open_idx: int) -> int:
    depth = 0
    for i in range(brace_open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


def fmt_visual_extinguisher(def_name: str, x: float, y: float, name: str) -> str:
    """Visual-only fire-extinguisher look-alike: red body + dark valve.
    No boundingObject, no Physics -> the robot passes through it."""
    return (
        f"DEF {def_name} Solid {{\n"
        f"  translation {x:.4f} {y:.4f} 0.30\n"
        f"  name \"{name}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.82 0.08 0.08 "
        f"roughness 0.35 metalness 0.25 }}\n"
        f"      geometry Cylinder {{ radius 0.085 height 0.55 }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0 0.31 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.18 0.18 0.18 "
        f"roughness 0.5 metalness 0.5 }}\n"
        f"      geometry Cylinder {{ radius 0.035 height 0.12 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


def process(world: Path, dry_run: bool) -> bool:
    text = world.read_text(encoding="utf-8")
    replaced = 0
    out = []
    cursor = 0
    for head in EXT_HEAD_RE.finditer(text):
        end = _block_end(text, head.end() - 1)
        block = text[head.start():end]
        mt = TRANSLATION_RE.search(block)
        mn = NAME_RE.search(block)
        if not mt:
            continue
        out.append(text[cursor:head.start()])
        out.append(fmt_visual_extinguisher(
            head.group(1), float(mt.group(1)), float(mt.group(2)),
            mn.group(1) if mn else head.group(1).lower()))
        cursor = end
        if cursor < len(text) and text[cursor] == "\n":
            cursor += 1
        replaced += 1
    out.append(text[cursor:])
    new_text = "".join(out)

    # drop the now-unused EXTERNPROTO declaration
    new_text = re.sub(r'EXTERNPROTO "[^"]*FireExtinguisher\.proto"\n', "",
                      new_text)

    print(f"[{world.name}] replaced {replaced} FireExtinguisher instances")

    # ── audit: anything left that could collide? ──
    issues = []
    if new_text.count("boundingObject"):
        issues.append(f"{new_text.count('boundingObject')}x boundingObject")
    for proto in RISKY_PROTOS:
        n = len(re.findall(rf"\b{proto} \{{", new_text))
        if n:
            issues.append(f"{n}x {proto} PROTO")
    if issues:
        print(f"[{world.name}]   REMAINING COLLIDABLES: {', '.join(issues)}")
    else:
        print(f"[{world.name}]   audit clean: no boundingObject / physics "
              f"PROTO outside TIAGO+Floor")

    if replaced == 0:
        return not issues
    if dry_run:
        print(f"[{world.name}]   dry-run: no changes written")
        return not issues
    bak = world.with_suffix(world.suffix + ".bak-physics")
    shutil.copy2(world, bak)
    world.write_text(new_text, encoding="utf-8")
    print(f"[{world.name}]   written ({len(text):,} -> {len(new_text):,} "
          f"bytes), backup {bak.name}")
    return not issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", help="one .wbt to process")
    ap.add_argument("--all", action="store_true",
                    help="process every iln20_*.wbt in the worlds dir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        worlds = sorted(WORLDS_DIR.glob("iln20_*.wbt"))
    elif args.world:
        worlds = [Path(args.world).resolve()]
    else:
        sys.exit("pass --world <wbt> or --all")

    ok = True
    for w in worlds:
        if not w.is_file():
            sys.exit(f"world not found: {w}")
        ok &= process(w, args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
