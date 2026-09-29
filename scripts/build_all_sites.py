"""Batch-build clean replay worlds for EVERY site/floor of the ILN 2.0 dump.

For each valid floor (has floor_info.json + geojson_map.json +
path_data_files/) this runs the full gated chain via
    replay_site.py prepare --site S --floor F --out-name NAME --no-stage
i.e. convert -> build world (--clean --robot rig) -> dedup -> ceiling ->
non-collidable audit -> cleanliness gate -> alignment gate. No replay is
staged (there is ONE shared replay_config.json -- stage the floor you
actually open in Webots).

Naming: iln20_<site8>_<floor>, EXCEPT when several sites share the same
8-char prefix (41 collision groups in the dump; the ids differ at the END,
so even 12-char prefixes collide) -- those get iln20_<site8><last4>_<floor>.
The three floors converted before this scheme existed (F1/F2/B1) keep their
8-char names: an existing data/iln20_<site8>_<floor> whose meta/dataset.json
site_id matches grandfathers the short name.

Resumable: floors whose world .wbt already exists are skipped (--rebuild to
force). Progress + per-floor status land incrementally in
    src/simulation/worlds/iln20_batch_manifest.json
Failures are recorded with the tail of their output and NEVER stop the batch.

Scale (2026-07-22 dump): 204 sites, 981 valid floors, ~27k traces,
~12 GB converted output, several hours single-threaded.

Usage (PowerShell, from repo root):
    .venv\\Scripts\\python.exe scripts\\build_all_sites.py [--limit 5]
        [--rebuild] [--min-free-gb 15] [--site <id>]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time as pytime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "data" / "iln20" / "data"
# converted source floors live in a sub-dir (2026-09-07 reorg)
DATA_ROOT = REPO_ROOT / "data" / "iln20_converted"
WORLDS_DIR = REPO_ROOT / "src" / "simulation" / "worlds"
SCRIPTS = REPO_ROOT / "scripts"
MANIFEST = WORLDS_DIR / "iln20_batch_manifest.json"


def enumerate_floors() -> list[tuple[str, str, int]]:
    out = []
    for site in sorted(d for d in RAW_ROOT.iterdir() if d.is_dir()):
        for fl in sorted(d for d in site.iterdir() if d.is_dir()):
            if not ((fl / "floor_info.json").is_file()
                    and (fl / "geojson_map.json").is_file()
                    and (fl / "path_data_files").is_dir()):
                continue
            n = sum(1 for _ in (fl / "path_data_files").iterdir())
            if n:
                out.append((site.name, fl.name, n))
    return out


def pick_name(site: str, floor: str, colliding_prefixes: set[str]) -> str:
    short = f"iln20_{site[:8]}_{floor}"
    # grandfather pre-batch conversions (F1/F2/B1) that own the short name
    dsj = DATA_ROOT / short / "meta" / "dataset.json"
    if dsj.is_file():
        try:
            if json.loads(dsj.read_text(encoding="utf-8")).get("site_id") == site:
                return short
        except json.JSONDecodeError:
            pass
    if site[:8] in colliding_prefixes:
        return f"iln20_{site[:8]}{site[-4:]}_{floor}"
    return short


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N floors (0 = all; use for smoke tests)")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild floors whose world already exists")
    ap.add_argument("--min-free-gb", type=float, default=15.0,
                    help="stop gracefully when X: free space drops below this")
    ap.add_argument("--site", default=None, help="only this site id")
    ap.add_argument("--shrink-shops", type=float, default=0.75,
                    help="corridor-widening shop shrink in metres, forwarded "
                         "to the world builder (default 0.75 -- see "
                         "replay_site.py prepare)")
    args = ap.parse_args()

    floors = enumerate_floors()
    if args.site:
        floors = [f for f in floors if f[0] == args.site]
    by_prefix: dict[str, set[str]] = {}
    for s, _, _ in floors:
        by_prefix.setdefault(s[:8], set()).add(s)
    colliding = {p for p, ss in by_prefix.items() if len(ss) > 1}

    names = {}
    for s, fl, n in floors:
        names[(s, fl)] = pick_name(s, fl, colliding)
    if len(set(names.values())) != len(names):
        sys.exit("naming scheme produced duplicates -- aborting before any work")

    print(f"[batch] {len(floors)} floors across "
          f"{len({s for s, _, _ in floors})} sites "
          f"({len(colliding)} colliding 8-char prefixes)", flush=True)

    entries = []
    counts = {"ok": 0, "exists": 0, "failed": 0}
    t0 = pytime.time()
    for i, (site, floor, n_traces) in enumerate(floors):
        if args.limit and (counts["ok"] + counts["failed"]) >= args.limit:
            print(f"[batch] --limit {args.limit} reached", flush=True)
            break
        free_gb = shutil.disk_usage(str(REPO_ROOT)).free / 2**30
        if free_gb < args.min_free_gb:
            print(f"[batch] STOP: only {free_gb:.1f} GB free "
                  f"(< {args.min_free_gb}). Resume later -- existing worlds "
                  f"are skipped automatically.", flush=True)
            break

        name = names[(site, floor)]
        world = WORLDS_DIR / f"{name}.wbt"
        e = {"site": site, "floor": floor, "name": name, "n_traces": n_traces}
        if world.is_file() and not args.rebuild:
            e["status"] = "exists"
            counts["exists"] += 1
        else:
            t_w = pytime.time()
            r = subprocess.run(
                [sys.executable, str(SCRIPTS / "replay_site.py"), "prepare",
                 "--site", site, "--floor", floor, "--out-name", name,
                 "--no-stage", "--shrink-shops", str(args.shrink_shops)],
                cwd=REPO_ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace")
            e["seconds"] = round(pytime.time() - t_w, 1)
            if r.returncode == 0:
                e["status"] = "ok"
                counts["ok"] += 1
            else:
                e["status"] = "failed"
                tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
                e["tail"] = tail[-4:]
                counts["failed"] += 1
        entries.append(e)
        print(f"[batch] {i + 1}/{len(floors)} {name}: {e['status']}"
              + (f" ({e.get('seconds', 0)}s, {n_traces} traces)"
                 if e["status"] != "exists" else "")
              + (f"  << {e['tail'][-1] if e.get('tail') else ''}"
                 if e["status"] == "failed" else ""),
              flush=True)

        MANIFEST.write_text(json.dumps({
            "generated": pytime.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(pytime.time() - t0, 1),
            "totals": dict(counts, planned=len(floors), done=len(entries)),
            "floors": entries,
        }, indent=2), encoding="utf-8")

    print(f"\n[batch] DONE in {(pytime.time() - t0) / 60:.1f} min: "
          f"{counts['ok']} built, {counts['exists']} already existed, "
          f"{counts['failed']} failed  -> {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
