"""One command per MSILN site/floor: raw traces -> converted dataset ->
clean Webots world (walls, shop floor-signs, decorations, ceiling,
non-collidable) -> alignment + cleanliness gates -> staged replay ->
(after the Webots run) batch verification.

Phases (subcommands):
  prepare : everything headless, up to "open Webots and press Play".
            Chain: convert_iln20_floor (if the dataset dir is missing)
            -> build_iln20_webots_world --clean --robot rig
            -> dedup_wbt_walls -> add_ceiling_to_wbt
            -> make_worlds_noncollidable
            -> cleanliness gate (no debug DEFs / Tiago++ / physics in .wbt)
            -> diagnose_world_alignment --gate
            -> run_replay (writes replay_config.json, resume-aware).
  verify  : after the Webots run -- verify_replay --all; writes
            <replay>/replay_manifest.json; exit 0 only if every path passes.
  package : (iteration 3) splits/meta/config + FusionDataModule smoke.

The ONLY manual step is the Webots run itself (CLAUDE.md rule 4: cameras
render NULL outside a real desktop session -- use Parsec).

Data layout (2026-09-07 reorg): converted source floors live under
data/iln20_converted/<name>/ (regenerable, gitignored); finished replay
datasets land at data/<name>_replay/ next to the other DVC datasets.

Usage (PowerShell, from repo root):
    # pilot on the already-converted F2 dataset, paths 0-4:
    .venv\\Scripts\\python.exe scripts\\replay_site.py prepare \\
        --dataset data/iln20_converted/iln20_5d27099f_F2 --paths 0-4

    # a fresh site/floor straight from the raw dump:
    .venv\\Scripts\\python.exe scripts\\replay_site.py prepare \\
        --site 5a0546857ecc773753327266 --floor B1

    # after the Webots run:
    .venv\\Scripts\\python.exe scripts\\replay_site.py verify \\
        --dataset data/iln20_converted/iln20_5d27099f_F2 \\
        --replay  data/iln20_5d27099f_F2_replay
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
SCRIPTS = REPO_ROOT / "scripts"
WORLDS_DIR = REPO_ROOT / "src" / "simulation" / "worlds"
# Converted source floors live in a sub-dir so data/'s top level stays a
# short list of real datasets (2026-09-07 reorg). Replay OUTPUT datasets
# (the products) stay at data/ top level next to the other DVC datasets.
CONVERTED_ROOT = REPO_ROOT / "data" / "iln20_converted"


def run_step(title: str, argv: list[str]) -> None:
    print(f"\n{'='*64}\n  [{title}]\n  $ {' '.join(str(a) for a in argv)}\n{'='*64}",
          flush=True)
    r = subprocess.run([sys.executable] + [str(a) for a in argv], cwd=REPO_ROOT)
    if r.returncode != 0:
        sys.exit(f"\n[replay_site] STOP: step '{title}' failed "
                 f"(exit {r.returncode}). Fix and re-run prepare.")


def cleanliness_gate(world: Path) -> None:
    """The replay camera must never see debug geometry, and the world must
    contain zero physics. Deterministic .wbt text check -- stronger than any
    pixel forensic (what does not exist cannot be rendered)."""
    text = world.read_text(encoding="utf-8")
    forbidden = ["DEF MARKER_", "DEF WP_", "DEF PS_", "ROBOT_PATHS",
                 "Tiago++", "boundingObject", "Physics"]
    required = ["DEF REPLAY_RIG Robot", 'controller "replay_collector"',
                'supervisor TRUE']
    print(f"\n{'='*64}\n  [gate: world cleanliness]  {world.name}\n{'='*64}")
    ok = True
    for token in forbidden:
        n = text.count(token)
        print(f"  [{'PASS' if n == 0 else 'FAIL'}] zero '{token}': {n} found")
        ok &= n == 0
    for token in required:
        present = token in text
        print(f"  [{'PASS' if present else 'FAIL'}] contains '{token}'")
        ok &= present
    if not ok:
        sys.exit("\n[replay_site] STOP: world cleanliness gate failed.")


def all_path_ids(dataset: Path) -> list[int]:
    return sorted(int(d.name[5:]) for d in dataset.iterdir()
                  if d.is_dir() and d.name.startswith("path_")
                  and d.name[5:].isdigit())


def cmd_prepare(args) -> None:
    # ── Resolve dataset ──
    if args.dataset:
        dataset = Path(args.dataset).resolve()
    else:
        if not (args.site and args.floor):
            sys.exit("prepare needs --dataset OR (--site AND --floor)")
        dataset = CONVERTED_ROOT / (
            args.out_name or f"iln20_{args.site[:8]}_{args.floor}")

    # ── 1. Convert (only when missing) ──
    if not dataset.is_dir():
        if not (args.site and args.floor):
            sys.exit(f"dataset dir not found ({dataset}) and no --site/--floor "
                     f"given to convert it from the raw dump")
        conv_argv = [SCRIPTS / "convert_iln20_floor.py",
                     "--site", args.site, "--floor", args.floor,
                     "--out-root", CONVERTED_ROOT]
        if args.out_name:
            conv_argv += ["--out-name", args.out_name]
        run_step("convert", conv_argv)
        if not dataset.is_dir():
            sys.exit(f"convert finished but {dataset} still missing")
    else:
        print(f"[replay_site] dataset present: {dataset} (convert skipped)")

    name = args.out_name or dataset.name
    world = WORLDS_DIR / f"{name}.wbt"
    # Products (replay datasets) go to data/ TOP level, next to the other
    # DVC datasets -- never into iln20_converted/ with the sources.
    output = Path(args.output).resolve() if args.output \
        else REPO_ROOT / "data" / f"{dataset.name}_replay"
    paths_spec = args.paths or ",".join(str(i) for i in all_path_ids(dataset))
    if not paths_spec:
        sys.exit(f"no path_XX dirs found in {dataset}")

    # ── 2. Build the world (clean, rig robot) ──
    build_argv = [SCRIPTS / "build_iln20_webots_world.py",
                  "--dataset-dir", dataset, "--clean", "--robot", "rig",
                  "--out-name", name, "--seed", str(args.seed)]
    if args.shrink_shops > 0:
        build_argv += ["--shrink-shops", str(args.shrink_shops)]
    run_step("build world", build_argv)

    # ── 3-5. Wall dedup, ceiling, non-collidable audit ──
    run_step("dedup walls", [SCRIPTS / "dedup_wbt_walls.py", "--world", world])
    run_step("ceiling", [SCRIPTS / "add_ceiling_to_wbt.py", "--world", world,
                         "--force"])
    run_step("non-collidable audit",
             [SCRIPTS / "make_worlds_noncollidable.py", "--world", world])

    # ── 6. Cleanliness gate ──
    cleanliness_gate(world)

    # ── 7. Alignment gate ──
    run_step("alignment gate", [SCRIPTS / "diagnose_world_alignment.py",
                                "--world", world, "--dataset", dataset,
                                "--gate"])

    # ── 8. Stage the replay (skipped in batch world-building: there is ONE
    #      shared replay_config.json; stage the floor you actually open) ──
    if args.no_stage:
        print(f"\n[replay_site] PREPARE (no-stage) COMPLETE -- world ready: "
              f"{world}\n  stage later with: replay_site.py prepare "
              f"--dataset {dataset.relative_to(REPO_ROOT)}")
        return
    stage_argv = [SCRIPTS / "run_replay.py", "--world", world,
                  "--dataset", dataset, "--output", output,
                  "--paths", paths_spec]
    if args.fresh:
        stage_argv.append("--fresh")
    run_step("stage replay", stage_argv)

    n_paths = len(paths_spec.split(",")) if "," in paths_spec else None
    print(f"""
{'='*64}
  PREPARE COMPLETE -- all headless gates green
{'='*64}
  world   : {world}
  dataset : {dataset}
  paths   : {paths_spec if n_paths is None or n_paths <= 12 else f'{n_paths} paths'}
  output  : {output}

  Manual step (Parsec desktop session -- rule 4):
    1. Open Webots, File > Open World: {world}
    2. Press Play. Fast mode (>> button) is fine and faster.
    3. The controller prints per-path progress; interrupted runs RESUME
       (completed paths are skipped via _done.json).

  Afterwards, back here:
    .venv\\Scripts\\python.exe scripts\\replay_site.py verify ^
        --dataset {dataset.relative_to(REPO_ROOT)} --replay {output.relative_to(REPO_ROOT) if output.is_relative_to(REPO_ROOT) else output}
""")


def cmd_verify(args) -> None:
    argv = [SCRIPTS / "verify_replay.py", "--all",
            "--source", args.dataset, "--replay", args.replay]
    if args.paths:
        argv += ["--paths", args.paths]
    run_step("verify replay", argv)
    print("\n[replay_site] verification PASSED -- manifest written. "
          "Next: 'package' (iteration 3).")


def _load_split_ids(src: Path) -> dict[str, list[int]]:
    """train/val/test path ids from the source dataset. Prefers
    splits/*.txt; falls back to configs/data/<source-name>.yaml (the msiln
    case, where splits live only in the training config)."""
    src_splits = src / "splits"
    if src_splits.is_dir():
        out = {}
        for part in ("train", "val", "test"):
            f = src_splits / f"{part}.txt"
            ids = []
            if f.is_file():
                for line in f.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("path_"):
                        ids.append(int(line[5:]))
            out[part] = sorted(ids)
        return out
    cfgp = REPO_ROOT / "configs" / "data" / f"{src.name}.yaml"
    if cfgp.is_file():
        from omegaconf import OmegaConf
        d = OmegaConf.load(cfgp).data.split
        return {"train": sorted(int(i) for i in d.train_paths),
                "val": sorted(int(i) for i in d.val_paths),
                "test": sorted(int(i) for i in d.test_paths)}
    sys.exit(f"no splits found: neither {src_splits} nor {cfgp} exists")


def cmd_package(args) -> None:
    src = Path(args.dataset).resolve()
    rep = Path(args.replay).resolve()
    if not src.is_dir():
        sys.exit(f"source dataset not found: {src}")
    if not rep.is_dir():
        sys.exit(f"replay dir not found: {rep}")

    # ── Gate: a verified manifest is the entry ticket ──
    manifest_p = rep / "replay_manifest.json"
    if not manifest_p.is_file():
        sys.exit(f"{manifest_p} missing -- run 'replay_site.py verify' first")
    man = json.loads(manifest_p.read_text(encoding="utf-8"))
    passed = sorted(int(k[5:]) for k, v in man["paths"].items() if v.get("ok"))
    failed = sorted(int(k[5:]) for k, v in man["paths"].items() if not v.get("ok"))
    if not man.get("all_pass") and not args.allow_partial:
        sys.exit(f"manifest has {len(failed)} failing paths {failed} -- fix "
                 f"them or pass --allow-partial to package only the "
                 f"{len(passed)} passing ones")
    if failed:
        print(f"[package] WARNING: dropping {len(failed)} failed paths from "
              f"all splits: {failed}")

    # ── meta/: copy from source, then stamp replay provenance ──
    meta_out = rep / "meta"
    meta_out.mkdir(exist_ok=True)
    src_meta = src / "meta"
    if src_meta.is_dir():
        for f in src_meta.iterdir():
            if f.is_file():
                shutil.copy2(f, meta_out / f.name)
        print(f"[package] meta/: copied {sum(1 for f in src_meta.iterdir() if f.is_file())} files from source")
    elif (src / "metadata.json").is_file():
        shutil.copy2(src / "metadata.json", meta_out / "source_metadata.json")
        print("[package] meta/: source has no meta/ dir -- copied root "
              "metadata.json as source_metadata.json")

    world_id = "unknown"
    rep_meta_p = rep / "metadata.json"
    if rep_meta_p.is_file():
        world_id = json.loads(rep_meta_p.read_text(encoding="utf-8")) \
            .get("config", {}).get("world_dataset_id", "unknown")

    dsj_p = meta_out / "dataset.json"
    base = json.loads(dsj_p.read_text(encoding="utf-8")) if dsj_p.is_file() else {}
    base["name"] = rep.name
    base["replay"] = {
        "source_dataset": src.name,
        "world": f"{world_id}.wbt",
        "created": pytime.strftime("%Y-%m-%dT%H:%M:%S"),
        "modalities_verbatim": ["wifi", "imu"],
        "modalities_synthesized": ["camera", "odometry", "ground_truth_dense"],
        "n_paths_replayed": len(passed),
        "n_paths_failed": len(failed),
        "failed_paths": failed,
        "verify_thresholds": man.get("thresholds", {}),
    }
    dsj_p.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"[package] meta/dataset.json: replay provenance stamped")

    # ── splits/: source splits intersected with the passing paths ──
    split_ids = _load_split_ids(src)
    pset = set(passed)
    in_any_split = set()
    splits_out = rep / "splits"
    splits_out.mkdir(exist_ok=True)
    kept_ids: dict[str, list[int]] = {}
    for part, ids in split_ids.items():
        in_any_split |= set(ids)
        kept = [i for i in ids if i in pset]
        dropped = [i for i in ids if i not in pset]
        kept_ids[part] = kept
        (splits_out / f"{part}.txt").write_text(
            "".join(f"path_{i:02d}\n" for i in kept), encoding="utf-8")
        msg = f"[package] splits/{part}.txt: {len(kept)} paths"
        if dropped:
            msg += f" (dropped {len(dropped)} not-passing: {dropped})"
        if not kept:
            msg += "  << EMPTY -- training config unusable for this split"
        print(msg)
    orphans = sorted(pset - in_any_split)
    if orphans:
        print(f"[package] note: {len(orphans)} passing paths in no split "
              f"(unused by training): {orphans}")

    # ── configs/data/<name>.yaml (starter; conventions follow msiln) ──
    name = args.config_name or rep.name
    cfg_p = REPO_ROOT / "configs" / "data" / f"{name}.yaml"
    n_aps = base.get("n_bssids", "?")

    def fmt_ids(ids: list[int]) -> str:
        return "[" + ", ".join(str(i) for i in ids) + "]"

    cfg_p.write_text(f"""# {name} -- 4-modality replay dataset (generated by replay_site.py package,
# {pytime.strftime('%Y-%m-%d')}). Source {src.name}: wifi+imu+GT kept verbatim; camera+odometry
# synthesized by Webots replay in {world_id}.wbt (kinematic TIAGO-shell rig,
# pose-anchored, gates in meta/dataset.json:replay.verify_thresholds).
# Preprocessing follows the msiln conventions: wifi_norm raw (M1: whitening
# destroys WiFi; raw mode also disables PCA), world-frame IMU (M4).
# {n_aps} APs. Split = source split minus any failed-replay paths.
data:
  name: {name}
  root: ${{oc.env:NAVLORI_DATA_ROOT,data}}
  collection_dir: {rep.name}
  source: replay
  modalities: [wifi, imu, odom, camera]

  split:
    train_paths: {fmt_ids(kept_ids['train'])}
    val_paths: {fmt_ids(kept_ids['val'])}
    test_paths: {fmt_ids(kept_ids['test'])}

  preprocessing:
    normalize: true
    wifi_norm: raw
    wifi_pca: 128
    wifi_max_stale_s: null
    imu_frame: world

  windows:
    imu: 32
    odom: 16
    wifi: 1
    camera: 1
""", encoding="utf-8")
    print(f"[package] wrote {cfg_p.relative_to(REPO_ROOT)}")

    # ── Per-path 4-panel realtime videos (IMU+odom / camera / WiFi / map) ──
    if args.skip_videos:
        print("[package] videos SKIPPED (--skip-videos)")
    else:
        world_p = WORLDS_DIR / f"{world_id}.wbt"
        vid_argv = [SCRIPTS / "render_replay_video.py", "--replay", rep,
                    "--world", world_p, "--fps", str(args.video_fps)]
        if args.video_paths:
            vid_argv += ["--paths", args.video_paths]
        run_step("render videos", vid_argv)

    # ── FusionDataModule smoke: the dataset must load as 4 modalities ──
    if args.skip_smoke:
        print("[package] smoke SKIPPED (--skip-smoke)")
    else:
        code = (
            "from src.pipeline.fusion.builder import load_config, build_datamodule\n"
            f"cfg = load_config('{name}')\n"
            "dm = build_datamodule(cfg)\n"
            "b = next(iter(dm.train_dataloader()))\n"
            "def shp(v):\n"
            "    return tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__\n"
            "print('[smoke] batch keys:', {k: shp(v) for k, v in b.items()}\n"
            "      if isinstance(b, dict) else [shp(x) for x in b])\n"
            "print('[smoke] train/val/test sizes:', len(dm.train_ds), "
            "len(dm.val_ds), len(dm.test_ds))\n"
            "print('[smoke] OK -- replay dataset is FusionDataModule-consumable')\n"
        )
        run_step("FusionDataModule smoke", ["-c", code])

    print(f"""
{'='*64}
  PACKAGE COMPLETE
{'='*64}
  dataset : {rep}
  config  : {cfg_p.relative_to(REPO_ROOT)}
  paths   : {len(passed)} packaged, {len(failed)} dropped

  Suggested next commands (you run them -- never pushed by me):
    dvc add {rep.relative_to(REPO_ROOT) if rep.is_relative_to(REPO_ROOT) else rep}
    dvc push
    git add {cfg_p.relative_to(REPO_ROOT)} data/{rep.name}.dvc data/.gitignore
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="headless chain up to the Webots run")
    p.add_argument("--dataset", default=None,
                   help="already-converted dataset dir (skips conversion)")
    p.add_argument("--site", default=None, help="raw site id (for conversion)")
    p.add_argument("--floor", default=None, help="raw floor id, e.g. B1/F2")
    p.add_argument("--paths", default=None,
                   help="path ids spec, e.g. '0-4' (default: ALL paths)")
    p.add_argument("--output", default=None,
                   help="replay output dir (default: <dataset>_replay)")
    p.add_argument("--out-name", default=None,
                   help="world stem override (default: dataset dir name; must "
                        "equal it for the controller's world guard)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--shrink-shops", type=float, default=0.75,
                   help="move shop walls this many metres away from the "
                        "corridor (default 0.75 since 2026-07-22: surveyors "
                        "hug walls -- 22.9%% of F2 GT points were within "
                        "0.5 m of a wall). Perimeter walls never move.")
    p.add_argument("--fresh", action="store_true",
                   help="redo paths even if _done.json exists")
    p.add_argument("--no-stage", action="store_true",
                   help="stop after the world gates (build_all_sites.py batch "
                        "mode) -- do not write replay_config.json")
    p.set_defaults(fn=cmd_prepare)

    v = sub.add_parser("verify", help="post-Webots batch gates + manifest")
    v.add_argument("--dataset", required=True, help="source dataset root")
    v.add_argument("--replay", required=True, help="replay output root")
    v.add_argument("--paths", default=None, help="restrict to ids, e.g. '0-4'")
    v.set_defaults(fn=cmd_verify)

    g = sub.add_parser("package",
                       help="post-verify packaging: meta/splits/config yaml "
                            "+ FusionDataModule smoke")
    g.add_argument("--dataset", required=True, help="source dataset root")
    g.add_argument("--replay", required=True, help="verified replay root")
    g.add_argument("--config-name", default=None,
                   help="configs/data/<name>.yaml stem (default: replay dir name)")
    g.add_argument("--allow-partial", action="store_true",
                   help="package even if some paths failed verification "
                        "(they are dropped from every split)")
    g.add_argument("--skip-smoke", action="store_true",
                   help="skip the FusionDataModule load test")
    g.add_argument("--skip-videos", action="store_true",
                   help="skip the per-path 4-panel realtime videos")
    g.add_argument("--video-paths", default=None,
                   help="render videos only for these ids, e.g. '0-4'")
    g.add_argument("--video-fps", type=float, default=10.0,
                   help="video frames per sim second (default 10; rendering "
                        "runs at roughly 1.3x realtime at 10 fps)")
    g.set_defaults(fn=cmd_package)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
