"""One GPU, several agents: a shared lock file so only one GPU job runs at a time.

The lock lives at X:/navlori-fusion/.gpu_lock (seen as /mnt/x/navlori-fusion/.gpu_lock from WSL),
so every worktree, side_navlori and WSL use the same one.

    python scripts/gpu_lock.py status
    python scripts/gpu_lock.py run --who experiments --what "fusion side_golden" -- python scripts/_train_side_golden.py --epochs 60
    python scripts/gpu_lock.py run --wait --who replay --what "dpvo cache" -- <command>      # wait until free
    python scripts/gpu_lock.py acquire --who side_navlori --what "camera notebook"          # for detached jobs...
    python scripts/gpu_lock.py release --who side_navlori                                    # ...release when they end

`run` releases the lock when the command ends, even if it crashes. Use acquire/release only for jobs
you launch detached, and make the job's last step call `release`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path("/mnt/x/navlori-fusion/.gpu_lock") if os.name != "nt" else Path("X:/navlori-fusion/.gpu_lock")


def read() -> dict | None:
    try:
        return json.loads(LOCK.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"who": "?", "what": "unreadable lock file", "since": "?"}


def describe(info: dict) -> str:
    age = ""
    try:
        mins = (dt.datetime.now() - dt.datetime.fromisoformat(info["since"])).total_seconds() / 60
        age = f", {mins:.0f} min ago"
    except (KeyError, ValueError):
        pass
    return f"held by {info.get('who')} ({info.get('what')}) since {info.get('since')}{age} on {info.get('host')}"


def acquire(who: str, what: str, wait: bool) -> bool:
    info = {"who": who, "what": what, "since": dt.datetime.now().isoformat(timespec="seconds"),
            "host": f"{socket.gethostname()} ({'WSL' if os.name != 'nt' else 'Windows'})", "pid": os.getpid()}
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # atomic: fails if it exists
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(info, f)
            print(f"GPU lock acquired by {who}: {what}", flush=True)
            return True
        except FileExistsError:
            held = read()
            if not wait:
                print(f"GPU busy: {describe(held) if held else 'lock just released, retry'}", flush=True)
                return False
            print(f"waiting for GPU: {describe(held) if held else '...'}", flush=True)
            time.sleep(60)


def release(who: str, force: bool) -> bool:
    held = read()
    if held is None:
        print("GPU lock is not held")
        return True
    if held.get("who") != who and not force:
        print(f"not releasing: {describe(held)} (use --force if that job is really dead)")
        return False
    LOCK.unlink(missing_ok=True)
    print(f"GPU lock released ({held.get('who')}: {held.get('what')})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for name in ("acquire", "run"):
        p = sub.add_parser(name)
        p.add_argument("--who", required=True, help="agent name: dataset / replay / experiments / side_navlori")
        p.add_argument("--what", required=True, help="short job description")
        p.add_argument("--wait", action="store_true", help="wait for the GPU instead of failing")
        if name == "run":
            p.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command to run")
    p = sub.add_parser("release")
    p.add_argument("--who", required=True)
    p.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.cmd == "status":
        held = read()
        print(describe(held) if held else "GPU free")
        return 0
    if a.cmd == "release":
        return 0 if release(a.who, a.force) else 1
    if not acquire(a.who, a.what, a.wait):
        return 3
    if a.cmd == "acquire":
        return 0
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    exe = shutil.which(cmd[0]) or (str(Path(cmd[0]).resolve()) if Path(cmd[0]).exists() else cmd[0])
    cmd = [exe] + cmd[1:]
    try:
        return subprocess.call(cmd)
    finally:
        release(a.who, force=False)


if __name__ == "__main__":
    sys.exit(main())
