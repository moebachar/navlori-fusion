# Working on NavLoRI with several agents

Four agents work on the project at the same time. Read this file, then your own brief.

| Agent | Works in | Branch | Brief |
|---|---|---|---|
| **dataset** (TurtleBot3 dataset) | `X:\navlori-fusion\.worktrees\dataset` | `agent/dataset` | [dataset.md](dataset.md) |
| **replay** (Webots replay dataset) | `X:\navlori-fusion\.worktrees\replay` | `agent/replay` | [replay.md](replay.md) |
| **experiments** (training and benchmarks) | `X:\navlori-fusion\.worktrees\experiments` | `agent/experiments` | [experiments.md](experiments.md) |
| **side_navlori** (per-sensor notebooks) | `X:\side_navlori` (its own repo) | `main` | `X:\side_navlori\docs\AGENT_BRIEF.md` |

`X:\navlori-fusion` itself stays on `main` and is only used to merge. Nobody develops there.

## The three project folders (there are no others)

- **`X:\navlori-fusion`**: all project code. GitHub `moebachar/navlori-fusion`.
- **`X:\navlori-data`**: the data vault, git + DVC. GitHub `moebachar/navlori-data` (pointers only).
  - `datasets/`: every training collection. Each worktree's `data\` is a link to it.
  - `robot/`: TurtleBot3 runs with ground truth.
  - `raw/`: rosbags.
  - `archive/`: old material, read-only.
- **`X:\side_navlori`**: the sandbox with the per-sensor notebooks. `data\` links to `X:\navlori-data\robot`.

Do not create new top-level folders on X:. Put scratch files in your worktree's `runs/` (git-ignored) or the session scratchpad.

## Ownership: edit your own area

| Area | Owner |
|---|---|
| `scripts/dataset/**`, `scripts/convert_side_golden.py`, `configs/data/side_golden.yaml`; vault `robot/`, `raw/`, `datasets/side_golden/` | dataset |
| `src/simulation/**`, the replay scripts (`replay_site.py`, `run_replay.py`, `verify_replay.py`, `package_run.py`, `make_world.py`, `build_*world*.py`, `build_all_sites.py`, `stage_drive.py`, `pretreat_dilate.py`, `render_replay_video.py`, `convert_iln20_floor.py`), `configs/data/iln20_*`; vault `datasets/replay_runs/`, `datasets/iln20*/` | replay |
| `src/pipeline/**`, `configs/stage_a/`, `configs/stage_c/`, `scripts/_train_*`, `scripts/_eval_*`, `scripts/eval_*`, `colab/`, `notebooks/` | experiments |
| `README.md`, `CLAUDE.md`, `pyproject.toml`, `docs/` | shared: small edits only, say so in the commit message |

Everyone reads the whole repo and every dataset. If you need a change in someone else's area, keep it minimal, commit it separately, and name the area in the message (`touches experiments: add wheel columns to ODOM_COLS`). `src/pipeline` is used by everybody: never change its function signatures or defaults silently.

## Git routine

In your worktree, on your branch:

```bash
git fetch origin
git rebase main                 # pick up what others merged; do this often
# ... work, commit small and often ...
git push origin agent/<you>     # your branch is your backup
```

When a piece works and is tested, merge it into main (fast-forward only), from your worktree:

```bash
git rebase main
git -C X:/navlori-fusion merge --ff-only agent/<you>
git -C X:/navlori-fusion push origin main
```

If the fast-forward fails, someone merged in between: `git rebase main` again and retry. Never force-push `main`. Never use a bare `git stash`, because the stash stack is shared between worktrees.

## Data routine

Change data only in the vault folders you own. After changing a tracked folder:

```powershell
cd X:\navlori-data
git pull --rebase
X:\navlori-fusion\.venv\Scripts\dvc.exe add robot\golden_run_7      # the folder you changed
git add robot\golden_run_7.dvc; git commit -m "golden_run_7: <why>"; git push
```

Add new datasets as `datasets/<name>/` plus `dvc add`. The vault README explains the layout.

## One GPU: always take the lock

The machine has a single GTX 1080 (8 GB), and it also drives the display. Two GPU jobs at once froze the whole PC on 2026-09-28. Every GPU job goes through the lock:

```bash
python scripts/gpu_lock.py status
python scripts/gpu_lock.py run --who <you> --what "<job>" -- <command>           # releases when the command ends
python scripts/gpu_lock.py run --wait --who <you> --what "<job>" -- <command>    # queue behind the current job
```

- From WSL, use `/root/navlori/venv/bin/python /mnt/x/navlori-fusion/scripts/gpu_lock.py ...`.
- For detached jobs: `acquire` before, and make `release --who <you>` the job's last step.
- CPU-only work doesn't need the lock.
- Keep batch sizes modest, and stop a job that heads toward 8 GB of GPU memory.

## Environments

- **Windows:** `X:\navlori-fusion\.venv`. Run scripts from your worktree root; scripts put their own repo root first on `sys.path`, so they use your branch's code.
- **WSL:** distro `Ubuntu-WSL2`, venv `/root/navlori/venv`. Used for the dataset pipeline, the side_navlori notebooks and Docker/Kalibr.
- **Pretrained weights:** each worktree's `runs/_weights` links to the shared `X:\navlori-fusion\runs\_weights`.
