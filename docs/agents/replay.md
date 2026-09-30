# Brief: replay agent (Webots replay dataset)

## Mission

Continue the replay pipeline: re-drive real walks from the Microsoft Indoor Location Competition 2.0 (ILC 2.0) data with a TIAGo++ robot inside Webots reconstructions of the same floors. The real WiFi and IMU are kept verbatim; the simulation adds the camera and wheel odometry. The product is 4-sensor datasets in the common per-path format.

## What changed around you (2026-09-29 cleanup)

- **Your work is safe on `main`.** The replay/Vive work that was uncommitted was committed as-is (`bc46bd9` "replay pipeline v2 + Vive GT ..."), together with your 6 earlier local commits, and pushed.
- **You now work in `X:\navlori-fusion\.worktrees\replay` on branch `agent/replay`,** no longer in `X:\navlori-fusion`. See [README.md](README.md) for the git and GPU rules.
- **Data moved to the vault;** `data\` is a link, so paths such as `data/replay_runs/...` and `data/iln20/...` still work.
  - Live folders: `X:\navlori-data\datasets\replay_runs\`, `datasets\iln20_5d27099f_F2_replay\`, `datasets\iln20\` (raw ILC 2.0 dump, 66 GB, not versioned).
  - The v1 replay product moved from `X:\navlori-archive\replay_v1` to `X:\navlori-data\archive\replay_v1`.
- **DVC is no longer inside navlori-fusion.** To version a finished replay dataset: `dvc add datasets\<name>` in `X:\navlori-data` (see the data routine in README.md).
- **Webots and GPU:** Webots rendering and camera capture use the GPU. Take the GPU lock for long replay/render runs, so an experiments training job doesn't start at the same time.

## Where to pick up

Rebuild your state from `git log -- src/simulation scripts/replay_site.py scripts/run_replay.py`, the replay section of `CLAUDE.md`, and your previous session. The last recorded state (2026-07-22):
- the F2 site is staged for a full run;
- the IMU-shaped speed profile (anchored time warp) reaches r = 0.87 speed-vs-IMU on F2;
- package and verify gates are in place.

`configs/data/iln20_5d27099f_F2_replay.yaml` describes the 49-path replay dataset that passes the accuracy gate.

## You own

- `src/simulation/**`;
- the replay scripts listed in README.md;
- `configs/data/iln20_*`;
- vault `datasets/replay_runs/` and `datasets/iln20*/`.
