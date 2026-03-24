# NavLoRI-Fusion — Agent Context

## What This Project Is

Indoor localization research project using a TIAGO++ robot in Webots simulation.
Predicts (x, y) position by fusing 4 modalities: WiFi RSSI, Vision, IMU, Odometry.
Author: Mohamed Bachar, CESI LINEACT.

## Current Phase: Data Collection (C0)

The ML pipeline (stages A-E) is scaffolded but not yet implemented. Right now:
- **Simulation** is the active workstream — collecting multi-modal sensor data
- 30 paths defined in `src/simulation/controllers/async_collector/paths.json`
- DWA obstacle avoidance is implemented and working
- Robot arm width is measured dynamically after `tuck_arms()` for accurate collision radius
- Paths corrected with 0.855m clearance from walls (0.55 × 1.1 + 0.25m extra)
- Data collection runs in 3 batches: 0-9, 10-19, 20-29 (edit `BATCH_START`/`BATCH_END` in async_collector.py)
- Data saves to `data/async_collection/path_XX/` with per-modality CSVs + camera PNGs

## Critical Rules

1. **NEVER push to GitHub directly** — always give the user git commands to run
2. **After every dev task**: update requirements.txt, .gitignore, README as needed, then give user git commands
3. **No WSL/bash scripts** — this is a Windows machine, all scripts must be PowerShell
4. **Webots needs Parsec** — cameras return NULL in SSH sessions (no GPU context). Must launch Webots in a real desktop session via Parsec
5. **GitHub via HTTPS** — remote is `https://github.com/moebachar/navlori-fusion.git` (SSH deploy key was for old repo only)

## Key Technical Details

### Simulation Controller
- `src/simulation/controllers/async_collector/async_collector.py` — main controller
- Uses embedded controller mode (not `<extern>`) with `runtime.ini` pointing to project venv
- `supervisor TRUE` on the TIAGO++ node for ground truth + arm position reading
- InfluxDB is optional: guarded by `if influx and influx.ready:`

### DWA Planner
- `dwa_planner.py` — adapted from PythonRobotics (MIT license)
- Uses depth camera for obstacle detection
- Robot radius set dynamically from measured arm width + safety margin

### Path Safety
- `fix_paths.py` — pushes waypoints away from inflated wall polygons, inserts midpoints
- `viz_paths.py` — generates overview + grid PNGs in `viz/` directory
- Wall geometry is defined as rotated rectangles extracted from the .wbt file
- Safety zone = wall polygon inflated by `ROBOT_RADIUS * SAFETY_COEFF + EXTRA_MARGIN` = 0.855m

### Sensor Rates
| Modality | Rate | Sim Steps |
|----------|------|-----------|
| IMU | ~31 Hz | 1 (32ms) |
| Odometry | ~15 Hz | 2 |
| Ground Truth | ~10 Hz | 3 |
| WiFi | ~1 Hz | 31 |
| Camera | ~0.5 Hz | 62 |

## Environment

- **GPU**: Quadro P4000 8GB, PyTorch 2.4.1+cu124 (sm_61)
- **Python**: 3.11, single venv at `.venv/`
- **Webots**: R2025a, world file: `src/simulation/worlds/Tiago++'s world.wbt`
- **Services**: InfluxDB (port 8086) + Grafana (port 3000), start with `scripts/services.ps1`
- **Remote**: VS Code SSH via Tailscline (100.126.253.37), Parsec for Webots GUI
- **DVC**: for data versioning (`dvc add data/async_collection` after each batch)
- **Hydra**: config composition in `configs/`
- **MLflow**: local file-based in `mlruns/`
- **pre-commit**: ruff + format + trailing whitespace + large file check

## What's Next (in priority order)

1. **Run fix_paths.py** with EXTRA_MARGIN=0.25, regenerate viz to verify no edges cross safety zone
2. **Run data collection** — 30 paths in 3 batches via Parsec + Webots
3. **DVC add** collected data after each batch
4. **Data exploration** — notebook at `notebooks/data_exploration.ipynb` (kernel: NavLoRI Fusion)
5. **Stage A implementation** — per-modality encoders in `src/pipeline/encoders/`
6. Continue with stages B → E

## Old Repo

The old `navlori` repo (same machine at `C:\Users\Administrateur\navlori`) has legacy code in `models/` and the original `fusion/` directory. It's kept for reference but all new work happens in `navlori-fusion`.
