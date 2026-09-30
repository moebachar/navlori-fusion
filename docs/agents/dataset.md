# Brief: dataset agent (TurtleBot3 dataset)

## Mission

Own the real-robot dataset. Bring the 12 "golden runs" to a state where they can be published as a dataset paper: correct, reproducible from the rosbags with one command, and documented. Keep `side_golden` (the training-format copy) in sync for the other agents.

## What exists

- **Recordings:** 12 golden runs recorded 2026-09-24 on a TurtleBot3 Waffle Pi (ROS 2 Humble) in the CESI building.
  - 25 min and 262 m in total.
  - Camera 640×480 at 30 Hz, IMU/odometry/joints at ~145 Hz, LDS-02 lidar at 9.6 Hz, WiFi about 1 scan per 4 s.
  - 26 AprilTags (tag36h11, 15 cm) at floor-plan (CAD) coordinates.
  - Older recordings `run1..run8` (first campaign) are also in the vault.
- **Vault:**
  - `X:\navlori-data\robot\golden_run_N\`: packaged runs with `camera/ imu/ wheel_odom/ wifi/ lidar/ mag/ ground_truth/ calib/ README.md`.
  - `robot\golden_floorplan\`: floor plan and plan registration.
  - `robot\tags_ground_truth.json`: CAD coordinates of the tags.
  - `robot\calib\kalibr_640x480\`: camera calibration.
  - `robot\staging_golden\`: intermediate pipeline outputs.
  - `raw\golden_bags\`: the 12 rosbags. `raw\calib_cam\`: the calibration bag.
- **Code** (`scripts/dataset/`):
  - `export/`: rosbag → per-sensor files; lidar scan-to-map point-to-line ICP SLAM (`build_ground_truth.py`); the run loader (`load_dataset.py`).
  - `golden/`, the ground-truth pipeline, in order:
    1. `batch_golden.sh`: export and SLAM;
    2. `tags_detect.sh`: OpenCV AprilTag detection + IPPE PnP;
    3. `gyro_bias_lidar.py`;
    4. `tag_ba2.sh`: joint rigid alignment of all runs on the tag network, Cauchy loss, soft CAD prior σ = 10 cm;
    5. `gt_icp2.py`: map matching for runs 2 and 8, which see one tag;
    6. `golden_package.sh`;
    7. `plan_register.py` / `plan_plots.py`.
  - `alternatives/` (pose graph, fused variants) were tested and not kept; `diagnostics/` holds one-off checks.
  - `scripts/convert_side_golden.py` → `data/side_golden` (path id = run − 1; split train 1,4,5,7,9,10,11 · val 3,8 · test 2,6,12).
- **Ground truth:** heading comes from the gyroscope, distance from lidar SLAM, and each run is placed rigidly on the tag network.
  - Held-out accuracy: leave one tag sighting out, re-solve, measure the error there. Median 15 cm (per run 2–43 cm).
  - Walls from different runs coincide within 6–7 cm.
  - Tags only place the runs; they do not bend them. A tag pose graph was tested and did not improve held-out error, because some CAD tag coordinates are off.
- **Environment:** the pipeline runs in WSL (`/root/navlori/venv`, OpenCV 5, rosbags). Camera frames are exported natively to `/root/navlori/golden_cam/` for speed. Kalibr runs in Docker on the native WSL dockerd (`golden/start_dockerd.sh`, image `kalibr`). The converter runs on Windows (`.venv`).

## Backlog, in priority order

1. **Make the pipeline reproducible.**
   - Scripts still hard-code paths from their WSL origin (`/mnt/x/side_navlori/data/...`, `/root/navlori/logs`, `sys.path` hacks). Most resolve through the links, but make them take a config or arguments instead.
   - Add one entry point that goes from the bags to packaged runs.
   - Rerun it end to end, and check the output matches the vault (`dvc status` should show no change beyond expected float noise).
2. **Unit mislabel.** `wheel_odom/joint_states.csv` columns `left_vel_radps` / `right_vel_radps` actually hold m/s (verified: they integrate to wheel travel within 0.2 %). Fix the exporter (`export/export_telemetry.py`) and the docs. **Ask the user before renaming columns in the vault runs** (golden runs and run1..8): it changes published files.
3. **Tag survey follow-up.** CAD coordinates of tags 12, 14 and 6 look off by 53, 39 and 26 cm. When the user supplies re-measured coordinates:
   - update `tags_ground_truth.json`;
   - rerun `tag_ba2` → `gt_icp2` → `golden_package` → `convert_side_golden`;
   - retry the tag-corrected pose graph (`alternatives/gt_posegraph.py`) and keep it only if the held-out error improves.
4. **Dataset documentation for publication:**
   - a datasheet (sensors, rates, frames, extrinsics, calibration, ground-truth method and its measured accuracy, known limitations such as the dead magnetometer);
   - a single `robot/README.md` index;
   - figures.
5. **run1..run8:** decide with the user whether they are re-processed with the golden pipeline or kept as a separate, older subset.

## Rules specific to you

- You own the robot data. The experiments and side_navlori agents read it. Announce any change that alters existing files (commit message plus a line in the vault commit).
- Regenerate `data/side_golden` after any ground-truth change. Keep the split unless the user agrees to change it.
- Use the GPU lock for DPVO or camera-heavy work. Most of your pipeline is CPU-only.
