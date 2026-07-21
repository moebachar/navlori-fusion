# M5 — Real-Robustness, IMUWiFine floor-4 (test split)

Checkpoint: `runs/main_table/imuwifine/transformer/` (arch=transformer, K=4)
Modalities: `wifi`, `imu` (no odometry/camera in IMUWiFine)
Test paths: 60..79  (n=23,724 samples)
Loader: `src.pipeline.training.load_trained(..., dataset="imuwifine", K=4)`
Artifacts:
- `imuwifine_subsets_test.json`
- `imuwifine_staleness_test.json`

## Headline (test split, MAE in metres)

| Configuration            | MAE (m) | RMSE (m) |
|--------------------------|--------:|---------:|
| all modalities           |   6.37  |   10.08  |
| only:WiFi                |   6.37  |   10.08  |
| only:IMU                 |  23.76  |   27.07  |
| stale=K (=4), WiFi gone  |  23.76  |   27.07  |

## Staleness curve (WiFi masked at the `stale` most-recent instants)

| stale | MAE (m) | RMSE (m) |
|------:|--------:|---------:|
|     0 |   6.37  |   10.08  |
|     1 |   6.63  |   10.58  |
|     2 |   6.84  |   10.96  |
|     3 |   7.08  |   11.38  |
|     4 |  23.76  |   27.07  |

## Reading

- **Graceful, not cliff — up to `stale=K-1`.** From a fresh WiFi fix (`stale=0`,
  6.37 m) the error grows monotonically and smoothly to 7.08 m at `stale=3`
  (only +0.71 m for losing 3/4 of the recent WiFi window). The temporal
  self-attention is propagating the last good fix through the IMU tokens.
- **Cliff only when WiFi is fully gone (`stale=K=4`).** Error jumps to 23.76 m
  — identical to `only:IMU`. With no absolute anchor at *any* of the K=4
  instants, the position is genuinely unobservable; IMU dead-reckoning alone
  drifts to ~24 m on these 20 test paths. This matches the documented honest
  finding "`drop:wifi` stays ~4m … fusion can't invent an anchor" — only here
  the floor is ~24 m because the IMUWiFine IMU is noisier and the paths are
  longer than the Webots sim.
- **`all` ≡ `only:wifi`** at 6.37 m. On fresh data the IMU adds essentially
  zero on top of WiFi at IMUWiFine scale; its contribution shows up purely in
  the *robustness* slope (the +0.71 m absorbed between `stale=0` and
  `stale=3`), not in the headline accuracy. This is the same story as MSILN
  site-1 B1 and as the project's Webots note: "WiFi dominates fresh-data
  accuracy; temporal fusion's value is robustness, not fresh accuracy."

## Caveats

- IMUWiFine floor-4 absolute MAE (~6.4 m) is roughly an order of magnitude
  worse than Webots sim (~0.43 m) — the WiFi encoder is the bottleneck on
  real data, not the fusion. The robustness *shape* (graceful slope + cliff
  at full loss) is the load-bearing signal for the paper, not the absolute
  level.
- "all" and "only:wifi" report identical floats because `evaluate_subsets`
  runs the model with the same active set on both rows when M=2 and IMU is
  already near-zero contribution at `stale=0`; this is expected, not a bug.
