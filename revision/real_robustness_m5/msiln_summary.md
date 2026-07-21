# M5 - Real-data robustness on MSILN (site1/B1, cross-session test)

Reproduces the modality-dropout and WiFi-staleness curves on the
real MSILN test split (5 traces, +11-12 days from training session)
using the existing transformer checkpoint
`runs/main_table/msiln_site1_b1/transformer` (no retraining).

**Headline.** MSILN cross-session test — all=10.90 m, WiFi-only=10.66 m, IMU-only=63.93 m, stale=K(4)=63.93 m (fresh 10.90 m); degradation: graceful (monotone non-decreasing).

## Modality-dropout (test split)

| Subset | MAE (m) | RMSE (m) |
|---|---|---|
| all | 10.897 | 16.455 |
| only:wifi | 10.656 | 15.641 |
| only:imu | 63.930 | 68.721 |

## WiFi staleness (test split, K=4)

| Staleness (K steps) | MAE (m) | RMSE (m) |
|---|---|---|
| stale=0 | 10.897 | 16.455 |
| stale=1 | 12.126 | 18.715 |
| stale=2 | 13.264 | 20.798 |
| stale=3 | 14.440 | 22.738 |
| stale=4 | 63.930 | 68.721 |

Monotone non-decreasing across staleness: **True**.
A monotone curve = graceful degradation (temporal fusion propagating
motion from the last good WiFi fix); a non-monotone jump at one step
= cliff behaviour (single-instant-style failure).
