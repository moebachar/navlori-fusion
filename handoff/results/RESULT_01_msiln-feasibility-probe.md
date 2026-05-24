# Result 01 — msiln-feasibility-probe

## TL;DR

**GO.** The vendored `indoor-location-competition-20` "sample" turns out
to be 2.1 GB of real competition data (sites 1+2, 14 floors, 1095
traces, 4 distinct survey days each). Site1 alone is 1.4 GB / 642
traces and is fully self-contained — **no Kaggle or HuggingFace
download was needed**. All four feasibility axes pass, with one
caveat on (a):

| axis | target | observed | verdict |
|---|---|---|---|
| (a) WiFi scan rate ≥ 0.5 Hz | denser than IPIN (~0.15–0.25 Hz) | 0.43–0.51 Hz (B1 ≥ 0.5; F1–F4 ≈ 0.45) | **PASS w/ caveat** — ~2× IPIN, but borderline. Stale-WiFi pressure reduced not eliminated. |
| (b) per-site disk < 5 GB | yes | site1 = 1.4 GB | **PASS** |
| (c) schema → async_collection | natural mapping | vendored `read_data_file` returns per-modality arrays | **PASS** — straightforward |
| (d) cross-session train/test | yes | B1/F2/F3 each span 4 distinct days (12-day window) | **PASS** |

Recommended first site for conversion (PLAN_02): **`site1/B1`** — most
traces (160), broadest day spread (2019-11-24 → 2019-12-06), highest
WiFi scan rate (0.51 Hz median).

## Numbers

| step | acceptance | observed | pass? |
|---|---|---|---|
| 1. clone msiln20 starter | `data/` dir + `*.txt` present | 1145 files, `data/site1` + `data/site2` each with multi-floor `path_data_files/` | ✅ |
| 2. acquire ≥1 site | site or sample present | github sample = 2.1 GB real data — no HF/Kaggle needed | ✅ |
| 3. inspector report written | `runs/overnight/iter_01/inspect_msiln.txt` with all required fields | written (5 floors × {GT, IMU, WiFi, sessions}); 642 traces inspected | ✅ |
| 4. 4 yes/no answers w/ evidence | all four answered | see TL;DR table above; evidence below | ✅ |
| 5. GO/NO-GO recommendation | one paragraph | GO; rationale below | ✅ |

## What was changed

- `scripts/inspect_msiln.py`: new diagnostic probe (joins
  `scripts/inspect_*.py` family). Reads one site/floor of msiln20,
  reports GT/IMU/WiFi/session metrics. Imports vendored `io_f.py` —
  no upstream modification (Demand #3 honoured).
- `runs/overnight/iter_01/inspect_msiln.txt`: full inspection report
  (gitignored under `runs/`; pasted verbatim below).
- `handoff/results/RESULT_01_msiln-feasibility-probe.md`: this file.
- `handoff/STATE.md`: iteration log row appended.

**Also committing in iter 01** (one-time housekeeping — first iteration
of the run): `handoff/PROTOCOL.md`, `handoff/STATE.md`,
`handoff/ENGINEER_LOOP.md`, `handoff/SCIENTIST_LOOP.md`,
`handoff/SCIENTIST_BRIEF.md`, `handoff/plans/PLAN_01_*.md`,
`handoff/results/.gitkeep`. These pre-existed in the working tree
untracked from the night's setup and need to be in git so the user
sees the protocol in the morning. Flagged here so the diff is not a
surprise.

## What was reverted (if any)

None.

## Logs

- `runs/overnight/iter_01/inspect_msiln.txt` — full report (gitignored)

### Verbatim inspect_msiln.txt (human summary only — full JSON in file)

```
==============================================================================
Microsoft ILN 2.0 schema inspection -- site1
==============================================================================
floors scanned: 5    total traces: 642    distinct day span: 4 dates

--- B1 (160 paths) ---
  GT extent : 229.8 m x 146.2 m   rate ~ 0.00 Hz  (1034 waypoints)
  step len  : median 553.0 cm   p90 1155.4 cm   max 2007.4 cm
  IMU acc   : 50.0 Hz mean   NaN/Inf 0/795366 samples
  WiFi      : scan 0.51 Hz median   127 APs/scan mean   1452 unique BSSIDs   RSSI parse 100.0%
  Sessions  : 4 distinct days  [2019-11-24 ... 2019-12-06]

--- F1 (120 paths) ---
  GT extent : 191.8 m x 157.6 m   rate ~ 0.01 Hz  (975 waypoints)
  step len  : median 612.9 cm   p90 1179.4 cm   max 1496.1 cm
  IMU acc   : 50.3 Hz mean   NaN/Inf 0/872898 samples
  WiFi      : scan 0.43 Hz median   379 APs/scan mean   2524 unique BSSIDs   RSSI parse 100.0%
  Sessions  : 2 distinct days  [2019-11-24 ... 2019-11-25]

--- F2 (123 paths) ---
  GT extent : 158.7 m x 164.5 m   rate ~ 0.00 Hz  (1049 waypoints)
  step len  : median 656.3 cm   p90 1161.9 cm   max 1669.8 cm
  IMU acc   : 50.1 Hz mean   NaN/Inf 0/1147659 samples
  WiFi      : scan 0.47 Hz median   312 APs/scan mean   2272 unique BSSIDs   RSSI parse 100.0%
  Sessions  : 4 distinct days  [2019-11-24 ... 2019-12-05]

--- F3 (117 paths) ---
  GT extent : 154.2 m x 162.0 m   rate ~ 0.00 Hz  (1012 waypoints)
  step len  : median 605.4 cm   p90 1180.6 cm   max 1486.3 cm
  IMU acc   : 49.7 Hz mean   NaN/Inf 0/1426383 samples
  WiFi      : scan 0.46 Hz median   299 APs/scan mean   2209 unique BSSIDs   RSSI parse 100.0%
  Sessions  : 4 distinct days  [2019-11-24 ... 2019-12-05]

--- F4 (122 paths) ---
  GT extent : 163.0 m x 160.9 m   rate ~ 0.00 Hz  (1042 waypoints)
  step len  : median 635.4 cm   p90 1124.9 cm   max 1474.8 cm
  IMU acc   : 50.3 Hz mean   NaN/Inf 0/1070424 samples
  WiFi      : scan 0.47 Hz median   283 APs/scan mean   1982 unique BSSIDs   RSSI parse 100.0%
  Sessions  : 2 distinct days  [2019-11-25 ... 2019-12-05]
```

### Note on "GT rate"

The 0.00–0.01 Hz figure is misleading and is an artefact of how the
inspector concatenates timestamps from independent traces. GT is
**waypoint anchors**, ~6–7 per trace (median ~6 m apart on the
walking path; not "step length" between footsteps — it's the
inter-waypoint distance along the surveyor's path). The Kaggle
competition format linearly interpolates position between
consecutive waypoints in time, treating IMU/WiFi at each intermediate
sample as having a derived (x,y) label. Our pipeline already
predicts per-timestamp positions, so the same interpolation works
for us. **This is structurally identical to RoNIN's waypoint-anchor
GT, not to IPIN's per-second tracker output.**

## Comparability evidence (the four questions)

(a) **WiFi scan rate ≥ 0.5 Hz?** — PASS with caveat.
  Median scan rate: B1 0.51 Hz, F1 0.43 Hz, F2 0.47 Hz, F3 0.46 Hz,
  F4 0.47 Hz. All ≥ 0.43 Hz, ~2× IPIN's 0.15–0.25 Hz. Borderline
  on the 0.5 Hz target — B1 the only floor that clears it cleanly.
  This still meaningfully reduces stale-WiFi pressure (the IPIN
  ceiling driver per autopsy Probe 2.1) but does not eliminate it.

(b) **Per-site < 5 GB?** — PASS, by a wide margin.
  Site1 = 1.4 GB / 642 traces. Site2 = 0.7 GB / 453 traces.
  Total `data/` dir = 2.1 GB. Single floor (B1, the recommended
  first conversion target) ≈ 280 MB.

(c) **Schema → async_collection?** — PASS.
  Vendored `io_f.read_data_file` already groups one .txt into a
  `ReadData(acce, acce_uncali, gyro, gyro_uncali, magn, magn_uncali,
  ahrs, wifi, ibeacon, waypoint)` dataclass of per-modality arrays.
  Conversion to our async_collection format (per-modality CSVs +
  GT CSV per trace) is mechanical: parse with `read_data_file`,
  emit one CSV per modality with the same timestamp + value cols.
  WiFi rows already include BSSID + RSSI; matches what
  `Anchor2Vec` expects. **No camera modality** (it's a phone-held
  dataset) — we already run IMU+WiFi-only configs on IPIN/RoNIN, so
  this is not new.

(d) **Cross-session train/test splits?** — PASS.
  Three of the five site1 floors have ≥ 3 distinct survey days:
  - B1: 4 days, 12-day span (Nov-24 → Dec-06), paths split 73 / 75 / 7 / 5
  - F2: 4 days, 12-day span, paths split 13 / 91 / 10 / 9
  - F3: 4 days, 12-day span, paths split 82 / 18 / 10 / 7
  Cleanest train/test: **train on earlier day (Nov-24/25), test on
  later day (Dec-04/05/06)** — same site, different surveyor session,
  no chunk leakage. This is the publishable cross-session axis from
  the scientist brief.

## GO / NO-GO recommendation

**GO.** Convert `site1/B1` first. Justification:

- **160 traces** = largest pool on site1, gives ~130 train / ~30 test
  with the Nov-24/25 vs Dec-04/05/06 cross-session split.
- **0.51 Hz WiFi scan rate** — the only floor that cleanly clears the
  0.5 Hz target; least stale-WiFi pressure on first attempt.
- **1452 unique BSSIDs** — large enough vocabulary for `Anchor2Vec`
  without being so large it dominates training (cf. IPIN floor 0 had
  ~4k BSSIDs in similar form).
- **229.8 m × 146.2 m extent** — large enough to make MAE meaningful
  (≥ tens of metres possible); not so large that conformal radius
  becomes uninformative.
- **Disk cost: 280 MB.** Conversion + training fits trivially in the
  iteration budget.

PLAN_02 should be: write `scripts/convert_msiln.py` (mirrors
`scripts/convert_ipin2024.py` / `convert_ronin.py`), targetting
`site1/B1`, output `data/msiln_site1_b1/`. Acceptance: shape parity
with `imuwifine_floor4` async_collection layout, plus the splits CSV
encoding the Nov-vs-Dec cross-session train/test.

## Open questions for scientist

**Q1 (priority).** GT is **anchor-only waypoints, ~6 m apart along the
path**, not per-step trajectory. The Kaggle competition interpolates
linearly between consecutive waypoints in time (assumes constant
pace). RoNIN takes the same approach. **Confirm**: should
`convert_msiln.py` (a) do the same linear time-interp at GT-rate
matching IMU rate, or (b) keep waypoints as anchors and add a
training mask so only waypoint-timestamps incur a loss? Option (a) is
the SOTA-comparable choice; option (b) is theoretically cleaner but
diverges from the leaderboard methodology.

## Wall-clock

- Clone + inspect (steps 1–3): ~4 minutes (00:23 → 00:27 local)
- Including this writeup: ~9 minutes total iteration time.
