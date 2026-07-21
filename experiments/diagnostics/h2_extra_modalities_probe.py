"""H2: How much magnetometer / iBeacon data is in MSILN site1/B1 raw traces?

Counts TYPE_* row frequencies across a sample of raw .txt files for site1/B1.
"""
from __future__ import annotations

import collections
import pathlib
import random
import sys

ROOT = pathlib.Path(r"x:/navlori-fusion/data/iln20/data/5a0546857ecc773753327266/B1/path_data_files")
files = sorted(ROOT.glob("*.txt"))
print(f"total traces: {len(files)}")

random.seed(0)
sample = random.sample(files, min(40, len(files)))

agg = collections.Counter()
file_with = collections.Counter()
n_with_magn = 0
n_with_beacon = 0
n_with_magn_uncali = 0

magn_rates = []
beacon_rates = []
wifi_rates = []

for f in sample:
    types = collections.Counter()
    t_lo, t_hi = None, None
    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                ts = int(parts[0])
                t_lo = ts if t_lo is None else min(t_lo, ts)
                t_hi = ts if t_hi is None else max(t_hi, ts)
            except ValueError:
                pass
            types[parts[1]] += 1
            agg[parts[1]] += 1
    dur_s = (t_hi - t_lo) / 1000.0 if t_lo and t_hi else 0
    has_magn = types.get("TYPE_MAGNETIC_FIELD", 0) > 0
    has_magn_unc = types.get("TYPE_MAGNETIC_FIELD_UNCALIBRATED", 0) > 0
    has_beacon = types.get("TYPE_BEACON", 0) > 0
    if has_magn: n_with_magn += 1
    if has_magn_unc: n_with_magn_uncali += 1
    if has_beacon: n_with_beacon += 1
    file_with["TYPE_MAGNETIC_FIELD"] += has_magn
    file_with["TYPE_MAGNETIC_FIELD_UNCALIBRATED"] += has_magn_unc
    file_with["TYPE_BEACON"] += has_beacon
    if dur_s > 0:
        magn_rates.append(types.get("TYPE_MAGNETIC_FIELD", 0) / dur_s)
        beacon_rates.append(types.get("TYPE_BEACON", 0) / dur_s)
        wifi_rates.append(types.get("TYPE_WIFI", 0) / dur_s)

print(f"\nSample: {len(sample)} traces (random.seed=0)")
print("\nAggregate TYPE_* row counts:")
for k, v in agg.most_common():
    print(f"  {k:42s} {v:>10d}")

print("\nFile presence (out of {} sampled):".format(len(sample)))
print(f"  files with TYPE_MAGNETIC_FIELD:               {n_with_magn:>3d}  ({100.0*n_with_magn/len(sample):.0f}%)")
print(f"  files with TYPE_MAGNETIC_FIELD_UNCALIBRATED:  {n_with_magn_uncali:>3d}  ({100.0*n_with_magn_uncali/len(sample):.0f}%)")
print(f"  files with TYPE_BEACON:                       {n_with_beacon:>3d}  ({100.0*n_with_beacon/len(sample):.0f}%)")

def stats(name, arr):
    if not arr:
        print(f"  {name}: no data")
        return
    arr = sorted(arr)
    n = len(arr)
    mean = sum(arr) / n
    med = arr[n // 2]
    print(f"  {name}: mean={mean:.2f} Hz  median={med:.2f} Hz  min={arr[0]:.2f}  max={arr[-1]:.2f}")

print("\nPer-trace sample rate (Hz):")
stats("magnetometer (TYPE_MAGNETIC_FIELD)", magn_rates)
stats("beacon (TYPE_BEACON)              ", beacon_rates)
stats("wifi (TYPE_WIFI rows, scan rate ~10-30x lower)", wifi_rates)
