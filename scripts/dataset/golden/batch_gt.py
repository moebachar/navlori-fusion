#!/usr/bin/env python
# Batch: for each recorded bag, export lidar + telemetry and build the lidar-SLAM
# ground truth -> staging/<name>/ground_truth/gt_overview.png. Camera/wifi export is
# deferred until the user picks which runs to keep. Each exporter's hardcoded path
# constants are patched in-memory then exec'd (they were written with fixed paths).
import os, re, glob, time, traceback
os.environ["MPLBACKEND"] = "Agg"
DATA = "/mnt/x/side_navlori/data"
SCRIPTS = DATA + "/scripts"

bags = sorted(glob.glob(DATA + "/navlori_*"))
def short(b):
    n = os.path.basename(b)
    return "big" if n == "navlori_big" else n.replace("navlori_", "")

def run_script(fname, repls):
    src = open(os.path.join(SCRIPTS, fname)).read()
    for pat, rep in repls:
        src, k = re.subn(pat, rep, src, count=1)
        if k != 1:
            raise RuntimeError(f"patch pattern not found in {fname}: {pat}")
    g = {"__name__": "__main__", "__file__": os.path.join(SCRIPTS, fname)}
    exec(compile(src, fname, "exec"), g)

print(f"found {len(bags)} bags", flush=True)
results = {}
for b in bags:
    name = short(b); run = f"{DATA}/staging/{name}"
    os.makedirs(run, exist_ok=True)
    print(f"\n===== {name}  <-  {os.path.basename(b)} =====", flush=True)
    t0 = time.time()
    try:
        print("  [1/3] lidar export ...", flush=True)
        run_script("export_lidar.py", [
            (r'BAGDIR = Path\(r".*?"\)', f'BAGDIR = Path(r"{b}")'),
            (r'OUTDIR = Path\(r".*?"\)', f'OUTDIR = Path(r"{run}/lidar")')])
        print("  [2/3] telemetry export ...", flush=True)
        run_script("export_telemetry.py", [
            (r'BAGDIR = Path\(r".*?"\)', f'BAGDIR = Path(r"{b}")'),
            (r'OUTROOT = Path\(r".*?"\)', f'OUTROOT = Path(r"{run}")')])
        print("  [3/3] SLAM ground truth ...", flush=True)
        run_script("build_ground_truth.py", [
            (r'ROOT = r".*?"', f'ROOT = r"{run}"')])
        dt = time.time() - t0
        results[name] = f"OK ({dt:.0f}s)"
        print(f"  DONE {name} in {dt:.0f}s", flush=True)
    except Exception:
        results[name] = "FAILED"
        print(f"  !!! FAILED {name}", flush=True)
        traceback.print_exc()

print("\n==== SUMMARY ====", flush=True)
for k, v in results.items():
    print(f"  {k:10s} {v}", flush=True)
print("BATCH_GT_DONE", flush=True)
