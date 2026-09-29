#!/usr/bin/env python
# Golden runs: export lidar + telemetry + wifi and build lidar-SLAM ground truth
# -> data/staging_golden/<name>/. Camera export is a separate, later step.
import os, re, glob, time, traceback
os.environ["MPLBACKEND"] = "Agg"
DATA = "/mnt/x/side_navlori/data"
SCRIPTS = DATA + "/scripts"
OUT = DATA + "/staging_golden"

bags = sorted(glob.glob(DATA + "/golden_run_*"), key=lambda p: int(p.rsplit("_", 1)[1]))

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
    name = os.path.basename(b); run = f"{OUT}/{name}"
    os.makedirs(run, exist_ok=True)
    print(f"\n===== {name} =====", flush=True)
    t0 = time.time()
    try:
        print("  [1/4] lidar", flush=True)
        run_script("export_lidar.py", [(r'BAGDIR = Path\(r".*?"\)', f'BAGDIR = Path(r"{b}")'),
                                       (r'OUTDIR = Path\(r".*?"\)', f'OUTDIR = Path(r"{run}/lidar")')])
        print("  [2/4] telemetry", flush=True)
        run_script("export_telemetry.py", [(r'BAGDIR = Path\(r".*?"\)', f'BAGDIR = Path(r"{b}")'),
                                           (r'OUTROOT = Path\(r".*?"\)', f'OUTROOT = Path(r"{run}")')])
        print("  [3/4] wifi", flush=True)
        run_script("export_wifi.py", [(r'BAGDIR = Path\(r".*?"\)', f'BAGDIR = Path(r"{b}")'),
                                      (r'OUTDIR = Path\(r".*?"\)', f'OUTDIR = Path(r"{run}/wifi")')])
        print("  [4/4] SLAM ground truth", flush=True)
        run_script("build_ground_truth.py", [(r'ROOT = r".*?"', f'ROOT = r"{run}"')])
        dt = time.time() - t0; results[name] = f"OK ({dt:.0f}s)"
        print(f"  DONE {name} in {dt:.0f}s", flush=True)
    except Exception:
        results[name] = "FAILED"; print(f"  !!! FAILED {name}", flush=True); traceback.print_exc()

print("\n==== SUMMARY ====", flush=True)
for k, v in results.items():
    print(f"  {k:15s} {v}", flush=True)
print("BATCH_GOLDEN_DONE", flush=True)
