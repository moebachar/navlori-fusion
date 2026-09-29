import re
import os
import shutil
import sys
import numpy as np

def run_script(path, repls):
    src = open(path).read()
    for pat, rep in repls:
        src, n = re.subn(pat, rep, src, count=1)
        if n != 1:
            print(f"FAILED to patch {pat} in {path}")
            sys.exit(1)
    g = {"__name__": "__main__", "__file__": path}
    exec(compile(src, path, "exec"), g)

runs = [
    ("0917_152255", "run4"),
    ("0917_152704", "run5"),
    ("0917_151700", "run6"),
    ("0917_151405", "run7"),
    ("0917_150450", "run8"),
]

for raw_name, run_name in runs:
    print(f"=== Processing {raw_name} -> {run_name} ===")
    BAG = f"/mnt/x/side_navlori/data/navlori_{raw_name}"
    RUN = f"/mnt/x/side_navlori/data/staging/{raw_name}"
    FINAL_RUN = f"/mnt/x/side_navlori/data/{run_name}"
    ROOT_DATA = f"/root/navlori/data/{run_name}"
    TMP_CAM = f"/root/navlori/_tmp/{raw_name}_camera"

    if not os.path.exists(BAG):
        print(f"Bag {BAG} not found, skipping...")
        continue

    print("Exporting calib...")
    run_script("/mnt/x/side_navlori/data/scripts/export_calib.py", [
        (r'BAG = r".*?"', f'BAG = r"{BAG}"'),
        (r'OUT = r".*?"', f'OUT = r"{RUN}/calib"')
    ])

    print("Exporting wifi...")
    run_script("/mnt/x/side_navlori/data/scripts/export_wifi.py", [
        (r'BAGDIR = Path\(r".*?"\)', f'BAGDIR = Path(r"{BAG}")'),
        (r'OUTDIR = Path\(r".*?"\)', f'OUTDIR = Path(r"{RUN}/wifi")')
    ])

    print("Exporting camera...")
    os.makedirs(TMP_CAM, exist_ok=True)
    run_script("/mnt/x/side_navlori/data/scripts/export_camera.py", [
        (r'BAGDIR = Path\(r".*?"\)', f'BAGDIR = Path(r"{BAG}")'),
        (r'OUTROOT = Path\(r".*?"\)', f'OUTROOT = Path(r"{TMP_CAM}")')
    ])
    print("Moving camera from tmp to staging...")
    if os.path.exists(f"{RUN}/camera"):
        shutil.rmtree(f"{RUN}/camera")
    shutil.move(f"{TMP_CAM}/camera", f"{RUN}/camera")

    print(f"Promoting staging/{raw_name} to {run_name}...")
    if os.path.exists(FINAL_RUN):
        shutil.rmtree(FINAL_RUN)
    shutil.move(RUN, FINAL_RUN)

    print("Writing README...")
    readme = f"""# side_navlori dataset - {run_name} ({raw_name})

- Date: 2026-09-17
- Camera resolution: 640x480 RGB888 at 30 fps
- /cmd_vel is absent.
"""
    if raw_name == "0917_150450":
        readme += "\nNote: The ground-truth for this run is noisier (p95 ~39mm) than the others.\n"
    
    with open(f"{FINAL_RUN}/README.md", "w") as f:
        f.write(readme)

    print("Updating camera intrinsics to 640x480...")
    ref = dict(
        fx=322.0704122808738, fy=320.8673986158544, cx=199.2680620421962, cy=155.2533082600705,
        D=[0.1639958233797625, -0.271840030972792, 0.001055841660100477, -0.00166555973740089, 0.0],
    )
    sx = 640 / 410.0
    sy = 480 / 308.0
    intr = f"""# NOMINAL camera intrinsics for /camera/image_raw (640x480)
image_width: 640
image_height: 480
camera_name: raspicam_v2_640x480_nominal
camera_matrix:
  rows: 3
  cols: 3
  data: [{ref['fx']*sx:.10g}, 0, {ref['cx']*sx:.10g}, 0, {ref['fy']*sy:.10g}, {ref['cy']*sy:.10g}, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{', '.join(f"{d:.10g}" for d in ref['D'])}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
fov_deg_approx: {{horizontal: {2*np.degrees(np.arctan(640/(2*ref['fx']*sx))):.1f}, vertical: {2*np.degrees(np.arctan(480/(2*ref['fy']*sy))):.1f}}}
"""
    with open(f"{FINAL_RUN}/calib/camera_intrinsics_nominal.yaml", "w") as f:
        f.write(intr)

    print(f"Copying to /root/navlori/data/{run_name}...")
    if os.path.exists(ROOT_DATA):
        shutil.rmtree(ROOT_DATA)
    shutil.copytree(FINAL_RUN, ROOT_DATA)

    print("Deleting raw bag...")
    if os.path.exists(BAG):
        shutil.rmtree(BAG)
