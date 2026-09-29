#!/usr/bin/env python
# Golden runs: export camera frames (native path) + detect AprilTag36h11 tags + unit-size PnP.
# Tag physical size is applied later (PnP translation scales linearly with tag size).
import os, re, sys, json, time, numpy as np, pandas as pd, cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as Rot

RUNS = sys.argv[1:] or ["golden_run_7", "golden_run_10", "golden_run_11"]
DATA = "/mnt/x/side_navlori/data"; SCRIPTS = DATA + "/scripts"; CAMOUT = "/root/navlori/golden_cam"
STRIDE = 2                                   # detect on every 2nd frame (15 Hz)
K = np.array([[1275.6918, 0, 338.6633], [0, 1274.0334, 291.8436], [0, 0, 1.0]])
Dist = np.array([0.110053, -0.002819, 0.009096, 0.008142])
VALID_IDS = set(range(26))
ts = get_typestore(Stores.ROS2_HUMBLE)

def run_script(fname, repls):
    src = open(os.path.join(SCRIPTS, fname)).read()
    for pat, rep in repls:
        src, k = re.subn(pat, rep, src, count=1); assert k == 1, pat
    exec(compile(src, fname, "exec"), {"__name__": "__main__", "__file__": fname})

def static_tf(bag):
    st = {}
    with Reader(bag) as r:
        for conn, t, raw in r.messages():
            if conn.topic == "/tf_static":
                m = ts.deserialize_cdr(raw, conn.msgtype)
                for tr in m.transforms:
                    tl, q = tr.transform.translation, tr.transform.rotation
                    st[(tr.header.frame_id, tr.child_frame_id)] = (np.array([tl.x, tl.y, tl.z]), np.array([q.x, q.y, q.z, q.w]))
                break
    return st

def compose(st, chain):
    p, r = np.zeros(3), Rot.identity()
    for key in chain:
        xyz, q = st[key]; p = p + r.apply(xyz); r = r * Rot.from_quat(q)
    return p, r

det_params = cv2.aruco.DetectorParameters()
det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), det_params)
OBJ = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], dtype=np.float64)  # unit tag, TL TR BR BL

for run in RUNS:
    bag = f"{DATA}/{run}"; out = f"{CAMOUT}/{run}"; t0 = time.time()
    print(f"\n===== {run} =====", flush=True)
    # 1) static extrinsics: base_footprint -> camera optical frame
    st = static_tf(bag)
    print("  static tf frames:", sorted({c for _, c in st}), flush=True)
    chain = [("base_footprint", "base_link"), ("base_link", "camera_link"),
             ("camera_link", "camera_rgb_frame"), ("camera_rgb_frame", "camera_rgb_optical_frame")]
    p_bc, r_bc = compose(st, chain)
    os.makedirs(out, exist_ok=True)
    json.dump({"t_base_cam": p_bc.tolist(), "q_base_cam_xyzw": r_bc.as_quat().tolist()}, open(f"{out}/extrinsics_cam.json", "w"))
    print(f"  camera optical in base_footprint: t={np.round(p_bc,4)}  rpy(deg)={np.round(r_bc.as_euler('xyz',degrees=True),2)}", flush=True)
    # 2) camera export (skip if done)
    if not os.path.exists(f"{out}/camera/camera.csv"):
        run_script("export_camera.py", [(r'BAGDIR = Path\(r".*?"\)', f'BAGDIR = Path(r"{bag}")'),
                                        (r'OUTROOT = Path\(r".*?"\)', f'OUTROOT = Path(r"{out}")')])
    cam = pd.read_csv(f"{out}/camera/camera.csv")
    print(f"  camera frames: {len(cam)}  ({time.time()-t0:.0f}s)", flush=True)
    # 3) detect tags
    rows = []
    for i in range(0, len(cam), STRIDE):
        img = cv2.imread(f"{out}/camera/images/{cam.filename.iloc[i]}", cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        corners, ids, _ = det.detectMarkers(img)
        if ids is None: continue
        for c, tid in zip(corners, ids.ravel()):
            tid = int(tid)
            if tid not in VALID_IDS: continue
            c = c.reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(OBJ, c, K, Dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok: continue
            proj, _ = cv2.projectPoints(OBJ, rvec, tvec, K, Dist)
            rep = float(np.sqrt(((proj.reshape(4, 2) - c) ** 2).sum(1)).mean())
            side = float(np.mean([np.linalg.norm(c[j] - c[(j + 1) % 4]) for j in range(4)]))
            rows.append(dict(t_ns=int(cam.t_ns.iloc[i]), frame=i, tag_id=tid, side_px=side, reproj_px=rep,
                             tx_u=float(tvec[0]), ty_u=float(tvec[1]), tz_u=float(tvec[2]),
                             rx=float(rvec[0]), ry=float(rvec[1]), rz=float(rvec[2]),
                             **{f"c{j}{a}": float(c[j, k]) for j in range(4) for k, a in enumerate("xy")}))
        if i % 2000 == 0:
            print(f"    frame {i}/{len(cam)}  detections so far {len(rows)}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/tag_detections.csv", index=False)
    if len(df):
        s = df.groupby("tag_id").agg(n=("t_ns", "size"), side_px_max=("side_px", "max")).reset_index()
        print(f"  {len(df)} detections, {df.tag_id.nunique()} distinct tags: " +
              ", ".join(f"#{r.tag_id}x{r.n}" for r in s.itertuples()), flush=True)
    print(f"  DONE {run} in {time.time()-t0:.0f}s", flush=True)
print("TAGS_DETECT_DONE", flush=True)
