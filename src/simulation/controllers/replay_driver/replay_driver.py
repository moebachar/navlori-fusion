"""NavLoRI replay_driver -- pipeline v2, SCRIPT 2 of 3 (the controller).

Drives the REAL Tiago++ (physics untouched) along each real path:
  geometry = polyline through the real waypoint presses
  speed    = IMU-integrated per-segment velocity profile (velocity_profile.py)
  control  = real wheel velocities: feed-forward v(t) + schedule-error catch-up
             + carrot-point steering. NO pose anchoring while driving --
             the supervisor only places the robot at each path's start.
The world guarantees safety the other way round: walls/decor are visual-only,
only the floor is solid, so the robot can pass through anything but never
falls.

Outputs per path -> <output_dir>/path_XX/:
  ground_truth.csv  actual driven pose @10 Hz (+ schedule/lateral error)
  odometry.csv      REAL wheel-encoder odometry @15 Hz (+ slip noise k=0.01)
  camera.csv + camera/*.png                   @5 Hz
  flight_log.csv    tracking diagnostics @4 Hz
  metadata.json     incl. per-press hit errors (time is exact by schedule;
                    position error is the honest residual of real driving)
  _done.json        ONLY on genuine completion (Webots quit => abort, redo)

Guards (ported from v1, all battle-tested): world-name, build-stamp vs the
LIVE scene, debug-geometry scene check, resume via _done markers, abort on
quit, crash log.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time as pytime
from pathlib import Path

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
_VSP = r"x:\navlori-fusion\.venv\Lib\site-packages"
if os.path.isdir(_VSP) and _VSP not in sys.path:
    sys.path.insert(0, _VSP)
sys.path.insert(0, CONTROLLER_DIR)

from controller import Supervisor  # noqa: E402
from path_geometry import PathGeometry  # noqa: E402
from velocity_profile import build_profile  # noqa: E402

WHEEL_RADIUS = 0.0985
AXLE = 0.4044
K_SLIP = 0.01               # encoder slip-noise (v1 decision: ~1% drift)
SIGMA0 = 0.0002
# V_CAP / MAX_WHEEL are set at runtime from cfg["v_cap"] (Option C: the
# stock 0.62 m/s wheel cap can't match real walking speed).

ARM_TUCK = {
    "torso_lift_joint": 0.0,
    "arm_left_1_joint": 0.20, "arm_left_2_joint": -1.10,
    "arm_left_3_joint": -0.20, "arm_left_4_joint": 1.94,
    "arm_left_5_joint": -1.57, "arm_left_6_joint": 1.37, "arm_left_7_joint": 0.0,
    "arm_right_1_joint": 0.20, "arm_right_2_joint": -1.10,
    "arm_right_3_joint": 0.20, "arm_right_4_joint": 1.94,
    "arm_right_5_joint": -1.57, "arm_right_6_joint": 1.37, "arm_right_7_joint": 0.0,
}


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class CSVOut:
    def __init__(self, path: Path, cols: list[str]):
        self.path = path
        self.cols = cols if cols[0] == "sim_time" else ["sim_time"] + cols
        self.f = None
        self.w = None
        self.n = 0

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.cols, extrasaction="ignore")
        self.w.writeheader()

    def write(self, row):
        self.w.writerow(row)
        self.n += 1

    def close(self):
        if self.f:
            self.f.close()
            print(f"  {self.path.name:>18s}: {self.n} rows")


def load_cfg() -> dict:
    p = Path(CONTROLLER_DIR) / "replay_driver_config.json"
    return json.loads(p.read_text(encoding="utf-8"))


def get_pose(node):
    p = node.getPosition()
    o = node.getOrientation()
    return p[0], p[1], p[2], math.atan2(o[3], o[0])


def read_csv_rows(p: Path) -> list[dict]:
    with open(p, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def drive_path(robot, node, timestep, cfg, pid, cam, lm, rm, ls, rs, out_root,
               v_cap, max_wheel):
    dt = timestep / 1000.0
    in_dir = Path(cfg["input_dir"]) / f"path_{pid:02d}"
    out = Path(out_root) / f"path_{pid:02d}"

    wps = [(float(r["sim_time"]), float(r["gt_x"]), float(r["gt_y"]))
           for r in read_csv_rows(in_dir / "waypoints_raw.csv")]
    imu = read_csv_rows(in_dir / "imu.csv") if (in_dir / "imu.csv").is_file() else []
    geom = PathGeometry(wps, imu)   # curved path reconstructed from IMU heading
    prof = build_profile(geom, imu, v_cap=v_cap)
    st = prof.stats
    # Option A: a path with ANY segment demanding > cap is dropped WHOLE
    # (never drive a doomed path; dropping mid-path would break the track).
    if not st["feasible"]:
        out.mkdir(parents=True, exist_ok=True)
        (out / "_dropped.json").write_text(json.dumps(
            {"path_id": pid, "reason": "infeasible_segments",
             "n_infeasible": st["n_infeasible"],
             "n_segments": st["n_segments"]}, indent=2), encoding="utf-8")
        print(f"\n  [DROP] path {pid}: {st['n_infeasible']}/{st['n_segments']} "
              f"segments need > {v_cap:.2f} m/s -- dropped (not driven)",
              flush=True)
        return {"path_id": pid, "dropped": True, "reason": "infeasible_segments"}
    print(f"\n{'='*64}\n  DRIVE path {pid}: {len(wps)} presses, "
          f"L={geom.L:.1f}m, T={geom.times[-1]:.1f}s\n"
          f"  [profile] peak v={st['v_peak']:.2f} m/s, "
          f"infeasible segs={st['n_infeasible']}/{st['n_segments']}, "
          f"no-imu segs={st['n_no_imu']}, "
          f"integ err={st['integral_err_max']*1000:.2f} mm\n{'='*64}", flush=True)

    # ── place the robot at the start (setup teleport, then REAL driving) ──
    x0, y0 = geom.point_at(0.0)
    h0 = geom.heading_at(0.0)
    node.getField("translation").setSFVec3f([x0, y0, 0.0])
    node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, h0])
    node.resetPhysics()
    for _ in range(15):                       # settle on the floor
        robot.step(timestep)

    g = cfg["gains"]
    k_v, k_head = float(g["k_v"]), float(g["k_head"])
    k_lat, look = float(g["k_lat"]), float(g["lookahead_m"])
    rates = cfg["rates_hz"]
    per_gt, per_od, per_cam = (1.0 / rates["ground_truth"],
                               1.0 / rates["odometry"], 1.0 / rates["camera"])

    gt = CSVOut(out / "ground_truth.csv",
                ["gt_x", "gt_y", "gt_z", "gt_heading_rad", "gt_heading_deg",
                 "path_id", "s_actual", "s_target", "schedule_err_m",
                 "lateral_err_m"])
    od = CSVOut(out / "odometry.csv",
                ["odom_x", "odom_y", "odom_theta_deg", "odom_linear_vel",
                 "odom_angular_vel", "wheel_left_vel", "wheel_right_vel"])
    ca = CSVOut(out / "camera.csv", ["frame_id", "rgb_path", "depth_path",
                                     "cam_x", "cam_y", "cam_z"])
    fl = CSVOut(out / "flight_log.csv",
                ["s_target", "s_actual", "v_ff", "v_cmd", "omega_cmd",
                 "lateral_err_m", "heading_err_rad"])
    for c in (gt, od, ca, fl):
        c.open()
    (out / "camera").mkdir(exist_ok=True)

    # odometry state from REAL encoders
    import random
    rng = random.Random(12345 + pid)
    enc_l0, enc_r0 = ls.getValue(), rs.getValue()
    ox, oy, oth = x0, y0, h0
    last_l, last_r = enc_l0, enc_r0

    t_end = geom.times[-1]
    next_gt = next_od = next_cam = next_fl = 0.0
    frame = 0
    s_hint = 0.0
    behind = 0
    hits = []           # per-press (T_j, position error when t crosses T_j)
    next_press = 1
    t0_sim = robot.getTime()
    aborted = False
    finished = False
    flush = 20

    while True:
        if robot.step(timestep) == -1:
            aborted = not finished
            break
        t = robot.getTime() - t0_sim
        if t > t_end and not finished:
            lm.setVelocity(0.0)
            rm.setVelocity(0.0)
            finished = True
        if finished:
            flush -= 1
            if flush <= 0:
                break
            continue

        x, y, z, yaw = get_pose(node)
        s_a = geom.project(x, y, s_hint)
        s_hint = s_a
        s_t = prof.s_star(t)

        # press-hit bookkeeping (time is the schedule's; position measured)
        while next_press < len(geom.times) and t >= geom.times[next_press]:
            px, py = geom.wp[next_press]
            hits.append({"press": next_press,
                         "t": round(geom.times[next_press], 3),
                         "pos_err_m": round(math.hypot(x - px, y - py), 3)})
            next_press += 1

        # ── PURE-PURSUIT steering + clamped schedule catch-up ──
        # Proportional steering with a fixed short lookahead and no omega cap
        # made the robot circle in place on slow (heavily-dilated) paths:
        # v=0.57, omega=3.0 -> 0.2 m turn radius, s stuck. Pure pursuit ties
        # turn rate to speed and a speed-scaled lookahead, so it draws smooth
        # arcs instead; a pivot mode handles the "badly misaligned" case
        # without driving in a circle.
        lat = geom.lateral_offset(x, y, s_a)
        v_ff = prof.v(t)
        catch = k_v * max(-0.4, min(0.8, s_t - s_a))
        v_des = max(0.0, min(v_cap, v_ff + catch))
        Ld = min(2.0, max(0.7, 1.5 * v_des))              # speed-scaled lookahead
        cx, cy = geom.point_at(min(geom.L, s_a + Ld))
        alpha = wrap_pi(math.atan2(cy - y, cx - x) - yaw)  # angle to carrot
        h_err = alpha
        OMEGA_MAX = 1.5
        if abs(alpha) > 1.0:
            # badly misaligned: pivot (near-stop) instead of circling
            v_cmd = min(v_des, 0.12)
            om_cmd = max(-OMEGA_MAX, min(OMEGA_MAX, 1.5 * alpha))
        else:
            v_cmd = v_des * max(0.3, math.cos(alpha))
            dist = max(0.5, math.hypot(cx - x, cy - y))
            om_cmd = 2.0 * v_cmd * math.sin(alpha) / dist  # pure-pursuit curvature
            om_cmd = max(-OMEGA_MAX, min(OMEGA_MAX, om_cmd))
        vl = (v_cmd - om_cmd * AXLE / 2) / WHEEL_RADIUS
        vr = (v_cmd + om_cmd * AXLE / 2) / WHEEL_RADIUS
        pk = max(abs(vl), abs(vr))
        if pk > max_wheel:
            vl, vr = vl * max_wheel / pk, vr * max_wheel / pk

        # hopeless-behind guard: if the robot stays > 4 m behind schedule for
        # over a second, this path is infeasible in PHYSICS (accel/cap) --
        # abort cleanly and drop it rather than thrash off the map
        if s_t - s_a > 4.0:
            behind += 1
            if behind > int(1.0 / dt):
                for c in (gt, od, ca, fl):
                    c.close()
                print(f"  [DROP] path {pid}: {s_t - s_a:.1f} m behind schedule "
                      f">1 s -- infeasible in physics, dropping", flush=True)
                (out / "_dropped.json").write_text(json.dumps(
                    {"path_id": pid, "reason": "behind_schedule",
                     "gap_m": round(s_t - s_a, 2), "at_s": round(t, 1)},
                    indent=2), encoding="utf-8")
                return {"path_id": pid, "dropped": True,
                        "reason": "behind_schedule"}
        else:
            behind = 0
        lm.setVelocity(vl)
        rm.setVelocity(vr)

        # ── REAL encoder odometry (+ distance-proportional slip noise) ──
        el, er = ls.getValue(), rs.getValue()
        dl, dr = el - last_l, er - last_r
        last_l, last_r = el, er
        dl += rng.gauss(0.0, K_SLIP * abs(dl) + SIGMA0)
        dr += rng.gauss(0.0, K_SLIP * abs(dr) + SIGMA0)
        dsl, dsr = dl * WHEEL_RADIUS, dr * WHEEL_RADIUS
        lin, ang = (dsl + dsr) / 2, (dsr - dsl) / AXLE
        mid = oth + ang / 2
        ox += lin * math.cos(mid)
        oy += lin * math.sin(mid)
        oth = wrap_pi(oth + ang)

        if t >= next_gt:
            gt.write({"sim_time": round(t, 4), "gt_x": round(x, 5),
                      "gt_y": round(y, 5), "gt_z": round(z, 4),
                      "gt_heading_rad": round(yaw, 5),
                      "gt_heading_deg": round(math.degrees(yaw), 3),
                      "path_id": pid, "s_actual": round(s_a, 3),
                      "s_target": round(s_t, 3),
                      "schedule_err_m": round(s_t - s_a, 4),
                      "lateral_err_m": round(lat, 4)})
            next_gt += per_gt
        if t >= next_od:
            od.write({"sim_time": round(t, 4), "odom_x": round(ox, 5),
                      "odom_y": round(oy, 5),
                      "odom_theta_deg": round(math.degrees(oth), 3),
                      "odom_linear_vel": round(lin / dt, 5),
                      "odom_angular_vel": round(ang / dt, 5),
                      "wheel_left_vel": round(vl, 4),
                      "wheel_right_vel": round(vr, 4)})
            next_od += per_od
        if t >= next_cam and cam is not None:
            rel = f"camera/rgb_{frame:06d}.png"
            cam.saveImage(str(out / rel), 80)
            ca.write({"sim_time": round(t, 4), "frame_id": f"{frame:06d}",
                      "rgb_path": rel, "depth_path": "",
                      "cam_x": round(x, 4), "cam_y": round(y, 4),
                      "cam_z": round(z + 1.19, 4)})
            frame += 1
            next_cam += per_cam
        if t >= next_fl:
            fl.write({"sim_time": round(t, 4), "s_target": round(s_t, 3),
                      "s_actual": round(s_a, 3), "v_ff": round(v_ff, 3),
                      "v_cmd": round(v_cmd, 3), "omega_cmd": round(om_cmd, 3),
                      "lateral_err_m": round(lat, 4),
                      "heading_err_rad": round(h_err, 4)})
            next_fl += 0.25
        if int(t) != int(t - dt):
            print(f"  [{t:6.1f}/{t_end:6.1f}s] s={s_a:6.1f}/{s_t:6.1f} "
                  f"lat={lat:+.2f} v={v_cmd:.2f} frames={frame}", flush=True)

    for c in (gt, od, ca, fl):
        c.close()
    if aborted:
        print(f"  [ABORT] Webots quit -- path {pid} UNFINISHED "
              f"(no marker; redone on resume)", flush=True)
        return None

    result = {"path_id": pid, "frames": frame, "duration_s": t_end,
              "path_len_m": round(geom.L, 2), "n_presses": len(wps),
              "feasible": st.get("feasible", True),
              "n_infeasible_segments": st["n_infeasible"]}
    (out / "metadata.json").write_text(json.dumps({
        **result, "profile": st, "press_hits": hits,
        "gains": cfg["gains"], "v_cap": v_cap, "pipeline": "replay_v2",
    }, indent=2), encoding="utf-8")
    (out / "_done.json").write_text(json.dumps(
        {"summary": result,
         "completed": pytime.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2),
        encoding="utf-8")
    worst = max((h["pos_err_m"] for h in hits), default=0.0)
    print(f"  [hits] worst press position error: {worst:.2f} m "
          f"({len(hits)} presses)", flush=True)
    return result


def main():
    cfg = load_cfg()
    mt = pytime.strftime("%Y-%m-%d %H:%M:%S",
                         pytime.localtime(os.path.getmtime(os.path.abspath(__file__))))
    print("=" * 64)
    print("  NavLoRI replay_driver (pipeline v2 -- real Tiago++, real wheels)")
    print("  code mtime =", mt, " (stale-reload check)")
    print("  input  =", cfg["input_dir"])
    print("  output =", cfg["output_dir"])
    print("  paths  =", len(cfg["path_ids"]))
    print("=" * 64, flush=True)

    robot = Supervisor()
    node = robot.getSelf()
    if node is None:
        print("ERROR: TIAGO node needs 'supervisor TRUE'")
        return
    timestep = int(robot.getBasicTimeStep())

    # ── guards (v1-proven) ──
    stem = Path(robot.getWorldPath()).stem
    if stem != cfg["run_name"]:
        print(f"!!! WRONG WORLD OPEN: {stem}.wbt vs staged "
              f"{cfg['run_name']}.wbt -- REFUSING", flush=True)
        return
    exp = cfg.get("world_build_stamp")
    if exp:
        n = robot.getFromDef("BUILD_STAMP")
        live = n.getField("name").getSFString() if n else None
        if live != exp:
            print(f"!!! STALE SCENE (live {live} vs staged {exp}) -- "
                  f"File > Reload World, then Play", flush=True)
            return
    dirty = [d for d in ("ROBOT_PATHS", "MARKER_START_GREEN", "WP_00000")
             if robot.getFromDef(d) is not None]
    if dirty:
        print(f"!!! DEBUG GEOMETRY IN SCENE {dirty} -- Reload World", flush=True)
        return

    # ── Option A: drive TIAGO at its TRUE, unmodified wheel cap. The stock
    #     RotationalMotor maxVelocity (~10.15 rad/s -> ~1.0 m/s) cannot be
    #     raised from the controller (Webots ignores Supervisor writes to
    #     PROTO-internal motor fields), and raising it would violate
    #     "TIAGO as-is" anyway. We read the real limit and keep a small
    #     headroom so differential steering never exceeds it (no warnings). ──
    lm = robot.getDevice("wheel_left_joint")
    rm = robot.getDevice("wheel_right_joint")
    max_wheel = min(lm.getMaxVelocity(), rm.getMaxVelocity())   # rad/s (tuned in proto)
    v_cap = 0.98 * max_wheel * WHEEL_RADIUS                     # m/s, tiny headroom
    for m in (lm, rm):
        m.setPosition(float("inf"))
        m.setVelocity(0.0)
    print(f"  [OK] TIAGO true wheel cap: {max_wheel:.2f} rad/s "
          f"-> v_cap {v_cap:.2f} m/s (as-is, unmodified)")
    ls = lm.getPositionSensor()
    rs = rm.getPositionSensor()
    ls.enable(timestep)
    rs.enable(timestep)
    # camera: the real Tiago++ PROTO does NOT name it 'head_front_camera'
    # (that was the v1 rig). Auto-detect by device TYPE so we never miss it.
    cam = robot.getDevice("head_front_camera")
    if cam is None:
        from controller import Node
        for i in range(robot.getNumberOfDevices()):
            d = robot.getDeviceByIndex(i)
            if d.getNodeType() == Node.CAMERA:
                cam = robot.getDevice(d.getName())
                print(f"  [camera] auto-detected '{d.getName()}'")
                break
    if cam:
        cam.enable(timestep)
        print(f"  [OK] camera {cam.getWidth()}x{cam.getHeight()}")
    else:
        names = [robot.getDeviceByIndex(i).getName()
                 for i in range(robot.getNumberOfDevices())]
        print(f"  [FATAL] no Camera device found. devices: {names}", flush=True)
        return
    for name, pos in ARM_TUCK.items():
        m = robot.getDevice(name)
        if m:
            m.setVelocity(0.07 if name == "torso_lift_joint" else 1.0)
            m.setPosition(pos)
    for _ in range(80):
        robot.step(timestep)

    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    resume = bool(cfg.get("resume", True))
    summary = {}
    for pid in cfg["path_ids"]:
        pdir = out_root / f"path_{pid:02d}"
        done = pdir / "_done.json"
        dropped = pdir / "_dropped.json"
        if resume and done.is_file():
            try:
                summary[pid] = json.loads(done.read_text())["summary"]
                print(f"[resume] path {pid}: complete -- skipping", flush=True)
                continue
            except (json.JSONDecodeError, KeyError):
                pass
        if resume and dropped.is_file():
            print(f"[resume] path {pid}: previously dropped -- skipping", flush=True)
            continue
        if pdir.is_dir():
            print(f"[resume] path {pid}: partial -- wiping + redoing", flush=True)
            shutil.rmtree(pdir)
        try:
            r = drive_path(robot, node, timestep, cfg, pid, cam, lm, rm,
                           ls, rs, out_root, v_cap, max_wheel)
            if r is None:
                print("[abort] stopping (Webots quitting)", flush=True)
                break
            if not r.get("dropped"):
                summary[pid] = r
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[ERROR] path {pid}: {type(e).__name__}: {e}")
            traceback.print_exc()
            lm.setVelocity(0.0)
            rm.setVelocity(0.0)
            with open(out_root / "controller_crash.log", "a", encoding="utf-8") as f:
                f.write(f"=== path {pid} @ {pytime.strftime('%H:%M:%S')}\n"
                        f"{traceback.format_exc()}\n")
            continue

    print("\n" + "=" * 64 + "\n  DRIVE COMPLETE: "
          f"{len(summary)}/{len(cfg['path_ids'])} paths", flush=True)
    (out_root / "metadata.json").write_text(json.dumps(
        {"controller": "replay_driver", "config": cfg, "paths": summary,
         "created": pytime.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
