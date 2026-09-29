"""Register the Vive tracking frame to real room coordinates using taped anchor points.

Protocol (see README): record a short dedicated session in which you place the
controller on each taped anchor IN THE ORDER listed in anchors.json, holding the
trigger ~1 s at each. Then:

    python register_frame.py sessions\vive_gt_XXXX.csv anchors.json
        -> fits the Vive->room transform, prints residuals, saves frame_calibration.json

    python register_frame.py --apply sessions\vive_gt_YYYY.csv
        -> adds room_x/room_y columns to any session CSV using the saved calibration
           (writes vive_gt_YYYY_room.csv next to it)

Fit is a 2D orthogonal Procrustes (rotation + translation, reflection allowed,
no scaling) on the floor plane: Vive (x, z) -> room (x, y).
"""

import argparse
import json
from pathlib import Path

import numpy as np

CALIB_DEFAULT = Path(__file__).parent / "frame_calibration.json"


def load_session(csv_path):
    return np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def primary_controller(data):
    ctrl = data[data["device"] == "controller"]
    if len(ctrl) == 0:
        raise SystemExit("no controller rows in this CSV")
    serials, counts = np.unique(ctrl["serial"], return_counts=True)
    return ctrl[ctrl["serial"] == serials[np.argmax(counts)]]


def tap_positions(d, min_gap_s=0.5):
    """Mean valid (x, z) of each contiguous trigger-held block, in time order."""
    taps = []
    block = []
    last_t = None
    for row in d:
        if row["trigger"] == 1 and row["valid"] == 1:
            if block and last_t is not None and row["t_mono"] - last_t > min_gap_s:
                taps.append(block)
                block = []
            block.append((row["x"], row["z"]))
            last_t = row["t_mono"]
        elif block:
            taps.append(block)
            block = []
    if block:
        taps.append(block)
    return np.array([np.mean(b, axis=0) for b in taps]) if taps else np.empty((0, 2))


def fit_transform(vive_xy, room_xy):
    """Orthogonal Procrustes without scaling: room ~= R @ vive + t."""
    pc, qc = vive_xy.mean(axis=0), room_xy.mean(axis=0)
    H = (vive_xy - pc).T @ (room_xy - qc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T  # reflection allowed on purpose: axis conventions may mirror
    t = qc - R @ pc
    resid = room_xy - (vive_xy @ R.T + t)
    return R, t, resid


def apply_calibration(csv_path, calib_path):
    with open(calib_path) as f:
        calib = json.load(f)
    R = np.array(calib["R"])
    t = np.array(calib["t"])
    data = load_session(csv_path)
    vive = np.stack([data["x"], data["z"]], axis=1)
    room = vive @ R.T + t
    out_path = Path(csv_path).with_name(Path(csv_path).stem + "_room.csv")
    names = list(data.dtype.names) + ["room_x", "room_y"]
    with open(out_path, "w") as f:
        f.write(",".join(names) + "\n")
        for row, (rx, ry) in zip(data, room):
            f.write(",".join(str(row[n]) for n in data.dtype.names)
                    + f",{rx:.6f},{ry:.6f}\n")
    print(f"wrote {out_path} (room_x/room_y in meters, room frame)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="session CSV (anchor-tap session, or any session with --apply)")
    parser.add_argument("anchors", nargs="?", type=Path, help="anchors JSON (fit mode)")
    parser.add_argument("--apply", action="store_true", help="apply saved calibration instead of fitting")
    parser.add_argument("--calib", type=Path, default=CALIB_DEFAULT, help="calibration file path")
    args = parser.parse_args()

    if args.apply:
        apply_calibration(args.csv, args.calib)
        return

    if args.anchors is None:
        raise SystemExit("fit mode needs an anchors JSON (see anchors.example.json), "
                         "or pass --apply to use an existing calibration")
    with open(args.anchors) as f:
        spec = json.load(f)
    room_xy = np.array([[a["x"], a["y"]] for a in spec["anchors"]])
    names = [a["name"] for a in spec["anchors"]]
    if len(room_xy) < 3:
        raise SystemExit("need at least 3 anchors (4 recommended), not in a straight line")

    d = primary_controller(load_session(args.csv))
    taps = tap_positions(d)
    print(f"found {len(taps)} trigger taps, expected {len(room_xy)} anchors")
    if len(taps) != len(room_xy):
        raise SystemExit("tap count must match anchor count — redo the anchor session: "
                         "one clean trigger-hold per anchor, in the listed order")

    R, t, resid = fit_transform(taps, room_xy)
    err = np.linalg.norm(resid, axis=1)
    for name, e in zip(names, err):
        print(f"  anchor {name}: residual {e * 1000:.1f} mm")
    rmse = float(np.sqrt((err ** 2).mean()))
    print(f"RMSE {rmse * 1000:.1f} mm | rotation det {np.linalg.det(R):+.0f} "
          f"({'mirrored' if np.linalg.det(R) < 0 else 'direct'} — either is fine)")
    if rmse > 0.05:
        print("WARNING: RMSE > 5 cm — check anchor measurements, tap order, and that the "
              "controller sat still on each anchor. Not saving a bad calibration is wise.")

    calib = {
        "R": R.tolist(), "t": t.tolist(), "rmse_m": rmse,
        "anchor_names": names, "source_csv": str(args.csv),
        "residuals_mm": [round(float(e) * 1000, 1) for e in err],
    }
    with open(args.calib, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"saved {args.calib}")


if __name__ == "__main__":
    main()
