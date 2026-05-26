"""TartanAir hospital P000 — image-only Camera benchmark.

Data: ``data/tartanair_hospital/hospital/hospital/Easy/P000/``
- ``image_left/`` : 563 RGB frames (PNG, 480x640, ~10 Hz native).
- ``pose_left.txt`` : NED pose per frame (7 columns: x y z qx qy qz qw).

TartanAir v1 image-only (no IMU). RESULT_08 ran TartanVO (full SLAM,
3 compat shims) on this sequence; our DPVOMotionEncoder evaluated on
first 80 % training / last 20 % test slice.
"""
from __future__ import annotations

from pathlib import Path

from ._common import not_applicable, path_to

DATASET_NAME = "tartanair_hospital"
DATA_DIR = "tartanair_hospital"


def load(**kwargs):
    """Return a dict with image paths + GT poses. No DataLoader
    (this is a per-frame benchmark; users iterate the list)."""
    import numpy as np
    seq_root = path_to(f"data/{DATA_DIR}/hospital/hospital/Easy/P000")
    image_dir = seq_root / "image_left"
    pose_file = seq_root / "pose_left.txt"
    if not image_dir.is_dir():
        return {"image_files": [], "poses_ned": None,
                "note": f"P000 image directory not found at {image_dir}"}
    image_files = sorted(image_dir.glob("*.png"))
    poses = None
    if pose_file.is_file():
        poses = np.loadtxt(pose_file)  # (N, 7) — x y z qx qy qz qw, NED frame
    return {
        "image_files": image_files,
        "poses_ned": poses,
        "n_frames": len(image_files),
    }


def stats() -> dict:
    seq_root = path_to(f"data/{DATA_DIR}/hospital/hospital/Easy/P000")
    image_dir = seq_root / "image_left"
    pose_file = seq_root / "pose_left.txt"
    n_frames = len(list(image_dir.glob("*.png"))) if image_dir.is_dir() else 0
    return {
        "name": DATASET_NAME,
        "data_dir": str(seq_root.relative_to(path_to("."))) if seq_root.is_dir() else f"data/{DATA_DIR}/hospital/hospital/Easy/P000",
        "modalities_available": ["camera"],
        "subset": "hospital/Easy/P000",
        "n_frames": n_frames,
        "frame_resolution": "480x640 RGB",
        "native_fps": 10,
        "pose_format": "NED 7-float (x, y, z, qx, qy, qz, qw)",
        "split_convention": "first 80 % train / last 20 % test slice (RESULT_08)",
        "known_caveats": [
            "Image-only TartanAir v1 — NO IMU; only Camera modality.",
            "RESULT_08 reports TartanVO last-20 % ATE 0.012 m vs DPVOMotion 0.293 m → +2300 % gap, paper-soft per-leg verdict.",
            "Used to validate the DPVOMotion encoder's trunk transferability (Mode α Webots-trained head infeasible without saved head).",
        ],
        "source_result": "RESULT_08",
    }


def preprocessing_demo(modality: str, n_samples: int = 1) -> dict:
    if modality != "camera":
        return not_applicable(modality, DATASET_NAME)
    seq_root = path_to(f"data/{DATA_DIR}/hospital/hospital/Easy/P000")
    image_dir = seq_root / "image_left"
    images = sorted(image_dir.glob("*.png"))[:n_samples] if image_dir.is_dir() else []
    return {
        "raw": [str(p) for p in images],
        "preprocessed": None,
        "description_raw": f"RGB 480x640 frames (n={len(images)} sample path(s))",
        "description_preprocessed": "ImageNet normalisation -> DPVOMotion 2x-0.5 affine to [-1, 1] -> BasicEncoder4 trunk -> 128-d patch features",
        "preprocessing_pipeline": ["read PNG", "to_tensor / [0,1]", "ImageNet norm", "DPVO 2x-0.5", "BasicEncoder4 trunk"],
        "note": "Image-tensor preprocessing visualisation deferred to plot helpers; raw PNG path is returned for the plotter to load.",
    }


__all__ = ["load", "stats", "preprocessing_demo"]
