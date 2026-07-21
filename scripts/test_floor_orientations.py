"""Emit 4 variants of the floor texture (none / flip-Y / flip-X / rot-180) into
the worlds dir so you can swap-test which orientation matches the walls
without rebuilding the whole .wbt.

Usage:
    .venv\\Scripts\\python.exe scripts\\test_floor_orientations.py \\
        --dataset-dir data/iln20_5d27099f_F2

Then in your .wbt, change the texture URL line from:
    url [ "iln20_5d27099f_F2_floor.png" ]
to one of:
    url [ "iln20_5d27099f_F2_floor_flipY.png" ]
    url [ "iln20_5d27099f_F2_floor_flipX.png" ]
    url [ "iln20_5d27099f_F2_floor_rot180.png" ]
Reload the world, check alignment, repeat.
"""
import argparse, os
from pathlib import Path
from PIL import Image as PILImage

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-dir", required=True)
ap.add_argument("--worlds-dir", default=r"X:\navlori-fusion\src\simulation\worlds")
args = ap.parse_args()

name = Path(args.dataset_dir).name
worlds = Path(args.worlds_dir)
src = worlds / f"{name}_floor.png"
if not src.exists():
    raise SystemExit(f"missing {src} — run build_iln20_webots_world.py first")

img = PILImage.open(src)
flipY = img.transpose(PILImage.FLIP_TOP_BOTTOM)
flipX = img.transpose(PILImage.FLIP_LEFT_RIGHT)
rot180 = img.transpose(PILImage.ROTATE_180)

flipY.save(worlds / f"{name}_floor_flipY.png", "PNG", optimize=True)
flipX.save(worlds / f"{name}_floor_flipX.png", "PNG", optimize=True)
rot180.save(worlds / f"{name}_floor_rot180.png", "PNG", optimize=True)
print(f"wrote 3 variants alongside {src}:")
print(f"  {name}_floor_flipY.png")
print(f"  {name}_floor_flipX.png")
print(f"  {name}_floor_rot180.png")
