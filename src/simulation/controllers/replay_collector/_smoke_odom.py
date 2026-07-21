"""Quick check on OdometrySynthesizer: drive 0.5 m/s straight for ~1 s,
expect ~0.5 m forward with small noise."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from drive import OdometrySynthesizer


def main():
    o = OdometrySynthesizer()
    o.reset(0.0, 0.0, 0.0)
    last = None
    for _ in range(31):
        last = o.step(0.5, 0.0, 0.032)
    print(f"after ~1s @ 0.5 m/s straight:")
    print(f"  odom_x = {o.x:.4f} m (expected ~0.50)")
    print(f"  odom_y = {o.y:.4f} m (expected ~0.00 +/- noise)")
    print(f"  odom_theta = {o.theta:.4f} rad")
    print(f"  last row: {last}")

    # Spin in place at 1 rad/s for 1 s
    o2 = OdometrySynthesizer()
    o2.reset(0.0, 0.0, 0.0)
    for _ in range(31):
        o2.step(0.0, 1.0, 0.032)
    print(f"\nafter ~1s @ 1.0 rad/s spin:")
    print(f"  odom_theta = {o2.theta:.4f} rad (expected ~1.0)")
    print(f"  odom_x = {o2.x:.4f}, odom_y = {o2.y:.4f}  (expected ~0, 0)")


if __name__ == "__main__":
    main()
