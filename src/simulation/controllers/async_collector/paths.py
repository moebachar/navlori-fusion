"""
NavLoRI — 30 collision-free paths for async data collection.
Loads paths from paths.json (generated via interactive web editor).
All waypoints maintain safe clearance from walls (robot_radius * 1.5).
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATHS_FILE = os.path.join(_HERE, "paths.json")


def _load_paths():
    """Load paths from paths.json and return dict {id: {name, waypoints}}."""
    with open(_PATHS_FILE, "r") as f:
        data = json.load(f)

    paths = {}
    for p in data["paths"]:
        pid = p["id"]
        wps = [(w["x"], w["y"]) for w in p["waypoints"]]
        paths[pid] = {
            "name": f"Path-{pid}",
            "waypoints": wps,
        }
    return paths


ALL_PATHS = _load_paths()
