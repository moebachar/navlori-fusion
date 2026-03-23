"""
Fix path edges that cross the inflated wall safety zone.

Strategy:
1. For each edge, sample points every 3cm.
2. If any sample is inside an inflated wall polygon, push the nearest
   waypoint(s) away from the polygon centroid OR insert a midpoint that
   is pushed outward.
3. Iterate until no edge violates.

Uses the same wall geometry as viz_paths.py.
"""

import json
import math
import os
import copy

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS_FILE = os.path.join(HERE, "paths.json")

ROBOT_RADIUS = 0.55
SAFETY_COEFF = 1.1
EXTRA_MARGIN = 0.25
INFLATE = ROBOT_RADIUS * SAFETY_COEFF + EXTRA_MARGIN  # 0.605 + 0.25 = 0.855

RAW_WALLS = [
    (1.48, -1.47, 2.2327, 0.2, 2.7),
    (1.27, -3.35, -2.5454, 0.2, 2.707),
    (-0.61, -6.52, 2.2301, 0.2, 6.9709),
    (-10.33, -5.40, 2.15, 0.2, 7.5),
    (-2.16, -10.77, -2.6475, 0.2, 5.0),
    (1.10, -11.53, -1.0, 0.2, 5.1671),
    (3.99, -11.34, -2.6168, 0.2, 3.0),
    (-1.06, -16.10, 2.1194, 0.2, 13.8),
    (-8.01, -18.25, 0.6777, 0.2, 1.9),
    (-6.55, -16.13, -1.0151, 0.2, 5.1559),
    (-5.62, -12.74, 0.5114, 0.2, 4.6),
    (-8.43, -11.76, 2.1214, 0.2, 3.8606),
    (-12.77, -8.61, 0.5831, 0.2, 2.9),
    (-10.79, -11.60, 0.5831, 0.2, 2.7),
    (-6.30, -4.47, -2.482, 0.2, 3.1004),
    (-4.05, -4.58, -0.8897, 0.2, 3.6),
    (-3.72, -1.98, 0.5912, 0.2, 3.4447),
    (-2.75, 1.41, -0.8125, 0.2, 6.0),
    (0.84, 1.52, -2.5178, 0.2, 5.3),
]


def wall_polygons(inflate=0.0):
    polys = []
    for tx, ty, angle, sx, sy in RAW_WALLS:
        hx = sx / 2 + inflate
        hy = sy / 2 + inflate
        corners_local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        ca, sa = math.cos(angle), math.sin(angle)
        corners = [(tx + ca * cx - sa * cy, ty + sa * cx + ca * cy)
                    for cx, cy in corners_local]
        polys.append(corners)
    return polys


def point_in_polygon(px, py, poly):
    """Ray-casting algorithm."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def polygon_centroid(poly):
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return cx, cy


def nearest_point_on_segment(px, py, ax, ay, bx, by):
    """Project point (px,py) onto segment (ax,ay)-(bx,by), return nearest point and t."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return ax, ay, 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    return ax + t * dx, ay + t * dy, t


def point_to_polygon_push(px, py, poly, margin=0.05):
    """
    If point is inside polygon, compute a push vector to move it outside.
    Returns (new_x, new_y, was_inside).
    """
    if not point_in_polygon(px, py, poly):
        return px, py, False

    # Find the nearest edge and push outward past it
    min_dist = float("inf")
    best_nx, best_ny = px, py

    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]

        nx, ny, _ = nearest_point_on_segment(px, py, ax, ay, bx, by)
        d = math.hypot(px - nx, py - ny)
        if d < min_dist:
            min_dist = d

            # Push direction: from nearest edge point toward the test point,
            # then continue past the edge
            if d > 1e-6:
                dirx = (px - nx) / d
                diry = (py - ny) / d
            else:
                # Point is ON the edge — use edge normal
                ex, ey = bx - ax, by - ay
                elen = math.hypot(ex, ey)
                if elen > 1e-6:
                    # Outward normal (away from polygon center)
                    cx, cy = polygon_centroid(poly)
                    n1x, n1y = -ey / elen, ex / elen
                    # Pick the normal pointing away from centroid
                    if (nx + n1x - cx) ** 2 + (ny + n1y - cy) ** 2 > (nx - n1x - cx) ** 2 + (ny - n1y - cy) ** 2:
                        dirx, diry = n1x, n1y
                    else:
                        dirx, diry = -n1x, -n1y
                else:
                    dirx, diry = 0, 1

            # Push the point outside: move it to the edge + margin beyond
            push_dist = min_dist + margin
            best_nx = nx - dirx * margin  # Go from nearest edge point outward
            best_ny = ny - diry * margin
            # Actually: push from the original point in the OPPOSITE direction of
            # (point -> nearest_edge), i.e., away from the polygon
            # The push should go from the edge outward
            # edge_point + outward_normal * margin
            # outward = away from centroid
            cx, cy = polygon_centroid(poly)
            out_x = nx - cx
            out_y = ny - cy
            out_len = math.hypot(out_x, out_y)
            if out_len > 1e-6:
                out_x /= out_len
                out_y /= out_len
            best_nx = nx + out_x * margin
            best_ny = ny + out_y * margin

    return best_nx, best_ny, True


def edge_violates(ax, ay, bx, by, polys, step=0.03):
    """Check if any point along edge (a->b) is inside any inflated polygon."""
    d = math.hypot(bx - ax, by - ay)
    if d < 1e-6:
        return False, []
    n_samples = max(2, int(d / step))
    violations = []
    for i in range(n_samples + 1):
        t = i / n_samples
        px = ax + t * (bx - ax)
        py = ay + t * (by - ay)
        for pi, poly in enumerate(polys):
            if point_in_polygon(px, py, poly):
                violations.append((t, px, py, pi))
                break
    return len(violations) > 0, violations


def push_waypoint_away(wx, wy, polys, margin=0.08):
    """Push a waypoint out of ALL inflated polygons it's inside."""
    moved = True
    iterations = 0
    while moved and iterations < 20:
        moved = False
        iterations += 1
        for poly in polys:
            nx, ny, was_inside = point_to_polygon_push(wx, wy, poly, margin)
            if was_inside:
                wx, wy = nx, ny
                moved = True
    return wx, wy


def fix_paths():
    with open(PATHS_FILE) as f:
        data = json.load(f)

    polys = wall_polygons(INFLATE)

    total_fixes = 0
    total_midpoints = 0

    for max_iter in range(30):
        violations_found = 0
        push_margin = 0.15 + max_iter * 0.02  # increase push each iteration

        for path in data["paths"]:
            wps = path["waypoints"]

            # First: push all waypoints that are inside a polygon
            for wp in wps:
                new_x, new_y = push_waypoint_away(wp["x"], wp["y"], polys, margin=push_margin)
                if abs(new_x - wp["x"]) > 1e-6 or abs(new_y - wp["y"]) > 1e-6:
                    wp["x"] = round(new_x, 4)
                    wp["y"] = round(new_y, 4)
                    total_fixes += 1

            # Check edges and insert midpoints where needed
            i = 0
            while i < len(wps) - 1:
                ax, ay = wps[i]["x"], wps[i]["y"]
                bx, by = wps[i + 1]["x"], wps[i + 1]["y"]

                violates, viols = edge_violates(ax, ay, bx, by, polys)
                if violates:
                    violations_found += 1

                    # Sample multiple candidate midpoints (at 1/4, 1/2, 3/4 of violation zone)
                    t_values = sorted(set([v[0] for v in viols]))
                    mid_t = sum(t_values) / len(t_values)
                    mid_x = ax + mid_t * (bx - ax)
                    mid_y = ay + mid_t * (by - ay)

                    # Push the midpoint out of all polygons with escalating margin
                    new_x, new_y = push_waypoint_away(mid_x, mid_y, polys, margin=push_margin)

                    # Only insert if the pushed point is significantly different
                    if math.hypot(new_x - mid_x, new_y - mid_y) > 0.01:
                        new_wp = {"x": round(new_x, 4), "y": round(new_y, 4)}
                        wps.insert(i + 1, new_wp)
                        total_midpoints += 1
                    else:
                        # Push both endpoints away from the violating polygon
                        for idx in [i, i + 1]:
                            wx, wy = wps[idx]["x"], wps[idx]["y"]
                            for vi in viols:
                                pi = vi[3]
                                poly = polys[pi]
                                cx, cy = polygon_centroid(poly)
                                dx = wx - cx
                                dy = wy - cy
                                dl = math.hypot(dx, dy)
                                if dl > 1e-6:
                                    wps[idx]["x"] = round(wx + dx / dl * push_margin, 4)
                                    wps[idx]["y"] = round(wy + dy / dl * push_margin, 4)
                                    total_fixes += 1
                                break
                i += 1

        print(f"  Iteration {max_iter + 1}: {violations_found} violating edges (push={push_margin:.2f}m)")
        if violations_found == 0:
            break

    # Recalculate path lengths
    for path in data["paths"]:
        wps = path["waypoints"]
        length = 0
        for i in range(len(wps) - 1):
            length += math.hypot(wps[i + 1]["x"] - wps[i]["x"],
                                 wps[i + 1]["y"] - wps[i]["y"])
        path["length_m"] = round(length, 2)

    # Update metadata
    data["metadata"]["safe_clearance"] = INFLATE
    data["metadata"]["robot_radius"] = ROBOT_RADIUS

    # Save
    with open(PATHS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone: {total_fixes} waypoints pushed, {total_midpoints} midpoints inserted")
    print(f"Saved to {PATHS_FILE}")

    # Final verification
    total_remaining = 0
    for path in data["paths"]:
        wps = path["waypoints"]
        for i in range(len(wps) - 1):
            violates, _ = edge_violates(wps[i]["x"], wps[i]["y"],
                                         wps[i + 1]["x"], wps[i + 1]["y"],
                                         polys)
            if violates:
                total_remaining += 1
    print(f"Remaining violations: {total_remaining}")


if __name__ == "__main__":
    fix_paths()
