"""
Dynamic Window Approach (DWA) — Local Planner
==============================================
Adapted from PythonRobotics by Atsushi Sakai (@Atsushi_twi)
https://github.com/AtsushiSakai/PythonRobotics
License: MIT

Computes optimal (v_linear, v_angular) for a differential-drive robot
given its current state, a goal position, and obstacle points from a
depth camera.

Reference:
  D. Fox, W. Burgard, S. Thrun,
  "The Dynamic Window Approach to Collision Avoidance",
  IEEE Robotics & Automation Magazine, 1997.
"""

import math
import numpy as np


class DWAConfig:
    """Tunable parameters for the DWA planner."""

    # Robot kinematic limits
    max_speed = 0.35           # m/s — TIAGo++ comfortable cruise
    min_speed = -0.10          # m/s — allow slight reverse
    max_yaw_rate = 1.2         # rad/s
    max_accel = 0.3            # m/s²
    max_delta_yaw_rate = 2.0   # rad/s²

    # Velocity sampling resolution
    v_resolution = 0.02        # m/s
    yaw_rate_resolution = 0.05 # rad/s

    # Prediction
    predict_time = 2.0         # seconds — trajectory lookahead
    dt = 0.1                   # seconds — integration step

    # Cost weights
    heading_cost_gain = 0.8    # alignment with goal
    dist_cost_gain = 0.05      # prefer far from obstacles
    speed_cost_gain = 0.5      # prefer faster trajectories
    goal_dist_cost_gain = 1.5  # prefer trajectories closer to goal

    # Safety
    robot_radius = 0.35        # meters — TIAGo++ footprint radius
    obstacle_cost_inf = 1e6    # cost if trajectory collides


class DWAPlanner:
    """Dynamic Window Approach planner for differential-drive robots."""

    def __init__(self, config=None):
        self.config = config or DWAConfig()

    def plan(self, state, goal, obstacles):
        """
        Compute the best (v, omega) command.

        Parameters
        ----------
        state : array-like [x, y, yaw, v, omega]
            Current robot state in world frame.
        goal : array-like [x, y]
            Target waypoint in world frame.
        obstacles : np.ndarray (N, 2)
            Obstacle positions in world frame.
            Can be empty (no obstacles detected).

        Returns
        -------
        v : float — linear velocity (m/s)
        omega : float — angular velocity (rad/s)
        best_traj : np.ndarray (M, 5) — predicted trajectory of best command
        """
        cfg = self.config
        state = np.array(state, dtype=float)
        goal = np.array(goal, dtype=float)

        # Dynamic window = intersection of velocity limits and acceleration limits
        dw = self._calc_dynamic_window(state)

        best_cost = float("inf")
        best_v = 0.0
        best_omega = 0.0
        best_traj = np.array([state])

        # Sample velocity space
        v = dw[0]
        while v <= dw[1]:
            omega = dw[2]
            while omega <= dw[3]:
                traj = self._predict_trajectory(state, v, omega)

                # Cost components
                heading_cost = cfg.heading_cost_gain * self._heading_cost(traj, goal)
                obstacle_cost = cfg.dist_cost_gain * self._obstacle_cost(traj, obstacles)
                speed_cost = cfg.speed_cost_gain * (cfg.max_speed - traj[-1, 3])
                goal_dist_cost = cfg.goal_dist_cost_gain * self._goal_dist_cost(traj, goal)

                total_cost = heading_cost + obstacle_cost + speed_cost + goal_dist_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_v = v
                    best_omega = omega
                    best_traj = traj

                omega += cfg.yaw_rate_resolution
            v += cfg.v_resolution

        return best_v, best_omega, best_traj

    def _calc_dynamic_window(self, state):
        """Compute the dynamic window [v_min, v_max, omega_min, omega_max]."""
        cfg = self.config
        v = state[3]
        omega = state[4]

        # Velocity limits
        vs = [cfg.min_speed, cfg.max_speed,
              -cfg.max_yaw_rate, cfg.max_yaw_rate]

        # Acceleration limits for one timestep
        va = [v - cfg.max_accel * cfg.dt,
              v + cfg.max_accel * cfg.dt,
              omega - cfg.max_delta_yaw_rate * cfg.dt,
              omega + cfg.max_delta_yaw_rate * cfg.dt]

        # Intersection
        dw = [max(vs[0], va[0]), min(vs[1], va[1]),
              max(vs[2], va[2]), min(vs[3], va[3])]

        return dw

    def _predict_trajectory(self, state, v, omega):
        """Simulate a trajectory for predict_time given constant (v, omega)."""
        cfg = self.config
        traj = [state.copy()]
        s = state.copy()
        t = 0.0
        while t <= cfg.predict_time:
            s = self._motion(s, v, omega)
            traj.append(s.copy())
            t += cfg.dt
        return np.array(traj)

    def _motion(self, state, v, omega):
        """Single integration step."""
        cfg = self.config
        state[0] += v * math.cos(state[2]) * cfg.dt  # x
        state[1] += v * math.sin(state[2]) * cfg.dt  # y
        state[2] += omega * cfg.dt                     # yaw
        state[3] = v                                   # v
        state[4] = omega                               # omega
        return state

    def _heading_cost(self, traj, goal):
        """Angle between final trajectory heading and goal direction."""
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        goal_angle = math.atan2(dy, dx)
        cost = abs(self._angle_diff(goal_angle, traj[-1, 2]))
        return cost

    def _goal_dist_cost(self, traj, goal):
        """Euclidean distance from trajectory end to goal."""
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        return math.hypot(dx, dy)

    def _obstacle_cost(self, traj, obstacles):
        """Cost based on minimum distance to obstacles along trajectory."""
        cfg = self.config
        if obstacles is None or len(obstacles) == 0:
            return 0.0

        # Check each trajectory point against all obstacles
        traj_xy = traj[:, :2]
        min_dist = float("inf")

        for ox, oy in obstacles:
            dists = np.hypot(traj_xy[:, 0] - ox, traj_xy[:, 1] - oy)
            d = dists.min()
            if d <= cfg.robot_radius:
                return cfg.obstacle_cost_inf  # collision
            min_dist = min(min_dist, d)

        # Inverse distance — closer obstacles cost more
        return 1.0 / (min_dist - cfg.robot_radius + 0.001)

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d


def depth_to_obstacles(depth_sensor, robot_x, robot_y, robot_yaw,
                       max_range=3.0, sample_step=8):
    """
    Convert a depth camera image to obstacle (x, y) points in world frame.

    Samples a horizontal band of the depth image, projects each pixel to
    a 3D point using the camera's horizontal FOV, then rotates into world
    coordinates.

    Parameters
    ----------
    depth_sensor : Webots RangeFinder device
    robot_x, robot_y, robot_yaw : float — robot pose in world frame
    max_range : float — ignore readings beyond this (meters)
    sample_step : int — sample every Nth pixel (performance)

    Returns
    -------
    np.ndarray (N, 2) — obstacle positions in world frame
    """
    if depth_sensor is None:
        return np.empty((0, 2))

    try:
        ri = depth_sensor.getRangeImage()
        if not ri:
            return np.empty((0, 2))
    except Exception:
        return np.empty((0, 2))

    w = depth_sensor.getWidth()
    h = depth_sensor.getHeight()
    fov = depth_sensor.getFov()
    sensor_max = depth_sensor.getMaxRange()

    # Sample center rows
    obstacles = []
    rows_to_check = [h // 3, h // 2, 2 * h // 3]
    half_fov = fov / 2.0

    for row in rows_to_check:
        start = row * w
        for col in range(0, w, sample_step):
            d = ri[start + col]
            if d < 0.15 or d > min(max_range, sensor_max):
                continue
            # Pixel angle relative to camera center
            angle = half_fov - (col / w) * fov
            # Point in robot frame (camera faces forward)
            local_x = d * math.cos(angle)
            local_y = d * math.sin(angle)
            # Transform to world frame
            cos_y = math.cos(robot_yaw)
            sin_y = math.sin(robot_yaw)
            wx = robot_x + cos_y * local_x - sin_y * local_y
            wy = robot_y + sin_y * local_x + cos_y * local_y
            obstacles.append((wx, wy))

    if not obstacles:
        return np.empty((0, 2))
    return np.array(obstacles)
