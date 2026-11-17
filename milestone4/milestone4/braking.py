"""
Braking model shared by safety_node and gap_follow_node (no ROS code).

The AEB threshold is a variable TTC: the time the car needs to stop at its speed v,
given an obstacle approaching at e (0 if static):

    T(v, e) = t_react + (v / (2 a_brake)) * (v + 2 e) / (v + e) + ttc_margin

TTC is measured to the stop margin, (path distance - stop_margin) / closing speed. The
safety node brakes when ttc <= T and caps speed at slow_factor * T; the planner aims
for plan_factor * T.
"""

from __future__ import annotations

import math

import numpy as np


def required_ttc(speed, obstacle_speed, t_react: float, a_brake: float,
                 ttc_margin: float) -> np.ndarray:
    """
    Return the time-to-collision the car needs to stop in time (the variable TTC threshold).

    Args:
        speed: Our speed (m/s), a number or an array.
        obstacle_speed: Each obstacle's own speed towards us (m/s, >= 0), a number or array.
        t_react (float): Time before the brakes act (s).
        a_brake (float): Braking deceleration (m/s^2).
        ttc_margin (float): Extra time added for safety (s).

    Returns:
        np.ndarray: Required TTC (s), measured to the safety margin.
    """
    v = np.asarray(speed, dtype=np.float64)
    e = np.maximum(np.asarray(obstacle_speed, dtype=np.float64), 0.0)
    closing = v + e
    # share = (v + 2e) / (v + e); 1 when nothing is closing, to avoid dividing by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(closing > 1e-9, (v + 2.0 * e) / np.where(closing > 1e-9, closing, 1.0),
                         1.0)
    return t_react + v / (2.0 * a_brake) * share + ttc_margin


def slow_speed_cap(gap, obstacle_speed, factor: float, t_react: float, a_brake: float,
                   ttc_margin: float) -> np.ndarray:
    """
    Return the highest speed that keeps ttc at least factor times the required TTC.

    For a static obstacle, ttc = gap / v >= factor * (t_react + ttc_margin + v / (2 a_brake))
    means gap / factor >= v (t_react + ttc_margin) + v^2 / (2 a_brake), a quadratic in v
    that is solved for its positive root. An obstacle coming towards us takes its own
    speed off the cap, so the closing speed stays the same.

    Args:
        gap: Distance to the safety margin along the path (m), a number or an array.
        obstacle_speed: Each obstacle's own speed towards us (m/s, >= 0).
        factor (float): How many times the required TTC to keep (>= 1).
        t_react (float): Time before the brakes act (s).
        a_brake (float): Braking deceleration (m/s^2).
        ttc_margin (float): Extra time added for safety (s).

    Returns:
        np.ndarray: Speed cap (m/s, >= 0).
    """
    # positive root of v t + v^2 / (2 a_brake) = gap / factor, with t = t_react + ttc_margin
    g =np.maximum(np.asarray(gap, dtype=np.float64), 0.0) / max(factor, 1e-6)
    t = t_react + ttc_margin
    v = a_brake * (-t + np.sqrt(t * t + 2.0 * g / a_brake))
    return np.maximum(v - np.maximum(np.asarray(obstacle_speed, dtype=np.float64), 0.0), 0.0)


def path_distances(ranges: np.ndarray, angles: np.ndarray, steering: float, wheelbase: float,
                   half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the scan points on the car's path and how far along the path they are.

    On an arc of curvature k = tan(steering) / wheelbase, a point is on the path if it is
    within half_width of the arc; its distance is the arc length to it.

    Args:
        ranges (np.ndarray): Ranges (m), inf where invalid.
        angles (np.ndarray): Angle of each beam (rad, left positive).
        steering (float): Commanded steering angle (rad, left positive).
        wheelbase (float): Distance between the axles (m).
        half_width (float): Half the corridor width: half the car plus a margin (m).

    Returns:
        tuple[np.ndarray, np.ndarray]: (beam indices, distance along the path in m)
        of the points inside the corridor ahead of the car.
    """
    finite = np.isfinite(ranges)
    safe_r = np.where(finite, ranges, 0.0)
    x = safe_r * np.cos(angles)      # ahead
    y = safe_r * np.sin(angles)      # left

    curvature = math.tan(steering) / wheelbase
    if abs(curvature) < 1e-3:
        # (nearly) straight: a corridor along x
        on_path = finite & (x > 0.0) & (np.abs(y) <= half_width)
        along = x
    else:
        # an arc around the turning centre (0, radius); radius > 0 turns left
        radius = 1.0 / curvature
        rho = np.hypot(x, y - radius)                               # distance from the centre
        phi = np.arctan2(x, math.copysign(1.0, radius) * (radius - y))  # angle along the arc
        on_path = (finite & (np.abs(rho - abs(radius)) <= half_width)
                   & (phi > 0.0) & (phi <= math.pi / 2.0))          # up to a quarter turn ahead
        along = abs(radius) * phi

    index = np.nonzero(on_path)[0]
    return index, along[index]


def car_motion(speed: float, yaw_rate: float, dt: float) -> tuple[float, float, float]:
    """
    Return how the car moved in dt at a constant speed and yaw rate.

    Args:
        speed (float): Forward speed (m/s).
        yaw_rate (float): Yaw rate (rad/s, left positive).
        dt (float): Time (s).

    Returns:
        tuple[float, float, float]: (ahead, left, turn) in the starting frame (m, m, rad).
    """
    turn = yaw_rate * dt
    if abs(yaw_rate) < 1e-3:
        return speed * dt, 0.0, turn       # (nearly) straight line
    # constant-curvature arc of radius speed / yaw_rate
    return speed / yaw_rate * math.sin(turn), speed / yaw_rate * (1.0 - math.cos(turn)), turn
