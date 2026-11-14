"""
Braking model shared by safety_node and gap_follow_node (no ROS code).

AEB lecture: TTC = range / closing speed, and the car brakes when TTC is below a threshold.
Milestone 2 and 3 used one fixed threshold. Here the threshold is the time the car actually
needs to stop, so it grows with speed (a variable TTC):

    required TTC   T(v, e) = t_react + (v / (2 a_brake)) * (v + 2 e) / (v + e) + ttc_margin

    v: our speed, e: how fast the obstacle comes towards us on its own (0 if static)

It comes from the distance we need, divided by the closing speed (v + e):
    reaction distance v * t_react, braking distance v^2 / (2 a_brake), and the distance the
    obstacle covers meanwhile, e * (t_react + v / a_brake).
For a static obstacle it is simply t_react + v / (2 a_brake) + ttc_margin; with the default
values that is 0.38 s at 0.5 m/s, 0.50 s at 1 m/s, 0.62 s at 1.5 m/s and 0.75 s at 2 m/s.

TTC is measured to the safety margin, not to the obstacle:
    ttc = (distance along the path - stop_margin) / closing speed
so the margin in front of the bumper stays a distance at every speed, and a gap of zero is
always an emergency, even when crawling.

The same model sets the speeds:
    the safety node caps the speed so that ttc stays above slow_factor * T (soft stage) and
    brakes when ttc <= T (hard stage); the planner aims for plan_factor * T, a little more
    than slow_factor, so the safety node rarely has to step in.
"""

from __future__ import annotations

import math

import numpy as np


def stopping_distance(speed: float, t_react: float, a_brake: float, margin: float) -> float:
    """
    Return the distance needed to stop from a speed.

    Args:
        speed (float): Current speed (m/s).
        t_react (float): Time before the brakes act (s).
        a_brake (float): Braking deceleration (m/s^2).
        margin (float): Gap to leave in front of the obstacle (m).

    Returns:
        float: Reaction distance + braking distance + margin (m).
    """
    return speed * t_react + speed * speed / (2.0 * a_brake) + margin


def safe_speed(distance: float, t_react: float, a_brake: float) -> float:
    """
    Return the highest speed that can still stop within a distance.

    Solves v * t_react + v^2 / (2 a_brake) = distance for v.

    Args:
        distance (float): Distance available for reacting and braking (m).
        t_react (float): Time before the brakes act (s).
        a_brake (float): Braking deceleration (m/s^2).

    Returns:
        float: Speed (m/s), 0 if the distance is not positive.
    """
    if distance <= 0.0:
        return 0.0
    return a_brake * (-t_react + math.sqrt(t_react * t_react + 2.0 * distance / a_brake))


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
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(closing > 1e-9, (v + 2.0 * e) / np.where(closing > 1e-9, closing, 1.0),
                         1.0)
    return t_react + v / (2.0 * a_brake) * share + ttc_margin


def slow_speed_cap(gap, obstacle_speed, factor: float, t_react: float, a_brake: float,
                   ttc_margin: float) -> np.ndarray:
    """
    Return the highest speed that keeps ttc at least factor times the required TTC.

    For a static obstacle, ttc = gap / v >= factor * (t_react + ttc_margin + v / (2 a_brake))
    means gap / factor >= v (t_react + ttc_margin) + v^2 / (2 a_brake): the safe_speed()
    equation with a reaction time of t_react + ttc_margin. An obstacle coming towards us
    takes its own speed off the cap, so the closing speed stays the same.

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
    g = np.maximum(np.asarray(gap, dtype=np.float64), 0.0) / max(factor, 1e-6)
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
        return speed * dt, 0.0, turn
    return speed / yaw_rate * math.sin(turn), speed / yaw_rate * (1.0 - math.cos(turn)), turn
