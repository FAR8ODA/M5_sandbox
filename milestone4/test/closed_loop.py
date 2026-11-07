"""
Closed-loop simulation of the Milestone 4 nodes (no ROS needed).

The real fusion_node, gap_follow_node, safety_node and lap_counter_node run together
against a simulated stadium-shaped track: a LiDAR scan is ray-cast from the car's pose,
the nodes' messages are passed between them as ROS would, and the car follows the
/drive command with a simple bicycle model (limited acceleration and braking).

Used by test_closed_loop.py, and runnable on its own to print a lap report:
    cd milestone4/test && python3 closed_loop.py
"""

import math
import random

import numpy as np
import ros_stubs

from milestone4.fusion_node import FusionNode  # noqa: I100  (after ros_stubs)
from milestone4.gap_follow_node import GapFollowNode  # noqa: I100
from milestone4.lap_counter_node import LapCounterNode  # noqa: I100
from milestone4.safety_node import SafetyNode  # noqa: I100

WHEELBASE = 0.33
STEP = 0.025                  # LiDAR period (s)

# A car as it really is, with its steering servo calibrated (it turns the commanded angle):
# the steering acts 0.1 s late and lags, the odometry speed is 10% noisy, the LiDAR sits
# 0.27 m ahead of the rear axle and its ranges have 2 cm of noise.
REALISTIC_CAR = {'servo_lag': 0.1, 'steer_gain': 1.0, 'steer_offset': 0.0,
                 'lidar_offset': 0.27, 'range_noise': 0.02, 'steer_delay': 0.1,
                 'odom_noise': 0.1}

# The same car with an uncalibrated servo: it turns 80% of the command plus a 0.03 rad pull
# to the left. Calibrate the servo before driving (README, tuning): the planner assumes the
# car turns the commanded angle, and this car can run into the inner wall.
UNCALIBRATED_CAR = dict(REALISTIC_CAR, steer_gain=0.8, steer_offset=0.03)


def stadium_track(straight=6.0, radius=2.0, half_lane=0.6, step=0.005):
    """
    Return dense wall points of a stadium track and the length of its centreline.

    The centreline has two straights along x (y = -radius and y = +radius) joined by
    semicircles; the car starts at the origin of the lower straight heading +x.
    """
    walls = []
    for r in (radius - half_lane, radius + half_lane):
        n_straight = int(straight / step)
        xs = np.linspace(0.0, straight, n_straight)
        walls.append(np.stack([xs, np.full_like(xs, -r)], axis=1))
        walls.append(np.stack([xs, np.full_like(xs, r)], axis=1))
        n_arc = int(math.pi * r / step)
        a = np.linspace(-math.pi / 2, math.pi / 2, n_arc)
        walls.append(np.stack([straight + r * np.cos(a), r * np.sin(a)], axis=1))
        walls.append(np.stack([-r * np.cos(a), r * np.sin(a)], axis=1))
    world = np.vstack(walls) - [straight / 2.0, -radius]    # start in the middle of a straight
    return world, 2.0 * straight + 2.0 * math.pi * radius


def box(cx, cy, size=0.4, step=0.005):
    """Return dense points around a square box centred at (cx, cy)."""
    h = size / 2.0
    corners = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    sides = [ros_stubs.wall_points(*corners[i], *corners[(i + 1) % 4], step) for i in range(4)]
    return np.vstack(sides)


def blocking_boxes(radius=2.0, x=0.0):
    """
    Return two boxes side by side across the middle of the far straight (halfway round).

    The far straight's centreline is at y = 2 * radius. The boxes cover 0.8 m of the lane's
    middle, leaving gaps of 0.2 m each side on a 1.2 m lane: too narrow for the 0.3 m car,
    so it must stop in front of them.
    """
    centre = 2.0 * radius
    return np.vstack([box(x, centre - 0.2), box(x, centre + 0.2)])


def depth_scan(points, x, y, yaw, t, fov=math.radians(87.0), beams=90, range_max=2.5):
    """
    Return the simulated depth camera LaserScan of points seen from a car pose.

    The camera sits on the LiDAR (cam_x = cam_y = 0), looks straight ahead with the D435i's
    87 degree depth field of view, and reports range_max where it sees nothing.
    """
    d = np.asarray(points, dtype=float) - [x, y]
    c, s = math.cos(-yaw), math.sin(-yaw)
    px, py = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
    r, a = np.hypot(px, py), np.arctan2(py, px)
    angle_min, increment = -fov / 2.0, fov / (beams - 1)
    b = np.round((a - angle_min) / increment).astype(int)
    seen = (b >= 0) & (b < beams) & (px > 0.0) & (r < range_max)
    ranges = np.full(beams, range_max)
    np.minimum.at(ranges, b[seen], r[seen])
    return ros_stubs.make_scan(ranges, angle_min=angle_min, angle_increment=increment,
                               range_min=0.15, range_max=range_max, t=t)


def box_labels(scan, box_centres, x, y, yaw, t, probability, rng, half_size=0.2):
    """
    Return a simulated /obstacle_scan: the beams that hit a box, labelled with a probability.

    The real camera labels a box in some frames and not in others; each scan here labels
    every box with the given probability, like a flickering camera.
    """
    r = np.asarray(scan.ranges, dtype=float)
    a = scan.angle_min + np.arange(r.size) * scan.angle_increment
    out = np.full(r.size, np.inf)
    if rng.random() < probability:
        c, s = math.cos(-yaw), math.sin(-yaw)
        for cx, cy in box_centres:
            bx, by = c * (cx - x) - s * (cy - y), s * (cx - x) + c * (cy - y)
            dist = math.hypot(bx, by)
            half = math.atan2(half_size + 0.08, max(dist, 0.3))
            hit = (np.abs(a - math.atan2(by, bx)) <= half) & (np.abs(r - dist) < 0.35)
            out[hit] = r[hit]
    return ros_stubs.make_scan(out, t=t)


def run(world, num_laps=2, max_speed=1.0, min_lap_distance=10.0, max_time=80.0,
        a_brake=2.0, accel=3.0, lidar_dropout=None, low_obstacles=None, use_camera=False,
        servo_lag=0.0, box_centres=None, label_probability=1.0, seed=1, scan_lag=None,
        steer_gain=1.0, steer_offset=0.0, lidar_offset=0.0, range_noise=0.0,
        steer_delay=0.0, odom_noise=0.0, track_obstacles=None, margin=None, smoothing=None,
        planner=None):
    """
    Drive the nodes around a world until the lap counter stops the car (or max_time).

    Args:
        world: Nx2 wall points, or a function of time returning them (moving obstacles).
        lidar_dropout: (start, end) seconds with no LiDAR scans, or None.
        low_obstacles: Nx2 points below the LiDAR's scan plane: only the depth camera sees
            them (if use_camera), but the car can still hit them.
        use_camera: Feed fusion_node a simulated /depth_scan at 30 Hz.
        servo_lag: Time constant of the steering servo (s). The odometry yaw rate comes
            from the commanded steering (as vesc_to_odom does), but the car turns with the
            lagging real steering.
        box_centres: Centres of boxes for simulated camera box labels (/obstacle_scan),
            each scan labelled with label_probability; None leaves /obstacle_scan to fusion.
        scan_lag: Function of time giving how late scans reach the safety node (s), as on a
            busy Jetson; they keep their capture stamps and arrive in order, so a drop in
            lag delivers a burst. None delivers them at once.
        steer_gain, steer_offset: The real steering is steer_gain x the command plus
            steer_offset (rad): an uncalibrated servo. The nodes only know the command.
        lidar_offset: The LiDAR sits this far ahead of the rear axle (m), as on the car
            (about 0.27 m); the car turns about the rear axle.
        range_noise: Standard deviation of the LiDAR range noise (m), seeded by seed.
        steer_delay: The steering acts this long after it is commanded (s).
        odom_noise: Odometry speed error, a fraction of the speed (standard deviation).
        track_obstacles, margin, smoothing: Override the safety node's track_obstacles and
            the gap follower's margin and smoothing; None keeps the node defaults.
        planner: Other gap_follow_node parameters to override, e.g.
            {'lidar_x': 0.0} for pure pursuit from the LiDAR.

    Returns:
        dict: laps, safety mode, time, distance, closest wall distance, top speed, trace,
        the largest persistent obstacle speed the safety node saw, and the steering jitter
        (mean change of the steering command per scan while moving, rad).
    """
    ros_stubs.CLOCK.seconds = 0.0
    ros_stubs.PARAM_OVERRIDES.clear()
    ros_stubs.PARAM_OVERRIDES.update({
        'fusion_node': {'use_camera': use_camera, 'use_depth': use_camera},
        'gap_follow_node': {'max_speed': max_speed},
        'safety_node': {'drive_speed': max_speed, 'a_brake': a_brake},
        'lap_counter_node': {'use_camera': False, 'num_laps': num_laps,
                             'min_lap_distance': min_lap_distance},
    })
    if track_obstacles is not None:
        ros_stubs.PARAM_OVERRIDES['safety_node']['track_obstacles'] = track_obstacles
    if margin is not None:
        ros_stubs.PARAM_OVERRIDES['gap_follow_node']['margin'] = margin
    if smoothing is not None:
        ros_stubs.PARAM_OVERRIDES['gap_follow_node']['smoothing'] = smoothing
    # the planner knows where the LiDAR is on this car (unless a test overrides it)
    ros_stubs.PARAM_OVERRIDES['gap_follow_node']['lidar_x'] = lidar_offset
    if planner:
        ros_stubs.PARAM_OVERRIDES['gap_follow_node'].update(planner)
    fusion, planner, safety, laps = FusionNode(), GapFollowNode(), SafetyNode(), LapCounterNode()
    ros_stubs.PARAM_OVERRIDES.clear()

    def last(node, topic):
        messages = node.publishers_by_topic[topic].messages
        return messages[-1] if messages else None

    x = y = yaw = 0.0
    v = steer = actual = 0.0
    rng = random.Random(seed)
    noise = np.random.default_rng(seed)
    late = []                                              # scans on their way to safety
    pending = []                                           # steering on its way to the servo
    jitter, last_steer = [], None
    obstacle_peak = 0.0                                    # largest persistent obstacle speed
    distance, closest, top_speed, trace = 0.0, math.inf, 0.0, []
    n_requests = n_stops = 0
    t = 0.0
    while t < max_time:
        ros_stubs.CLOCK.seconds = t
        points = world(t) if callable(world) else world
        lx, ly = x + lidar_offset * math.cos(yaw), y + lidar_offset * math.sin(yaw)   # LiDAR
        measured = v * (1.0 + odom_noise * noise.standard_normal()) if v > 0.0 else v
        odom = ros_stubs.make_odom(measured, measured * math.tan(steer) / WHEELBASE)
        odom.header.stamp = ros_stubs.stamp(t)
        for node in (fusion, safety, laps, planner):
            node.odom_callback(odom)

        # depth camera (30 Hz) -> fusion
        if use_camera and low_obstacles is not None and int(t / STEP) % 4 != 3:
            fusion.depth_callback(depth_scan(low_obstacles, lx, ly, yaw, t))

        # LiDAR -> fusion -> planner and safety (raw /scan goes to everyone too)
        if lidar_dropout is None or not lidar_dropout[0] <= t < lidar_dropout[1]:
            ranges = ros_stubs.world_scan(points, lx, ly, yaw)
            if range_noise > 0.0:
                seen = np.isfinite(ranges)
                ranges[seen] += noise.normal(0.0, range_noise, int(seen.sum()))
            scan = ros_stubs.make_scan(ranges, t=t)
            fusion.lidar_callback(scan)
            fused, obstacles = last(fusion, 'fused_scan'), last(fusion, 'obstacle_scan')
            if box_centres is not None:
                obstacles = box_labels(fused, box_centres, lx, ly, yaw, t, label_probability, rng)
            planner.obstacle_callback(obstacles)
            planner.fused_callback(fused)
            planner.raw_callback(scan)
            if scan_lag is None:
                safety.fused_scan_callback(fused)
                safety.raw_scan_callback(scan)
            else:
                late.append((t + scan_lag(t), fused, scan))
            laps.lidar_callback(scan)
        while late and late[0][0] <= t + 1e-9:
            _, fused_late, raw_late = late.pop(0)
            safety.fused_scan_callback(fused_late)
            safety.raw_scan_callback(raw_late)

        obstacle_peak = max(obstacle_peak, safety.sources['fused'].obstacle_cap)

        # drive requests -> safety
        requests = planner.publishers_by_topic['drive_request'].messages
        for msg in requests[n_requests:]:
            safety.drive_request_callback(msg)
        n_requests = len(requests)
        step = int(round(t / STEP))
        if step % 2 == 0:
            laps.tick()                                    # 20 Hz timer
            safety.watchdog_callback()                     # 20 Hz timer
        if step % 4 == 0:
            laps.stop_timer_callback()                     # 10 Hz timer
        stops = laps.publishers_by_topic['drive_request'].messages
        for msg in stops[n_stops:]:
            safety.drive_request_callback(msg)
        n_stops = len(stops)

        # the car follows /drive
        command = last(safety, 'drive')
        target = command.drive.speed if command else 0.0
        steer = command.drive.steering_angle if command else 0.0
        dv = target - v
        v += max(-a_brake * STEP, min(accel * STEP, dv))
        x += v * STEP * math.cos(yaw)
        y += v * STEP * math.sin(yaw)
        pending.append(steer)
        acting = pending.pop(0) if len(pending) > int(round(steer_delay / STEP)) else 0.0
        real = steer_gain * acting + steer_offset           # what the servo really does
        if v > 0.2 and last_steer is not None:
            jitter.append(abs(steer - last_steer))
        last_steer = steer
        actual += (real - actual) * (min(1.0, STEP / servo_lag) if servo_lag > 0.0 else 1.0)
        yaw += v * math.tan(actual) / WHEELBASE * STEP
        distance += v * STEP
        top_speed = max(top_speed, v)

        # clearance: the car as two circles (at the LiDAR and 0.3 m behind it)
        solid = points if low_obstacles is None else np.vstack([points, low_obstacles])
        for back in (0.0, 0.3):
            cx = x + (lidar_offset - back) * math.cos(yaw)
            cy = y + (lidar_offset - back) * math.sin(yaw)
            closest = min(closest, float(np.min(np.hypot(solid[:, 0] - cx,
                                                         solid[:, 1] - cy))))
        trace.append((t, x, y, v, safety.safety_mode, laps.laps))

        if laps.done and v < 0.01:
            break
        t += STEP

    return {'laps': laps.laps, 'mode': safety.safety_mode, 'time': t, 'distance': distance,
            'obstacle_peak': obstacle_peak,
            'steer_jitter': float(np.mean(jitter)) if jitter else 0.0,
            'closest': closest, 'top_speed': top_speed, 'trace': trace,
            'logs': {name: node.get_logger().records for name, node in
                     (('fusion', fusion), ('planner', planner), ('safety', safety),
                      ('laps', laps))}}


def report(name, result, lap_length):
    """Print a short summary of a run."""
    print(f'{name}: {result["laps"]} laps in {result["time"]:.1f} s, '
          f'{result["distance"]:.1f} m driven (lap {lap_length:.1f} m), '
          f'top speed {result["top_speed"]:.2f} m/s, closest wall {result["closest"]:.3f} m, '
          f'final mode {result["mode"]}')


if __name__ == '__main__':
    track, lap_length = stadium_track()
    gate = 0.75 * lap_length               # more than half a lap: the far straight looks alike
    boxes = np.vstack([track, box(1.5, 0.25), box(-1.0, 3.75)])
    for top in (1.0, 1.5):
        report(f'empty track, max {top} m/s', run(track, min_lap_distance=gate, max_speed=top),
               lap_length)
        report(f'with boxes,  max {top} m/s', run(boxes, min_lap_distance=gate, max_speed=top),
               lap_length)
