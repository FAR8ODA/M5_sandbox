"""
Path-aware AEB safety node for the F1Tenth vehicle (Milestone 4).

The gap-follow node requests steering and speed. This node owns vehicle speed and is
the only node that publishes commands to /drive.

Kept from Milestone 3:
    Odom Sign Fix: abs() on the odometry speed (it can read negative going forward).
    Watchdog: if drive requests or scans stop for stale_timeout, the car brakes.
    Latched Stop: a negative speed on drive_request stops the car for good (lap counter).

New in Milestone 4:
    Path-aware corridor: only points the car would actually hit on its current arc
        (from the commanded steering) count. Walls beside the path in a corner no longer
        trigger AEB, which was our Milestone 3 false-braking problem.
    Variable TTC: AEB fires when the time-to-collision (to the safety margin) is shorter
        than the time the car needs to stop, t_react + v / (2 a_brake) + ttc_margin, which
        grows with our speed and with the obstacle's own speed (braking.required_ttc).
        Faster laps brake earlier (Milestone 2 used a fixed TTC and braked too late).
    Dynamic obstacles (track_obstacles:=true, off by default): the scan from about 0.1 s ago
        is moved by the car's own motion (odometry speed, and the yaw rate of the commanded
        steering) and matched with the current scan, so static walls stay still even while
        turning. An obstacle's own speed only counts on a surface that faces the car, if
        neighbouring points on the path agree on it and it lasts for obstacle_persist_scans
        scans in a row; then it adds to the closing speed and to the variable TTC threshold.
        Short-lived readings (servo lag, timing jitter, LiDAR edge noise) can no longer set
        off AEB in a static scene, and neither can barriers seen at a shallow angle when the
        car steers more or less than commanded.
    Capture times: every scan is matched with the car's pose at the time it was taken (its
        header stamp, from a pose history built from stamped odometry), not when it arrived.
        On a busy Jetson scans can arrive late and in bursts; pairing them with the pose at
        arrival made static walls look like they were rushing at the car. Distances are also
        shortened by how far the car drove since the scan was taken.
    Two stages: below slow_factor times the required TTC the speed is capped so the TTC
        stays there (soft); below the required TTC AEB brakes (hard) and stays latched
        until the node restarts.
    Sensor fallback: /fused_scan is used normally; if it goes stale, the raw /scan.
    Fail safe: any error inside a callback brakes instead of stopping the node.
    Stop on exit: on Ctrl+C (or any other exit) the node sends a zero-speed /drive
        command before it ends, so the car is never left with a moving last command.
    Quiet logs: one message per state change, warnings throttled.
"""

# include needed imports
from __future__ import annotations

from collections import deque
import math
import time

from ackermann_msgs.msg import AckermannDriveStamped
from milestone4.braking import car_motion, path_distances, required_ttc, slow_speed_cap
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class MotionTracker:
    """
    Measure how fast scan points move towards the car by themselves.

    Comparing the same beam index between scans does not work: when the car turns, a beam
    points at a different part of the world each scan, so static walls look like they are
    moving. Instead, the points of a scan about `baseline` seconds old are moved into the
    current frame using the car's own motion between the two scans (the poses dead-reckoned
    from odometry) and sorted into bearing bins. A static object then has the same range at
    the same bearing in both scans; only real movement is left over. For each current point,
    the old range is the nearest one in its bin and the two bins beside it, so sampling gaps
    can only make the old range shorter (no approach), never longer (a false approach).

    A speed is only trusted where the surface is smooth in both scans (the ranges in the
    three bins agree within smooth_tol). Edges, walls seen at a grazing angle and surfaces
    that are just being hidden or revealed are not, and they are where false speeds came
    from; a real obstacle's face is smooth.
    """

    def __init__(self, baseline: float = 0.1, match_gate: float = 0.5,
                 max_speed: float = 6.0, bin_width: float = math.radians(0.25),
                 smooth_tol: float = 0.1) -> None:
        """
        Store the settings.

        Args:
            baseline (float): Compare with a scan at least this old (s); a longer baseline
                averages out the LiDAR's ±4 cm noise.
            match_gate (float): A range change bigger than this is a different object (an
                edge, or something that just came into view), not motion; its speed is
                taken as 0 (m).
            max_speed (float): Largest obstacle speed believed (m/s).
            bin_width (float): Bearing bin width (rad), the LiDAR's resolution.
            smooth_tol (float): Largest range spread over three bins of a smooth surface (m).
        """
        self.baseline = baseline
        self.match_gate = match_gate
        self.max_speed = max_speed
        self.bin_width = bin_width
        self.smooth_tol = smooth_tol
        self.num_bins = int(math.ceil(2.0 * math.pi / bin_width)) + 1
        self.history: deque = deque(maxlen=50)    # (time, x, y, pose) of recent scans

    def add(self, t: float, x: np.ndarray, y: np.ndarray,
            pose: tuple[float, float, float]) -> None:
        """
        Remember a scan's points and where the car was.

        Args:
            t (float): Scan time (s).
            x (np.ndarray): Valid points ahead of the LiDAR (m).
            y (np.ndarray): Valid points left of the LiDAR (m).
            pose (tuple): Car pose (x, y, yaw) dead-reckoned from odometry (m, m, rad).
        """
        if self.history and t < self.history[-1][0]:
            self.history.clear()                    # clock reset (e.g. the simulator restarted)
        self.history.append((t, x, y, pose))
        # keep the newest scan that is at least `baseline` older than this one, and newer ones
        while len(self.history) >= 3 and t - self.history[1][0] >= self.baseline:
            self.history.popleft()

    def approach_speeds(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """
        Return how fast each point of the newest scan moves towards the car on its own.

        Args:
            px (np.ndarray): Points of the newest scan to check, ahead of the LiDAR (m).
            py (np.ndarray): Points of the newest scan to check, left of the LiDAR (m).

        Returns:
            np.ndarray: Approach speed of each point (m/s, >= 0; 0 if unknown).
        """
        speeds = np.zeros(px.size)
        if len(self.history) < 2 or px.size == 0:
            return speeds
        t1, nx, ny, (x1, y1, yaw1) = self.history[-1]
        t0, ox, oy, (x0, y0, yaw0) = self.history[0]
        dt = t1 - t0
        if dt < 0.5 * self.baseline or ox.size == 0:
            return speeds

        # the old points as they would look now if nothing in the world moved:
        # old car frame -> world (pose then) -> current car frame (pose now)
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        wx, wy = x0 + c0 * ox - s0 * oy, y0 + s0 * ox + c0 * oy
        c1, s1 = math.cos(yaw1), math.sin(yaw1)
        dx, dy = wx - x1, wy - y1
        sx, sy = c1 * dx + s1 * dy, -s1 * dx + c1 * dy

        # each point against the nearest old range in its bin and the bins beside it
        b = self.bins(px, py)
        before, old_spread = self.window(self.binned(sx, sy), b)
        _, new_spread = self.window(self.binned(nx, ny), b)
        change = before - np.hypot(px, py)               # > 0: the point came closer
        with np.errstate(invalid='ignore'):
            speeds = np.clip(change / dt, 0.0, self.max_speed)
            unknown = (~np.isfinite(change) | (np.abs(change) > self.match_gate)
                       | ~(old_spread <= self.smooth_tol) | ~(new_spread <= self.smooth_tol))
        speeds[unknown] = 0.0
        return speeds

    def binned(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return the nearest range in every bearing bin (inf where a bin is empty).

        Args:
            x (np.ndarray): Points ahead (m).
            y (np.ndarray): Points left (m).

        Returns:
            np.ndarray: Range per bin (m).
        """
        r = np.full(self.num_bins, np.inf)
        np.minimum.at(r, self.bins(x, y), np.hypot(x, y))
        return r

    def window(self, binned: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the nearest range and the range spread over bins b-1, b and b+1.

        Args:
            binned (np.ndarray): Range per bin, from binned().
            b (np.ndarray): Bin index of each point.

        Returns:
            tuple[np.ndarray, np.ndarray]: (nearest, spread); spread is inf if a bin is empty.
        """
        three = np.stack([binned[np.maximum(b - 1, 0)], binned[b],
                          binned[np.minimum(b + 1, self.num_bins - 1)]])
        with np.errstate(invalid='ignore'):
            spread = three.max(axis=0) - three.min(axis=0)
        spread[~np.isfinite(spread)] = np.inf
        return three.min(axis=0), spread

    def bins(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return the bearing bin of each point.

        Args:
            x (np.ndarray): Points ahead (m).
            y (np.ndarray): Points left (m).

        Returns:
            np.ndarray: Bin index of each point.
        """
        b = np.round((np.arctan2(y, x) + math.pi) / self.bin_width).astype(int)
        return np.clip(b, 0, self.num_bins - 1)


class ScanSource:
    """The latest scan from one topic, prepared for the safety checks."""

    def __init__(self, name: str, persist_scans: int = 4) -> None:
        """
        Start empty.

        Args:
            name (str): 'fused' or 'raw'.
            persist_scans (int): Scans in a row an obstacle speed must last to count.
        """
        self.name = name
        self.time: Time | None = None          # arrival time (node clock)
        self.ranges: np.ndarray | None = None  # ranges with inf where invalid
        self.angles: np.ndarray | None = None
        self.x: np.ndarray | None = None       # every beam as a point (0 where invalid)
        self.y: np.ndarray | None = None
        self.tracker = MotionTracker()
        self.obstacle_history: deque = deque(maxlen=persist_scans)  # per-scan obstacle speed
        self.obstacle_cap = 0.0                # obstacle speed seen in every recent scan (m/s)
        self.agreed: np.ndarray | None = None  # agreed obstacle speed per beam (this scan)
        self.capture = 0.0                     # when the scan was taken (header stamp, s)
        self.lag = 0.0                         # how late it arrived (s)
        self.tracked_at = -math.inf            # when obstacle speeds were last computed (s)


class SafetyNode(Node):
    """Control speed, apply path-aware Automatic Emergency Braking, watch for stale inputs."""

    # Extra time added to the time the car needs to stop (seconds).
    TTC_MARGIN = 0.1

    # Highest speed ever commanded (m/s).
    DRIVE_SPEED = 1.0

    # Brake if drive requests or scans are older than this (seconds).
    STALE_TIMEOUT = 0.25

    # How often the watchdog checks the inputs (seconds).
    WATCHDOG_PERIOD = 0.05

    def __init__(self) -> None:
        """Initialize parameters, ROS communication, safety state and the watchdog."""
        super().__init__('safety_node')

        # The launch file sets these; the class constants are the defaults.
        self.declare_parameter('ttc_margin', self.TTC_MARGIN)
        self.declare_parameter('drive_speed', self.DRIVE_SPEED)
        self.declare_parameter('stale_timeout', self.STALE_TIMEOUT)

        # Braking model and corridor (tune a_brake from a measured stop: a = v^2 / (2 d)).
        self.declare_parameter('a_brake', 2.0)          # braking deceleration (m/s^2)
        self.declare_parameter('t_react', 0.15)         # delay before braking starts (s)
        self.declare_parameter('stop_margin', 0.25)     # gap left in front of the LiDAR (m)
        self.declare_parameter('slow_factor', 2.0)      # soft cap below this x required TTC
        self.declare_parameter('wheelbase', 0.33)       # front to rear axle (m)
        self.declare_parameter('car_half_width', 0.15)  # half the car's width (m)
        self.declare_parameter('path_margin', 0.05)     # extra corridor width each side (m)
        self.declare_parameter('obstacle_deadband', 0.5)  # slower obstacle speeds are noise (m/s)
        self.declare_parameter('track_obstacles', False)  # True: also obstacle speeds
        self.declare_parameter('max_obstacle_speed', 3.0)  # faster readings are not believed (m/s)
        self.declare_parameter('obstacle_persist_scans', 4)  # must last this many scans
        self.declare_parameter('min_closing', 0.1)      # slower closing never triggers AEB (m/s)
        self.declare_parameter('blocked_time', 1.0)     # held at 0 m/s this long: latched stop (s)
        self.declare_parameter('scan_timeout', 0.15)    # fused scan older: use raw /scan (s)
        self.declare_parameter('odom_timeout', 0.5)     # odometry older: use commanded speed (s)

        p = self.get_parameter
        self.TTC_MARGIN = max(0.0, float(p('ttc_margin').value))
        self.drive_speed = float(p('drive_speed').value)
        self.stale_timeout = float(p('stale_timeout').value)
        self.A_BRAKE = max(0.1, float(p('a_brake').value))
        self.T_REACT = max(0.0, float(p('t_react').value))
        self.STOP_MARGIN = float(p('stop_margin').value)
        self.SLOW_FACTOR = max(1.0, float(p('slow_factor').value))
        self.WHEELBASE = float(p('wheelbase').value)
        self.CORRIDOR = float(p('car_half_width').value) + float(p('path_margin').value)
        self.OBSTACLE_DEADBAND = float(p('obstacle_deadband').value)
        self.TRACK_OBSTACLES = bool(p('track_obstacles').value)
        self.MAX_OBSTACLE_SPEED = max(0.0, float(p('max_obstacle_speed').value))
        self.PERSIST_SCANS = max(1, int(p('obstacle_persist_scans').value))
        self.TRACK_RANGE = 6.0          # obstacle speeds are watched up to this far ahead (m)
        self.TRACK_FOV = math.radians(90.0)  # ... and within this angle either side
        self.FACING_SPAN = math.radians(2.5)  # surface slope from the beams this far either side
        self.MAX_INCIDENCE = math.radians(45.0)  # steeper surfaces: no obstacle speed
        self.LAG_WARNING = 0.15         # warn when scans arrive this late (s)
        self.MIN_CLOSING = float(p('min_closing').value)
        self.BLOCKED_TIME = float(p('blocked_time').value)
        self.SCAN_TIMEOUT = float(p('scan_timeout').value)
        self.ODOM_TIMEOUT = float(p('odom_timeout').value)

        self.current_speed = 0.0                        # forward speed of the vehicle
        self.current_yaw_rate = 0.0                     # yaw rate, left positive (rad/s)
        self.pose = (0.0, 0.0, 0.0)                     # dead-reckoned (x, y, yaw), for tracking
        self.pose_time: float | None = None             # time of that pose (odom stamp, s)
        self.pose_history: deque = deque(maxlen=100)    # (time, x, y, yaw), about 1-2 s
        self.last_odom_time: Time | None = None         # when odometry last arrived
        self.latest_steering_angle = 0.0                # latest steering angle request
        self.last_command_speed = 0.0                   # last speed sent to /drive
        self.safety_mode = 'normal'                     # normal, slow, stale, aeb or stopped
        self.last_request_time: Time | None = None      # when the last drive request arrived
        self.blocked_since: Time | None = None          # when the car was first held at 0 m/s
        self.odom_warned = False                        # odometry-missing warning logged
        self.last_check = (math.inf, math.inf, math.inf, math.inf, 0.0)  # path, cap, ttc,
        #                                                                 needed, obstacle speed
        self.sources = {'fused': ScanSource('fused', self.PERSIST_SCANS),
                        'raw': ScanSource('raw', self.PERSIST_SCANS)}

        # create the required publishers and subscribers

        # note: only safety_node publishes to /drive.
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.state_publisher = self.create_publisher(String, 'safety/state', 10)

        # fused LiDAR + camera scan, and the raw LiDAR scan as a fallback
        self.create_subscription(LaserScan, 'fused_scan', self.fused_scan_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.raw_scan_callback, 10)

        # forward speed (remapped to /odom on the car)
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        # requests from the gap-follow node and the lap counter
        self.create_subscription(AckermannDriveStamped, 'drive_request',
                                 self.drive_request_callback, 10)

        # watchdog: brakes when inputs go stale, even if no callback fires
        self.watchdog_timer = self.create_timer(self.WATCHDOG_PERIOD, self.watchdog_callback)

        self.get_logger().info(
            f'safety_node started: drive speed {self.drive_speed} m/s, '
            f'a_brake {self.A_BRAKE} m/s^2, TTC margin {self.TTC_MARGIN} s, '
            f'obstacle tracking {"on" if self.TRACK_OBSTACLES else "off"}')

    # Inputs

    def odom_callback(self, msg: Odometry) -> None:
        """
        Update the measured forward speed.

        Args:
            msg (Odometry): Incoming odometry message.
        """
        self.current_speed = abs(msg.twist.twist.linear.x)
        self.last_odom_time = self.get_clock().now()
        self.current_yaw_rate = self.yaw_rate()
        self.advance_pose(self.current_speed, self.current_yaw_rate,
                          self.msg_time(msg.header.stamp))
        if self.odom_warned:
            self.get_logger().info('odometry back')
            self.odom_warned = False

    def measured_speed(self) -> float:
        """
        Return the speed used for the safety checks.

        Returns:
            float: Odometry speed, or the last commanded speed if odometry is stale.
        """
        if self.age_seconds(self.last_odom_time) <= self.ODOM_TIMEOUT:
            return self.current_speed
        if not self.odom_warned:
            self.get_logger().warn('odometry missing or stale: using the commanded speed')
            self.odom_warned = True
        return self.last_command_speed

    def yaw_rate(self) -> float:
        """
        Return the yaw rate used to remove the car's own motion from the scans.

        It comes from the commanded steering, not from the odometry's angular.z. On the
        F1TENTH stack vesc_to_odom derives angular.z from the servo command anyway, and some
        set-ups publish it flipped (with a backwards speed) or as zero; the model is the same
        everywhere.

        Returns:
            float: speed * tan(commanded steering) / wheelbase (rad/s, left positive).
        """
        return self.measured_speed() * math.tan(self.latest_steering_angle) / self.WHEELBASE

    def msg_time(self, stamp) -> float:
        """
        Return a header stamp in seconds, or the node clock if the stamp cannot be used.

        Args:
            stamp: builtin_interfaces/Time from a message header.

        Returns:
            float: Seconds; the node clock's time if the stamp is 0 (unstamped) or more than
            1 s away from it (a different clock).
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if t <= 0.0 or abs(t - now) > 1.0:
            return now
        return t

    def advance_pose(self, speed: float, yaw_rate: float, t: float) -> None:
        """
        Dead-reckon the car's pose up to time t, and remember it.

        Args:
            speed (float): Forward speed since the last update (m/s).
            yaw_rate (float): Yaw rate since the last update (rad/s).
            t (float): Time of this update (s), the odometry stamp.
        """
        if self.pose_time is not None and t < self.pose_time - 1.0:
            self.pose_history.clear()                 # clock jumped back (bag or sim restart)
            self.pose_time = None
        if self.pose_time is not None:
            dt = min(max(t - self.pose_time, 0.0), 0.1)
            ahead, left, turn = car_motion(speed, yaw_rate, dt)
            x, y, yaw = self.pose
            c, s = math.cos(yaw), math.sin(yaw)
            self.pose = (x + c * ahead - s * left, y + s * ahead + c * left, yaw + turn)
        self.pose_time = max(t, self.pose_time) if self.pose_time is not None else t
        self.pose_history.append((self.pose_time,) + self.pose)

    def pose_at(self, t: float) -> tuple[float, float, float]:
        """
        Return the car's dead-reckoned pose at time t (when a scan was taken).

        Interpolated between the odometry updates around t; past the newest one, moved on
        at the current speed and yaw rate (at most 0.2 s).

        Args:
            t (float): Time (s).

        Returns:
            tuple[float, float, float]: (x, y, yaw).
        """
        if not self.pose_history:
            return self.pose
        newest = self.pose_history[-1]
        if t >= newest[0]:
            dt = min(t - newest[0], 0.2)
            ahead, left, turn = car_motion(self.current_speed, self.current_yaw_rate, dt)
            _, x, y, yaw = newest
            c, s = math.cos(yaw), math.sin(yaw)
            return (x + c * ahead - s * left, y + s * ahead + c * left, yaw + turn)
        oldest = self.pose_history[0]
        if t <= oldest[0]:
            return oldest[1:]
        for before, after in zip(reversed(list(self.pose_history)[:-1]),
                                 reversed(list(self.pose_history)[1:])):
            if before[0] <= t <= after[0]:
                span = after[0] - before[0]
                k = (t - before[0]) / span if span > 0.0 else 1.0
                return tuple(b + k * (a - b) for b, a in zip(before[1:], after[1:]))
        return newest[1:]

    def fused_scan_callback(self, msg: LaserScan) -> None:
        """Store and check a fused scan."""
        self.scan_callback(msg, 'fused')

    def raw_scan_callback(self, msg: LaserScan) -> None:
        """Store and check a raw LiDAR scan."""
        self.scan_callback(msg, 'raw')

    def scan_callback(self, msg: LaserScan, name: str) -> None:
        """
        Store a scan and immediately brake if the current motion is unsafe.

        Args:
            msg (LaserScan): Incoming scan.
            name (str): 'fused' or 'raw'.
        """
        try:
            source = self.sources[name]
            now = self.get_clock().now()
            now_s = now.nanoseconds * 1e-9
            taken = self.msg_time(msg.header.stamp)        # when the scan was taken

            ranges = np.array(msg.ranges, dtype=np.float64)
            with np.errstate(invalid='ignore'):
                bad = ~np.isfinite(ranges) | (ranges < msg.range_min) | (ranges > msg.range_max)
            ranges[bad] = np.inf
            angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
            finite = np.isfinite(ranges)
            safe_r = np.where(finite, ranges, 0.0)

            source.time = now
            source.capture = taken
            source.lag = max(0.0, now_s - taken)
            source.ranges = ranges
            source.angles = angles
            source.x = safe_r * np.cos(angles)
            source.y = safe_r * np.sin(angles)
            if self.age_seconds(self.last_odom_time) > self.ODOM_TIMEOUT:
                self.advance_pose(self.measured_speed(), self.yaw_rate(), now_s)  # no odometry
            # the scan is matched with where the car was when it was taken, not on arrival
            source.tracker.add(taken, source.x[finite], source.y[finite], self.pose_at(taken))
            if source.lag > self.LAG_WARNING:
                self.get_logger().warn(f'scans arrive {source.lag:.2f} s late (CPU overloaded?)',
                                       throttle_duration_sec=2.0)

            # obstacle speeds: once per scan, for the scan in use only (drive requests reuse
            # them). An obstacle speed only counts once it has been seen in every recent scan,
            # in a fixed region ahead (not on the steered arc, which moves with the steering)
            if self.active_source() is source:
                self.update_obstacle_speeds(source, now_s)

            # only the scan in use can trigger AEB
            if self.is_latched() or self.active_source() is not source:
                return
            hard = self.check_path(source, self.latest_steering_angle, self.measured_speed())[0]
            if hard:
                self.engage_aeb()

        except Exception as exc:
            self.fail_safe('scan', exc)

    def update_obstacle_speeds(self, source: ScanSource, now_s: float) -> None:
        """
        Work out the agreed obstacle speed of every beam ahead, and its persistence.

        Args:
            source (ScanSource): The scan in use, just stored.
            now_s (float): Node clock time (s).
        """
        if now_s - source.tracked_at > 0.2:
            source.obstacle_history.clear()              # this source was not in use: restart
        source.tracked_at = now_s
        ranges, angles = source.ranges, source.angles
        ahead = np.nonzero(np.isfinite(ranges) & (source.x > 0.0) & (ranges <= self.TRACK_RANGE)
                           & (np.abs(angles) <= self.TRACK_FOV))[0]
        source.agreed = np.zeros(ranges.size)
        source.agreed[ahead] = self.agreed_speeds(source, ahead)
        source.obstacle_history.append(float(np.max(source.agreed, initial=0.0)))
        full = len(source.obstacle_history) == source.obstacle_history.maxlen
        source.obstacle_cap = min(source.obstacle_history) if full else 0.0

    def active_source(self) -> ScanSource | None:
        """
        Return the scan to use: fused if fresh, otherwise raw if fresh.

        Returns:
            ScanSource | None: The scan, or None if both are stale.
        """
        fused, raw = self.sources['fused'], self.sources['raw']
        if self.age_seconds(fused.time) <= self.SCAN_TIMEOUT:
            return fused
        if self.age_seconds(raw.time) <= self.stale_timeout:
            return raw
        return None

    # Safety checks

    def check_path(self, source: ScanSource, steering: float,
                   speed: float) -> tuple[bool, float, float, float, float]:
        """
        Check the points on the car's path against the variable TTC threshold.

        Args:
            source (ScanSource): Scan to check.
            steering (float): Steering angle of the path (rad).
            speed (float): Current speed (m/s).

        Returns:
            tuple: (hard, speed_cap, path_distance, ttc, needed_ttc)
                hard: True if AEB must engage now (ttc <= required TTC at some point)
                speed_cap: highest speed the soft stage allows (m/s)
                path_distance: nearest point on the path (m), inf if none
                ttc: TTC of the most urgent point, to the safety margin (s), inf if none
                needed_ttc: the required TTC (the AEB threshold) at that point (s)
        """
        index, along, agreed = self.path_obstacle_speeds(source, steering)
        if index.size == 0:
            self.last_check = (math.inf, math.inf, math.inf, math.inf, 0.0)
            return False, math.inf, math.inf, math.inf, math.inf

        # an obstacle moving towards us on its own adds its speed (our own motion removed),
        # but only as much as has been seen in every recent scan
        extra = np.minimum(agreed, source.obstacle_cap)
        closing = speed + extra

        # TTC to the safety margin, and the TTC the car needs at this speed (variable). The
        # car has driven on since the scan was taken, so the gap is that much shorter
        age = max(0.0, self.get_clock().now().nanoseconds * 1e-9 - source.capture)
        gap = along - self.STOP_MARGIN - speed * age
        needed = np.broadcast_to(
            required_ttc(speed, extra, self.T_REACT, self.A_BRAKE, self.TTC_MARGIN), gap.shape)
        moving = closing > self.MIN_CLOSING
        ttc = np.full(index.size, math.inf)
        ttc[moving] = gap[moving] / closing[moving]

        # hard stage: AEB if any point is nearer in time than the car needs to stop
        hard = bool(np.any(moving & (ttc <= needed)))

        # soft stage: the highest speed that keeps every point at slow_factor x the needed TTC
        cap = float(np.min(slow_speed_cap(gap, extra, self.SLOW_FACTOR, self.T_REACT,
                                          self.A_BRAKE, self.TTC_MARGIN)))

        # the most urgent point (smallest ttc / needed), for the log and /safety/state
        urgency = np.where(moving, ttc / needed, np.inf)
        worst = int(np.argmin(urgency))
        nearest = float(along.min())
        self.last_check = (nearest, cap, float(ttc[worst]), float(needed[worst]),
                           float(extra[worst]))
        return hard, cap, nearest, float(ttc[worst]), float(needed[worst])

    def path_obstacle_speeds(self, source: ScanSource,
                             steering: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the path points and the obstacle speed their neighbours agree on.

        The speeds come from update_obstacle_speeds(), computed once per scan for the beams ahead:
        each is the median of a point and its two neighbours on each side (agreed_speeds),
        since a real obstacle's face covers many beams and a glitch only one or two.

        Args:
            source (ScanSource): Scan to check.
            steering (float): Steering angle of the path (rad).

        Returns:
            tuple: (beam indices, distance along the path in m, agreed obstacle speed in m/s)
        """
        index, along = path_distances(source.ranges, source.angles, steering,
                                      self.WHEELBASE, self.CORRIDOR)
        if source.agreed is None or source.agreed.size != source.ranges.size:
            return index, along, np.zeros(index.size)
        return index, along, source.agreed[index]        # computed once per scan

    def agreed_speeds(self, source: ScanSource, index: np.ndarray) -> np.ndarray:
        """
        Return the obstacle speed of each point that its neighbours agree on.

        Args:
            source (ScanSource): Scan the points belong to.
            index (np.ndarray): Beam indices of the points, in beam order.

        Returns:
            np.ndarray: Median over each point and two neighbours on each side, capped at
            max_obstacle_speed, minus the deadband (m/s, >= 0); zeros if tracking is off.
            Points on a surface that does not face the car count as 0 (see facing()).
        """
        if index.size == 0 or not self.TRACK_OBSTACLES:
            return np.zeros(index.size)
        own = source.tracker.approach_speeds(source.x[index], source.y[index])
        own[~self.facing(source)[index]] = 0.0           # barriers seen at a shallow angle
        padded = np.pad(own, 2, mode='edge')
        agreed = np.median(np.stack([padded[k:k + own.size] for k in range(5)]), axis=0)
        agreed = np.clip(agreed, 0.0, self.MAX_OBSTACLE_SPEED)
        return np.maximum(0.0, agreed - self.OBSTACLE_DEADBAND)

    def facing(self, source: ScanSource) -> np.ndarray:
        """
        Return which beams hit a surface that faces the car.

        The car's own turning comes from the commanded steering. When the car really steers
        a little more or less (servo calibration, servo lag), the scans are lined up with a
        small heading error. On a barrier seen at a shallow angle, a small heading error is a
        large range change, so the barrier seemed to rush at the car (1.2 to 3.5 m/s in the
        car tests, on a static track). A surface that faces the car barely changes range, and
        an obstacle driving at the car shows it its front, so speeds are only measured where
        the angle of incidence is at most MAX_INCIDENCE.

        The angle comes from the range slope over FACING_SPAN either side of each beam:
        tan(incidence) = |dr/dtheta| / r. Beams near a missing return or an edge do not face
        the car.

        Args:
            source (ScanSource): The scan.

        Returns:
            np.ndarray: True for beams on a surface facing the car.
        """
        ranges = source.ranges
        out = np.zeros(ranges.size, dtype=bool)
        if ranges.size < 3:
            return out
        step = abs(float(source.angles[1] - source.angles[0]))
        k = max(1, int(round(self.FACING_SPAN / step))) if step > 0.0 else 1
        if ranges.size <= 2 * k:
            return out
        slope_limit = math.tan(self.MAX_INCIDENCE) * 2 * k * step
        with np.errstate(invalid='ignore'):
            change = np.abs(ranges[2 * k:] - ranges[:-2 * k])
            out[k:-k] = change <= slope_limit * ranges[k:-k]
        return out

    def drive_request_callback(self, msg: AckermannDriveStamped) -> None:
        """
        Forward a drive request at a safe speed, or stop.

        Args:
            msg (AckermannDriveStamped): Steering and speed request. A negative speed
                requests a latched stop; zero leaves the speed at drive_speed; a positive
                speed is used, never above drive_speed or the soft-stage cap.
        """
        try:
            self.handle_request(msg)
        except Exception as exc:
            self.fail_safe('drive request', exc)

    def handle_request(self, msg: AckermannDriveStamped) -> None:
        """
        Apply the safety checks to one drive request and publish the command.

        Args:
            msg (AckermannDriveStamped): Steering and speed request.
        """
        self.latest_steering_angle = msg.drive.steering_angle
        self.last_request_time = self.get_clock().now()

        # a negative speed requests a permanent stop (lap counter done)
        if msg.drive.speed < 0.0 and not self.is_latched():
            self.set_mode('stopped', 'latched stop requested (laps done)')

        # latched: once stopped, stay stopped until the node restarts
        if self.is_latched():
            self.emergency_brake()
            return

        # if a scan or the requests are stale, the watchdog handles braking
        source = self.active_source()
        if source is None or not self.inputs_fresh():
            return
        if self.safety_mode == 'stale':
            self.set_mode('normal', 'inputs fresh again: resuming')

        speed = self.drive_speed
        if msg.drive.speed > 0.0:
            speed = min(self.drive_speed, msg.drive.speed)

        # hard stage: AEB if the car cannot stop in time on the requested path
        hard, cap = self.check_path(source, self.latest_steering_angle,
                                    self.measured_speed())[:2]
        if hard:
            self.engage_aeb()
            return

        # soft stage: slow down so the car can still stop
        if cap < speed:
            speed = max(cap, 0.0)
            self.set_mode('slow', 'slowing for an obstacle on the path', throttle=2.0)
        elif self.safety_mode == 'slow':
            self.set_mode('normal', 'path clear: full speed', throttle=2.0)

        # held at 0 m/s by an obstacle for blocked_time: the path is blocked, stop for good
        now = self.get_clock().now()
        if speed < 0.05 and self.measured_speed() < 0.05:
            self.blocked_since = self.blocked_since or now
            if (now - self.blocked_since).nanoseconds * 1e-9 >= self.BLOCKED_TIME:
                self.engage_aeb(reason='path blocked')
                return
        else:
            self.blocked_since = None

        self.publish_drive_command(speed, self.latest_steering_angle)

    def age_seconds(self, stamp: Time | None) -> float:
        """
        Return how long ago an input arrived.

        Args:
            stamp (Time | None): Arrival time of the input, or None if never.

        Returns:
            float: Age in seconds, or inf if the input never arrived.
        """
        if stamp is None:
            return math.inf
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def inputs_fresh(self) -> bool:
        """
        Check that the drive requests and at least one scan are recent.

        Returns:
            bool: True if both arrived within stale_timeout.
        """
        return (self.age_seconds(self.last_request_time) <= self.stale_timeout
                and self.active_source() is not None)

    def watchdog_callback(self) -> None:
        """Command zero speed whenever an input is stale, and publish the safety state."""
        try:
            self.check_inputs()
        except Exception as exc:
            self.fail_safe('watchdog', exc)

    def check_inputs(self) -> None:
        """Brake while an input is stale; log once when the inputs go stale."""
        self.publish_state()
        if self.inputs_fresh():
            return

        if self.safety_mode in ('normal', 'slow'):
            request_age = self.age_seconds(self.last_request_time)
            scan_age = self.age_seconds(self.sources['fused'].time)
            self.set_mode('stale', f'inputs stale (request {request_age:.2f}s, '
                                   f'fused scan {scan_age:.2f}s): braking', warn=True)
        self.emergency_brake()

    # Outputs

    def is_latched(self) -> bool:
        """Return True once AEB or a latched stop has stopped the car for good."""
        return self.safety_mode in ('aeb', 'stopped')

    def set_mode(self, mode: str, message: str, warn: bool = False,
                 throttle: float = 0.0) -> None:
        """
        Change the safety mode, logging only when it changes.

        Args:
            mode (str): New mode.
            message (str): Log message for the change.
            warn (bool): Log as a warning instead of info.
            throttle (float): Log at most once per this many seconds (0 = always).
        """
        if mode == self.safety_mode:
            return
        self.safety_mode = mode
        # rclpy treats each logging line as one call site that must always use the same
        # severity and filters (it raises ValueError otherwise), so each kind has its own line
        if warn:
            self.get_logger().warn(message)
        elif throttle > 0.0:
            self.get_logger().info(message, throttle_duration_sec=throttle)
        else:
            self.get_logger().info(message)

    def fail_safe(self, where: str, exc: Exception) -> None:
        """
        Brake after an unexpected error, instead of letting the node stop.

        Args:
            where (str): Which callback failed, for the log.
            exc (Exception): The error.
        """
        self.get_logger().error(f'safety {where} failed, braking: {exc}',
                                throttle_duration_sec=1.0)
        try:
            self.emergency_brake()
        except Exception:
            pass

    def engage_aeb(self, reason: str = 'obstacle on the path') -> None:
        """
        Engage Automatic Emergency Braking and latch it.

        Args:
            reason (str): Why AEB engaged, for the log.
        """
        if not self.is_latched():
            nearest, _, ttc, needed, obstacle = self.last_check
            moving = f'approaching at {obstacle:.1f} m/s' if obstacle > 0.0 else 'static'
            self.set_mode('aeb', f'AEB engaged ({reason}): TTC {ttc:.2f} s '
                                 f'(needs {needed:.2f} s), path distance {nearest:.2f} m, '
                                 f'speed {self.current_speed:.2f} m/s, obstacle {moving}',
                          warn=True)
        self.emergency_brake()

    def publish_state(self) -> None:
        """Publish the safety mode and the last check on /safety/state, for debugging."""
        nearest, cap, ttc, needed, obstacle = self.last_check
        source = self.active_source()
        self.state_publisher.publish(String(data=(
            f'{self.safety_mode} ttc={ttc:.2f}s aeb<{needed:.2f}s '
            f'slow<{needed * self.SLOW_FACTOR:.2f}s path={nearest:.2f}m cap={cap:.2f}m/s '
            f'obs={obstacle:.1f}m/s lag={1000.0 * source.lag if source else 0.0:.0f}ms '
            f'scan={source.name if source else "none"}')))

    def publish_drive_command(self, speed: float, steering_angle: float) -> None:
        """
        Publish the final command to /drive.

        Args:
            speed (float): Target speed, in m/s.
            steering_angle (float): Target steering angle, in radians.
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        self.drive_publisher.publish(drive_msg)
        self.last_command_speed = float(speed)

    def emergency_brake(self) -> None:
        """Command zero speed while preserving steering."""
        self.publish_drive_command(0.0, self.latest_steering_angle)

    def stop_on_exit(self, repeats: int = 5, period: float = 0.02) -> None:
        """
        Send a zero-speed /drive command a few times before the node ends.

        When the launch is stopped with Ctrl+C every node shuts down, so the last /drive
        command is the last thing the car hears. It is repeated (and given a moment to be
        delivered) so the car is stopped, not left creeping on its previous command.

        Args:
            repeats (int): How many times to send the stop.
            period (float): Time between them (s).
        """
        for _ in range(repeats):
            try:
                self.emergency_brake()
            except Exception:
                return                  # ROS is already shut down; nothing more can be sent
            time.sleep(period)


def init_ros(args=None) -> None:
    """
    Start rclpy so that Ctrl+C reaches main() while ROS can still publish.

    From Humble on, rclpy's own Ctrl+C handler shuts ROS down before main() sees the
    KeyboardInterrupt, which would make the final stop command impossible to send. That
    handler is switched off where it exists; Foxy does not have it and needs nothing.

    Args:
        args: Arguments for rclpy.init, or None for the process arguments.
    """
    try:
        from rclpy.signals import SignalHandlerOptions
    except ImportError:
        rclpy.init(args=args)
        return
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)


def main(args=None) -> None:
    """Run the safety node; on Ctrl+C (or any exit), stop the car, then end."""
    init_ros(args)                 # initialize ROS 2
    safety_node = SafetyNode()     # create instance of SafetyNode
    try:
        rclpy.spin(safety_node)    # spin to handle callbacks
    except KeyboardInterrupt:
        safety_node.get_logger().info('Ctrl+C: stopping the car before exiting')
    finally:
        safety_node.stop_on_exit()     # zero speed on /drive, a few times
        safety_node.destroy_node()     # destroy safety node
        if rclpy.ok():
            rclpy.shutdown()           # shutdown ROS 2


if __name__ == '__main__':
    main()
