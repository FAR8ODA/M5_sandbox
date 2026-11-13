"""
Disparity-extender gap following for Milestone 4 (LiDAR-primary, camera-aware).

Our gap_follow_node subscribes to /fused_scan from fusion_node: the LiDAR scan with any
trusted camera-only obstacles written in. /obstacle_scan marks the beams the camera
confirmed as cardboard boxes. If /fused_scan stops arriving, the raw LiDAR /scan is used.

The algorithm (Gap Following lecture, FTG Tweaks 2-4):
    a. clean the scan and cap long ranges at MAX_RANGE (Tweak 4: less wiggle)
    b. extend every disparity by half the car's width plus a margin (Tweak 2);
       camera-confirmed boxes get a wider margin (BOX_MARGIN). Missing returns (black
       barriers often give none at their edges) are skipped, so they cannot hide an edge
    c. aim for the centre of the deepest run of beams (Tweak 4)
    d. steer to that point with pure pursuit (as in Milestone 3), measured from the rear
       axle: the LiDAR sits lidar_x ahead of it, and treating the LiDAR as the rear axle
       made every turn too sharp, which also pulled the car towards inner walls
    e. if something is right beside the car on the side it is turning towards,
       go straight instead (Tweak 3: disparity extender cornering)
    f. choose the speed from physics (variable speed), the lowest of:
         - max_speed
         - a braking limit: keep the TTC to whatever is on the planned arc at
           plan_factor x the time the car needs to stop (braking.slow_speed_cap)
         - a cornering limit: sqrt(a_lat / curvature) for the steering angle
         - pass_speed while passing a box the camera confirmed
       then let it rise by at most a_accel per second (it can drop at once)

Finally, gap_follow_node publishes the steering and speed on /drive_request.
This is received by safety_node, which caps the speed and applies AEB.

For debugging, every decision is also published on gap_follow/debug as one line of JSON
(only while something is subscribed, e.g. ros2 bag record): the target, the steering before
and after the side check, the free distance, each speed limit and which one set the speed.
This node logs nothing (only an unexpected error, once).
"""

# include needed imports
from __future__ import annotations

from collections import deque
import math

from ackermann_msgs.msg import AckermannDriveStamped
from milestone4.braking import path_distances, slow_speed_cap
from milestone4.debug_log import encode
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# Disparity Extender (UNC Chapel Hill, F1TENTH CPSWeek 2019; lecture slides 23-30)

# [ Step 1 ] find disparities: neighbouring beams whose ranges differ by more than
#            DISPARITY_THRESHOLD; the nearer side is an obstacle edge (beams with no
#            return in between are skipped)
# [ Step 2 ] extend each disparity: overwrite the farther side with the nearer range
#            for as many beams as it takes to cover half the car's width at that range
#            n = ceil(atan(width / r_near) / angle_increment)
# [ Step 3 ] the car can now drive straight down any remaining beam; take the deepest
# [ Step 4 ] cornering: never turn into something beside the car


def describe(distance: float, angle: float) -> str:
    """
    Describe where a point is, for gap_follow/debug, e.g. '0.9 m ahead, 35 deg left'.

    Args:
        distance (float): Range (m).
        angle (float): Bearing (rad, positive is left).

    Returns:
        str: Distance and side.
    """
    degrees = math.degrees(angle)
    side = 'straight' if abs(degrees) < 2.0 else ('left' if degrees > 0.0 else 'right')
    return f'{distance:.1f} m ahead, {abs(degrees):.0f} deg {side}'


class GapFollowNode(Node):
    """
    Disparity-extender controller on the fused LiDAR scan.

    Publishes steering and a requested speed on drive_request; the safety node caps
    the speed, applies AEB and stops the car if these messages stop.
    """

    def __init__(self) -> None:
        """Initialize parameters, state, subscribers and publisher."""
        super().__init__('gap_follow_node')

        # Main tuning parameters (values in config/milestone4.yaml):
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('min_speed', 0.35)
        self.declare_parameter('pass_speed', 0.8)
        self.declare_parameter('gain', 1.0)
        self.declare_parameter('lookahead', 2.0)

        # Disparity extender parameters:
        self.declare_parameter('fov_deg', 90.0)
        self.declare_parameter('max_range', 5.0)
        self.declare_parameter('disparity_threshold', 0.3)
        self.declare_parameter('car_half_width', 0.15)
        self.declare_parameter('margin', 0.20)
        self.declare_parameter('box_margin', 0.25)
        # 0.95 aims at the sliver of beams just past an inner corner, so the car rides about
        # half the car plus margin from the inner wall. Lower values (0.7) move it 6-7 cm out in
        # the closed-loop simulator, but when the view ahead is short they also count a nearby
        # obstacle as "deep" and aim at it (test_target_avoids_a_box_on_the_right), so 0.95 stays
        self.declare_parameter('deep_fraction', 0.95)
        self.declare_parameter('min_gap_range', 0.4)

        # Steering parameters:
        self.declare_parameter('lidar_x', 0.27)
        self.declare_parameter('min_lookahead', 0.5)
        self.declare_parameter('lookahead_time', 0.8)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steering', 0.4)
        self.declare_parameter('smoothing', 0.6)
        self.declare_parameter('side_angle_deg', 80.0)
        self.declare_parameter('side_max_angle_deg', 120.0)
        self.declare_parameter('side_min_range', 0.10)
        self.declare_parameter('side_clear', 0.25)
        self.declare_parameter('centre_gain', 2.0)

        # Speed parameters:
        self.declare_parameter('a_lat', 2.5)
        self.declare_parameter('a_accel', 2.0)
        self.declare_parameter('plan_factor', 2.5)

        # Braking model, shared with safety_node (the /** section of config/milestone4.yaml):
        self.declare_parameter('a_brake', 2.0)
        self.declare_parameter('t_react', 0.15)
        self.declare_parameter('ttc_margin', 0.1)
        self.declare_parameter('stop_margin', 0.25)
        self.declare_parameter('path_margin', 0.05)
        self.declare_parameter('odom_timeout', 0.5)
        self.declare_parameter('box_slow_range', 1.5)
        self.declare_parameter('scan_timeout', 0.15)
        self.declare_parameter('box_hold', 0.3)
        self.declare_parameter('pass_hold', 0.5)

        p = self.get_parameter

        # Disparity extender parameters:

        self.FOV = math.radians(float(p('fov_deg').value))   # ± view searched for a target
        self.MAX_RANGE = float(p('max_range').value)         # cap long ranges here (Tweak 4) (m)
        self.DISPARITY_THRESHOLD = float(p('disparity_threshold').value)  # disparity jump (m)
        self.CAR_HALF_WIDTH = float(p('car_half_width').value)  # half the car's width (m)
        self.MARGIN = float(p('margin').value)               # clearance at a disparity (m)
        self.BOX_MARGIN = float(p('box_margin').value)       # clearance at a confirmed box (m)
        self.DEEP_FRACTION = float(p('deep_fraction').value)  # near-deepest beams count too
        self.MIN_GAP_RANGE = float(p('min_gap_range').value)  # shallower deepest: no gap (m)
        self.SURFACE_SPAN = math.radians(60.0)               # obstacle surface searched (rad)
        self.LIDAR_X = float(p('lidar_x').value)             # LiDAR ahead of the rear axle (m)

        # Steering parameters:

        self.STEERING_GAIN = float(p('gain').value)          # scales the pure-pursuit angle
        self.LOOKAHEAD = float(p('lookahead').value)         # longest pure-pursuit distance (m)
        self.MIN_LOOKAHEAD = float(p('min_lookahead').value)  # shortest pure-pursuit distance (m)
        self.LOOKAHEAD_TIME = float(p('lookahead_time').value)  # lookahead = speed x this (s)
        self.WHEELBASE = float(p('wheelbase').value)         # front to rear axle (m)
        self.MAX_STEERING = float(p('max_steering').value)   # ±0.4 rad (±22.9°) car limit
        # weight of the previous command. MS3 drove this track with 0.7 on the previous command;
        # 0.3 (almost no smoothing) made the steering twitch on the car with every scan
        self.SMOOTHING = float(p('smoothing').value)
        self.SIDE_ANGLE = math.radians(float(p('side_angle_deg').value))  # side region starts here
        self.SIDE_CLEAR = float(p('side_clear').value)       # nearer beside us: no turn (m)
        self.SIDE_WARN = 0.15                                # this much earlier: slow down (m)
        self.SIDE_MAX_ANGLE = math.radians(float(p('side_max_angle_deg').value))  # region end
        self.SIDE_MIN_RANGE = float(p('side_min_range').value)  # nearer: the car itself (m)
        self.CENTRE_GAIN = float(p('centre_gain').value)     # pull to the middle of the lane
        self.LANE_MAX = 1.5                                  # a side farther away is open (m)

        # Speed parameters:

        self.MAX_SPEED = float(p('max_speed').value)         # speed with a clear straight (m/s)
        self.MIN_SPEED = float(p('min_speed').value)         # slowest requested speed (m/s)
        self.PASS_SPEED = float(p('pass_speed').value)       # cap while passing a box (m/s)
        self.A_LAT = max(0.1, float(p('a_lat').value))        # sideways grip in corners (m/s^2)
        self.A_ACCEL = max(0.1, float(p('a_accel').value))    # fastest speed-up (m/s^2)
        self.PLAN_FACTOR = max(1.0, float(p('plan_factor').value))  # x the TTC needed to stop
        self.A_BRAKE = max(0.1, float(p('a_brake').value))    # braking deceleration (m/s^2)
        self.T_REACT = max(0.0, float(p('t_react').value))    # delay before braking (s)
        self.TTC_MARGIN = max(0.0, float(p('ttc_margin').value))  # extra time to stop (s)
        self.STOP_MARGIN = float(p('stop_margin').value)      # gap kept in front (m)
        self.CORRIDOR = self.CAR_HALF_WIDTH + float(p('path_margin').value)  # as safety (m)
        self.ODOM_TIMEOUT = float(p('odom_timeout').value)   # older odometry is not used (s)
        self.BOX_SLOW_RANGE = float(p('box_slow_range').value)  # slow for nearer boxes (m)
        self.SCAN_TIMEOUT = float(p('scan_timeout').value)   # fused scan older: use raw /scan (s)
        self.BOX_HOLD = float(p('box_hold').value)           # keep a box label this long (s)
        self.PASS_HOLD = float(p('pass_hold').value)         # stay in passing this long (s)

        # What the node remembers between scans:

        self.previous_steering = 0.0    # previous steering cmd, initialized to straight ahead
        self.previous_speed = 0.0       # previous speed request (m/s), sets the lookahead
        self.speed_time: Time | None = None        # when the previous speed was requested
        self.measured_speed = 0.0       # speed from odometry (m/s), sets the lookahead
        self.odom_time: Time | None = None         # when odometry last arrived
        self.target_angle = 0.0         # bearing of the last target (rad)
        self.obstacles: np.ndarray | None = None   # latest /obstacle_scan ranges
        self.obstacle_time: Time | None = None     # when it arrived
        self.label_history: deque = deque(maxlen=3)  # recent raw box labels (per beam)
        self.box_until: np.ndarray | None = None   # per beam: box label held until (s)
        self.pass_until = -math.inf     # stay in passing mode until this time (s)
        self.fused_time: Time | None = None        # when the last fused scan arrived
        self.passing = False            # slowing for a confirmed box (on the last scan)
        self.box_where = ''             # where the nearest box ahead was, for gap_follow/debug

        # create subscribers and publisher
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            'drive_request',
            10)  # steering + speed request, safety node caps the speed
        self.debug_publisher = self.create_publisher(String, 'gap_follow/debug', 10)

        self.create_subscription(LaserScan, 'fused_scan', self.fused_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.raw_callback, 10)
        self.create_subscription(LaserScan, 'obstacle_scan', self.obstacle_callback, 10)
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

    # Inputs

    def age(self, stamp: Time | None) -> float:
        """Return seconds since stamp (inf if None)."""
        if stamp is None:
            return math.inf
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def odom_callback(self, msg: Odometry) -> None:
        """Store the measured speed (abs: the car's odometry can read negative going forward)."""
        self.measured_speed = abs(msg.twist.twist.linear.x)
        self.odom_time = self.get_clock().now()

    def lookahead_speed(self) -> float:
        """
        Speed that sets the pure-pursuit lookahead.

        The car's measured speed, which only drops as fast as the car can brake. Using the
        requested speed instead made the lookahead collapse the moment the planner asked to
        slow down, which sharpened the steering towards the inside wall and set off AEB.

        Returns:
            speed: odometry speed, or the previous request if odometry is stale (m/s)
        """
        if self.age(self.odom_time) <= self.ODOM_TIMEOUT:
            return self.measured_speed
        return self.previous_speed

    def obstacle_callback(self, msg: LaserScan) -> None:
        """
        Store the beams the fusion node marked as boxes, debounced.

        The camera's box labels can come and go from one frame to the next. A beam only
        counts as a box once it was labelled in 2 of the last 3 /obstacle_scan messages, and
        then it stays labelled for BOX_HOLD seconds. Without this, the box margin and the
        passing speed flickered, which jittered the steering and flooded the log.

        Args:
            msg: /obstacle_scan from the fusion node
        """
        self.obstacles = np.asarray(msg.ranges, dtype=float)
        self.obstacle_time = self.get_clock().now()
        with np.errstate(invalid='ignore'):
            raw = np.isfinite(self.obstacles) & (self.obstacles > 0.0)
        if self.label_history and self.label_history[-1].size != raw.size:
            self.label_history.clear()
            self.box_until = None
        self.label_history.append(raw)
        needed = min(2, len(self.label_history))
        votes = np.sum(np.stack(list(self.label_history)), axis=0) >= needed
        now = self.obstacle_time.nanoseconds * 1e-9
        if self.box_until is None:
            self.box_until = np.full(raw.size, -math.inf)
        self.box_until = np.where(votes, now + self.BOX_HOLD, self.box_until)

    def fused_callback(self, msg: LaserScan) -> None:
        """Drive on the fused scan (the normal case)."""
        self.fused_time = self.get_clock().now()
        self.drive_on(msg, 'fused')

    def raw_callback(self, msg: LaserScan) -> None:
        """Drive on the raw LiDAR scan only while the fused scan is stale."""
        if self.age(self.fused_time) > self.SCAN_TIMEOUT:
            self.drive_on(msg, 'raw')

    def box_beams(self, num_beams: int) -> np.ndarray:
        """
        Beams of the current scan that count as camera-confirmed boxes (debounced).

        Args:
            num_beams (int): Number of beams in the scan being processed.

        Returns:
            np.ndarray: Boolean mask; all False if there are no labels or another scan size.
        """
        if self.box_until is None or self.box_until.size != num_beams:
            return np.zeros(num_beams, dtype=bool)
        return self.box_until > self.get_clock().now().nanoseconds * 1e-9

    # [ Step 1 ] clean the scan

    def preprocess_scan(self, scan: LaserScan) -> tuple[np.ndarray, np.ndarray]:
        """
        Clean up invalid readings and cap the ranges (the whole scan is kept).

        Args:
            scan: LaserScan message

        Returns:
            ranges: cleaned ranges in meters        (0.0 means invalid)
            angles: angle in radians of each range  (positive angle indicates left)
        """
        ranges = np.asarray(scan.ranges, dtype=float)
        angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment

        with np.errstate(invalid='ignore'):
            # no return (inf, or beyond range_max) means open space: cap it
            ranges = np.where(np.isposinf(ranges) | (ranges > scan.range_max),
                              self.MAX_RANGE, ranges)
            # NaN and readings below range_min are invalid
            bad = ~np.isfinite(ranges) | (ranges < scan.range_min)
        ranges[bad] = 0.0

        # cap long ranges (Tweak 4)
        return np.minimum(ranges, self.MAX_RANGE), angles

    # [ Step 2 ] extend the disparities

    def extend_disparities(self, ranges: np.ndarray, boxes: np.ndarray,
                           angle_increment: float) -> np.ndarray:
        """
        Extend every disparity over the farther side by the car's half-width plus a margin.

        Disparities are found between neighbouring *valid* beams. Black barriers often give no
        return right at their edge; comparing only adjacent beams then found no disparity
        there at all (3 missing beams at an edge: nothing extended), and the car could aim at
        the end of a barrier. The extension is counted from the near edge across the missing
        beams, which stay invalid (never a target).

        The width is kept from the whole near obstacle, not only from its edge. A box seen
        from the side shows two faces as one surface, and the disparity is at the far end of
        the side face. Measuring the width there left the near corner too little room (0.27 m
        instead of 0.35 m in the simulator), and the planned arc brushed it at full speed.

        Args:
            ranges: cleaned ranges (0.0 = invalid)
            boxes: True for beams the camera confirmed as boxes
            angle_increment: angle between two beams in radians

        Returns:
            extended: ranges after the extension (a copy)
        """
        extended = ranges.copy()
        n = ranges.size

        # disparities: consecutive valid beams (missing returns between them skipped) whose
        # ranges jump by more than the threshold; i is the lower beam index, j the upper
        valid = np.nonzero(ranges > 0.0)[0]
        jumps = np.nonzero(np.abs(np.diff(ranges[valid])) > self.DISPARITY_THRESHOLD)[0]
        jump_set = set(int(k) for k in jumps)
        for k in jumps:
            i, j = int(valid[k]), int(valid[k + 1])
            near = min(ranges[i], ranges[j])

            # a box the camera confirmed gets the wider margin
            margin = self.BOX_MARGIN if (boxes[i] or boxes[j]) else self.MARGIN
            width = self.CAR_HALF_WIDTH + margin

            # the near obstacle's surface: the valid beams from the edge back to its other
            # edge (the next disparity), at most SURFACE_SPAN away
            span = int(self.SURFACE_SPAN / angle_increment)
            if ranges[j] > ranges[i]:
                first = k
                while first > 0 and (first - 1) not in jump_set and i - valid[first - 1] <= span:
                    first -= 1
                surface = valid[first:k + 1]
            else:
                last = k + 1
                while (last < valid.size - 1 and last not in jump_set
                       and valid[last + 1] - j <= span):
                    last += 1
                surface = valid[k + 1:last + 1]

            # beams needed to cover that width at each surface point's range: a point nearer
            # than the edge (the near corner of a box seen from the side) needs more of them
            counts = np.ceil(np.arctan2(width, ranges[surface]) / angle_increment).astype(int)

            # overwrite the farther side from the near edge on, never making a beam longer
            if ranges[j] > ranges[i]:
                lo, hi = i + 1, min(n, int(np.max(surface + counts)) + 1)
            else:
                lo, hi = max(0, int(np.min(surface - counts))), j
            extended[lo:hi] = np.minimum(extended[lo:hi], near)

        return extended

    # [ Step 3 ] pick a target

    def find_target(self, extended: np.ndarray, angles: np.ndarray) -> int | None:
        """
        Select the centre of the deepest run of beams within the field of view.

        Args:
            extended: ranges after extend_disparities
            angles: angle of each beam (rad)

        Returns:
            index of the target beam, or None if there is no gap
        """
        in_fov = np.nonzero(np.abs(angles) <= self.FOV)[0]
        if in_fov.size == 0:
            return None
        sub = extended[in_fov]
        deepest = float(sub.max())
        if deepest < self.MIN_GAP_RANGE:
            return None

        # runs of beams almost as deep as the deepest (Tweak 4: follow the gap's centre)
        deep = (sub >= self.DEEP_FRACTION * deepest).astype(int)
        edges = np.diff(np.concatenate([[0], deep, [0]]))
        starts = np.nonzero(edges == 1)[0]
        ends = np.nonzero(edges == -1)[0] - 1

        # the widest run; between equal runs, the one closest to the last target
        def score(k: int) -> tuple[int, float]:
            centre = in_fov[(starts[k] + ends[k]) // 2]
            return (-(ends[k] - starts[k]), abs(float(angles[centre]) - self.target_angle))

        best = min(range(starts.size), key=score)
        index = int(in_fov[(starts[best] + ends[best]) // 2])
        self.target_angle = float(angles[index])
        return index

    # [ Step 4 ] steering

    def lane_offset(self, ranges: np.ndarray, angles: np.ndarray) -> float:
        """
        How far the middle between the walls beside the car is to its left.

        The gap target alone pulls the car to the middle only weakly, so a small steering
        pull (servo trim) kept it riding along one wall on the straights.

        Args:
            ranges: cleaned ranges (0.0 = invalid)
            angles: angle of each beam (rad)

        Returns:
            offset: metres, left positive; 0.0 unless there is a wall within LANE_MAX on
            both sides (beams 45 to 100 degrees each side) and nothing nearer than the
            range cap in the 0.5 m wide strip straight ahead: near a box or a turn the gap
            target decides alone
        """
        in_path = ((np.abs(angles) < math.pi / 2) & (ranges > 0.0)
                   & (ranges < 0.95 * self.MAX_RANGE)
                   & (np.abs(ranges * np.sin(angles)) < 0.25))
        if in_path.any():
            return 0.0
        sides = []
        for sign in (1.0, -1.0):
            beside = ((angles * sign >= math.radians(45.0))
                      & (angles * sign <= math.radians(100.0))
                      & (ranges > 0.0) & (ranges < self.MAX_RANGE))
            lateral = np.abs(ranges[beside] * np.sin(angles[beside]))
            lateral = lateral[lateral < self.LANE_MAX]
            if lateral.size < 10:
                return 0.0
            sides.append(float(np.percentile(lateral, 20)))
        return 0.5 * (sides[0] - sides[1])

    def calculate_steering_angle(self, angle: float, distance: float,
                                 lane_offset: float = 0.0) -> float:
        """
        Pure-pursuit steering to a point on the target beam, clamped and smoothed.

        The disparity extender guarantees the car can drive straight along the target beam,
        so the car must turn onto that beam before it reaches the obstacle beside it. The
        lookahead therefore grows with the measured speed (LOOKAHEAD_TIME seconds ahead):
        short and decisive when the car is slow near obstacles, longer and smoother on fast
        straights. With a lookahead of about two wheelbases, pure pursuit steers straight
        at the beam.

        Args:
            angle: bearing of the target (rad, positive is left)
            distance: range of the target (m)
            lane_offset: from lane_offset(); the goal point is moved sideways by
                CENTRE_GAIN times this, towards the middle of the lane

        Returns:
            steering_angle: steering angle in radians (positive is left)
        """
        # goal point: ld along the target beam, from the LiDAR
        ld = min(distance, self.LOOKAHEAD_TIME * self.lookahead_speed(), self.LOOKAHEAD)
        ld = max(ld, self.MIN_LOOKAHEAD)

        # pure pursuit: curvature 2 y / L^2 (Coulter 1992), as in Milestone 3, but from the
        # rear axle, which the car turns about; the LiDAR is LIDAR_X ahead of it
        gx = ld * math.cos(angle) + self.LIDAR_X
        gy = ld * math.sin(angle) + self.CENTRE_GAIN * lane_offset
        curvature = 2.0 * gy / (gx * gx + gy * gy)
        steering = math.atan(self.WHEELBASE * curvature) * self.STEERING_GAIN

        # clamp to the car's steering range
        steering = max(-self.MAX_STEERING, min(self.MAX_STEERING, steering))

        # blend with the previous command to prevent sudden changes
        steering = self.SMOOTHING * self.previous_steering + (1.0 - self.SMOOTHING) * steering
        self.previous_steering = steering
        return steering

    def corner_check(self, ranges: np.ndarray, angles: np.ndarray, steering: float) -> float:
        """
        Go straight instead of turning into something right beside the car (Tweak 3).

        Uses the LiDAR beams between ±SIDE_ANGLE and ±SIDE_MAX_ANGLE, beside the car, which
        the camera cannot see. Further back the beams can hit the car's own body, and any
        reading nearer than SIDE_MIN_RANGE is the car itself, so both are left out.

        Args:
            ranges: cleaned ranges (0.0 = invalid)
            angles: angle of each beam (rad)
            steering: steering angle about to be sent (rad)

        Returns:
            steering: unchanged, or 0.0 if the turning side is too close
        """
        if self.side_distance(ranges, angles, steering) < self.SIDE_CLEAR:
            self.previous_steering = 0.0
            return 0.0
        return steering

    def side_distance(self, ranges: np.ndarray, angles: np.ndarray, steering: float) -> float:
        """
        Nearest thing beside the car on the side it is turning to.

        Args:
            ranges: cleaned ranges (0.0 = invalid)
            angles: angle of each beam (rad)
            steering: steering angle about to be sent (rad)

        Returns:
            distance: nearest side reading (m); inf if there is none or the car is not turning
        """
        if steering > 0.0:
            side = (angles >= self.SIDE_ANGLE) & (angles <= self.SIDE_MAX_ANGLE)
        elif steering < 0.0:
            side = (angles <= -self.SIDE_ANGLE) & (angles >= -self.SIDE_MAX_ANGLE)
        else:
            return math.inf
        beside = ranges[side & (ranges >= self.SIDE_MIN_RANGE)]
        return float(beside.min()) if beside.size else math.inf

    # [ Step 5 ] speed

    def path_clearance(self, scan: LaserScan, angles: np.ndarray, steering: float) -> float:
        """
        Free distance along the arc the car is about to drive (the planned path).

        Uses the scan without the MAX_RANGE cap (a far wall is not an obstacle at 5 m) and the
        same corridor as the safety node (half the car plus path_margin). A wider one made the
        planner stop beside boxes it was correctly passing.

        Args:
            scan: LaserScan being processed
            angles: angle of each beam (rad)
            steering: steering angle about to be sent (rad)

        Returns:
            free: distance along the path to the nearest point on it (m), inf if none
        """
        ranges = np.asarray(scan.ranges, dtype=float)
        with np.errstate(invalid='ignore'):
            bad = ~np.isfinite(ranges) | (ranges < scan.range_min) | (ranges > scan.range_max)
        ranges = np.where(bad, np.inf, ranges)
        _, along = path_distances(ranges, angles, steering, self.WHEELBASE, self.CORRIDOR)
        return float(along.min()) if along.size else math.inf

    def box_ahead(self, ranges: np.ndarray, angles: np.ndarray, boxes: np.ndarray) -> bool:
        """
        Check for a camera-confirmed box ahead within BOX_SLOW_RANGE (with a hold).

        Passing starts as soon as a box is ahead and ends only after no box has been ahead
        for PASS_HOLD seconds, so the speed cap does not toggle on and off.

        Args:
            ranges: cleaned ranges (0.0 = invalid)
            angles: angle of each beam (rad)
            boxes: True for beams the camera confirmed as boxes

        Returns:
            bool: True if the car should slow down to pass a box
        """
        near = (boxes & (np.abs(angles) <= self.FOV)
                & (ranges > 0.0) & (ranges < self.BOX_SLOW_RANGE))
        now = self.get_clock().now().nanoseconds * 1e-9
        if near.any():
            self.pass_until = now + self.PASS_HOLD
            i = int(np.argmin(np.where(near, ranges, np.inf)))
            self.box_where = describe(float(ranges[i]), float(angles[i]))
        return now < self.pass_until

    def braking_limit(self, free: float) -> float:
        """
        Highest speed that keeps the TTC to the path PLAN_FACTOR times the TTC needed to stop.

        PLAN_FACTOR is a little more than the safety node's slow_factor, so the planner
        always asks for a speed the safety node will not have to cut.

        Args:
            free: free distance along the planned path (m)

        Returns:
            speed: braking-limited speed (m/s)
        """
        if not math.isfinite(free):
            return math.inf
        return float(slow_speed_cap(free - self.STOP_MARGIN, 0.0, self.PLAN_FACTOR,
                                    self.T_REACT, self.A_BRAKE, self.TTC_MARGIN))

    def cornering_limit(self, steering: float) -> float:
        """
        Highest speed the tyres can hold on the arc of a steering angle.

        Sideways acceleration is v^2 * curvature, with curvature = tan(steering) / wheelbase,
        so v = sqrt(A_LAT / curvature).

        Args:
            steering: steering angle (rad)

        Returns:
            speed: cornering-limited speed (m/s), inf when driving straight
        """
        curvature = abs(math.tan(steering)) / self.WHEELBASE
        return math.sqrt(self.A_LAT / curvature) if curvature > 1e-6 else math.inf

    def calculate_speed(self, free: float, steering: float, passing: bool) -> float:
        """
        Speed request (variable speed): the lowest of the physical limits.

        MAX_SPEED, the braking limit for the free distance on the planned path, the
        cornering limit for the steering angle, and PASS_SPEED while passing a box.
        It never drops below MIN_SPEED unless the braking limit itself is lower.

        Args:
            free: free distance along the planned path (m)
            steering: steering angle (rad)
            passing: True while a confirmed box is near

        Returns:
            speed: requested speed (m/s), before the acceleration ramp
        """
        brake = self.braking_limit(free)
        speed = min(self.MAX_SPEED, brake, self.cornering_limit(steering))
        if passing:
            speed = min(speed, self.PASS_SPEED)
        return max(speed, min(self.MIN_SPEED, brake))

    def ramp(self, speed: float) -> float:
        """
        Let the requested speed rise by at most A_ACCEL per second; drops happen at once.

        Args:
            speed: speed from calculate_speed (m/s)

        Returns:
            speed: speed to request now (m/s)
        """
        now = self.get_clock().now()
        dt = 0.025 if self.speed_time is None else (now - self.speed_time).nanoseconds * 1e-9
        self.speed_time = now
        return min(speed, self.previous_speed + self.A_ACCEL * min(max(dt, 0.0), 0.1))

    def publish_drive(self, steering_angle: float, speed: float) -> None:
        """
        Publish a steering + speed request; the safety node caps the speed.

        Args:
            steering_angle: steering angle in radians (positive is left)
            speed: requested speed (m/s)
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = float(speed)
        self.publisher_.publish(drive_msg)

    def publish_debug(self, msg: LaserScan, source: str, target: tuple[float, float] | None,
                      wanted_steering: float, steering: float, beside: float, free: float,
                      straight: float | None, passing: bool, wanted: float,
                      speed: float) -> None:
        """
        Publish why this steering and speed were chosen, as one line of JSON.

        Nothing is built or sent unless something is subscribed to gap_follow/debug.

        Fields: t (scan stamp, s), scan ('fused' or 'raw'), gap (False if no target was
        found; the rest is then left out), target_deg and target_m (bearing, left positive,
        and range of the target), steer_want and steer (rad, before and after the side
        check), beside (nearest reading beside the turning side, m), free (distance used for
        the braking limit, m), free_straight (straight-path distance, when the side rule
        looked at it, m), v (measured speed, m/s), lim_max / lim_brake / lim_corner /
        lim_pass (each speed limit, m/s; null = no limit), limit (which one set the speed:
        max, brake, corner, pass, min = lifted to min_speed, ramp = still accelerating),
        speed (requested, m/s), passing, box (while passing: where the nearest confirmed box
        ahead was, e.g. '0.9 m ahead, 30 deg left'), boxes (beams labelled as a box).

        Args:
            msg: the scan that was processed
            source: 'fused' or 'raw'
            target: (bearing in rad, range in m) of the target, or None if there is no gap
            wanted_steering: steering before the side check (rad)
            steering: steering that was sent (rad)
            beside: nearest reading beside the turning side (m)
            free: free distance the braking limit used (m)
            straight: free distance straight ahead if the side rule checked it, else None (m)
            passing: True while passing a confirmed box
            wanted: speed before the acceleration ramp (m/s)
            speed: speed that was sent (m/s)
        """
        if self.debug_publisher.get_subscription_count() == 0:
            return
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if target is None:
            self.debug_publisher.publish(String(data=encode(t=t, scan=source, gap=False)))
            return
        limits = {'max': self.MAX_SPEED, 'brake': self.braking_limit(free),
                  'corner': self.cornering_limit(steering),
                  'pass': self.PASS_SPEED if passing else math.inf}
        limit = min(limits, key=limits.get)
        if speed < wanted - 1e-9:
            limit = 'ramp'
        elif wanted > limits[limit] + 1e-9:
            limit = 'min'
        self.debug_publisher.publish(String(data=encode(
            t=t, scan=source, gap=True,
            target_deg=math.degrees(target[0]), target_m=target[1],
            steer_want=wanted_steering, steer=steering, beside=beside,
            free=free, free_straight=straight, v=self.lookahead_speed(),
            lim_max=limits['max'], lim_brake=limits['brake'], lim_corner=limits['corner'],
            lim_pass=limits['pass'], limit=limit, speed=speed, passing=bool(passing),
            box=self.box_where if passing else None,
            boxes=int(np.count_nonzero(self.box_beams(len(msg.ranges)))))))

    # Processes each scan to calculate and publish the drive request.
    def drive_on(self, msg: LaserScan, source: str) -> None:
        """
        Run the disparity extender on one scan and publish the drive request.

        a. preprocess the scan
        b. extend the disparities (wider at camera-confirmed boxes)
        c. find the target and steer to it, unless the turning side is too close
        d. choose the speed and publish to drive_request

        Error Handling:
        publishes nothing if there is no gap or if any error is raised; the safety node's
        watchdog then stops the car.

        Args:
            msg: fused (or raw) LaserScan
            source: 'fused' or 'raw', for gap_follow/debug
        """
        try:
            # a. preprocess the scan
            ranges, angles = self.preprocess_scan(msg)
            if ranges.size < 2:
                return
            boxes = self.box_beams(ranges.size)

            # b. extend the disparities
            extended = self.extend_disparities(ranges, boxes, msg.angle_increment)

            # c. find the target and steer to it
            index = self.find_target(extended, angles)
            if index is None:
                # no gap: nothing is published, and the safety node's watchdog brakes
                self.publish_debug(msg, source, None, 0.0, 0.0, math.inf, math.inf, None,
                                   False, 0.0, 0.0)
                return
            target = (float(angles[index]), float(extended[index]))
            wanted_steering = self.calculate_steering_angle(
                *target, self.lane_offset(ranges, angles))
            beside = self.side_distance(ranges, angles, wanted_steering)
            steering = self.corner_check(ranges, angles, wanted_steering)

            # d. choose the speed and publish
            passing = self.box_ahead(ranges, angles, boxes)
            self.passing = passing
            free = self.path_clearance(msg, angles, steering)
            # the turning side is nearly close enough for corner_check to cancel the turn:
            # the straight path it would switch to must be safe at this speed as well
            straight = None
            if beside < self.SIDE_CLEAR + self.SIDE_WARN:
                straight = self.path_clearance(msg, angles, 0.0)
                free = min(free, straight)
            wanted = self.calculate_speed(free, steering, passing)
            speed = self.ramp(wanted)
            self.previous_speed = speed
            self.publish_drive(steering, speed)
            self.publish_debug(msg, source, target, wanted_steering, steering, beside, free,
                               straight, passing, wanted, speed)

        # error logging (once: the watchdog stops the car if no requests follow)
        except Exception as exc:
            self.get_logger().error(f'gap follow failed: {exc}', once=True)


def main(args=None) -> None:
    """Run the gap follow node."""
    rclpy.init(args=args)                   # initialize ROS 2
    gap_follow_node = GapFollowNode()       # creates GapFollowNode instance
    try:
        rclpy.spin(gap_follow_node)         # spin the node running + callback
    except KeyboardInterrupt:
        pass
    gap_follow_node.destroy_node()          # clears up the node
    if rclpy.ok():
        rclpy.shutdown()                    # shut down ROS2 (skip if Ctrl+C already did)


if __name__ == '__main__':
    main()
