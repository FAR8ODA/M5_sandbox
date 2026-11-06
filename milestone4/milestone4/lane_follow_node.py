"""
Lane-centering for Milestone 4 (camera-based).

Our lane_follow_node subscribes to /camera_scan, which is generated from RGB images published on /camera/color/image_raw.
The image is processed by camera_scan_node using OpenCV and the perception package modules: camera_model, masks, boundary and debug_draw.
The preprocessing pipeline converts image pixels into estimated positions and distances on the ground plane, producing a LaserScan message.

The lane-follow algorithm then:
    a. finds the walls in the scan and measures the lane width
    b. estimates the lane centreline from the walls
    c. selects a target on the centreline at the lookahead distance (or a passing line around an obstacle)
    d. calculates the steering and speed with pure pursuit

Finally, lane_follow_node publishes the steering and speed command as an AckermannDriveStamped message on /drive_request.
This is received by safety_node before sending drive command.
"""

# include needed imports
from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


# Lane Following Algorithm

# [ Step 1 ] find the walls
# a: convert the camera scan ranges into 2D points (x: ahead, y: left)
# b: split the points into chains of neighbours; long chains are walls, short chains may be obstacles
# c: measure the lane width on straights, where both the left and right walls are visible

# [ Step 2 ] find the centreline
# shift each wall point half a lane width along its normal, toward the car: c = p + (w/2) * n
# at a corner the inner wall leaves the view, but the outer wall still gives centreline points

# [ Step 3 ] pick a target
# the centreline point about LOOKAHEAD metres away, continuing the previous choice
# if an obstacle blocks the centreline on a straight, move the target to a free passing line

# [ Step 4 ] pure pursuit (Coulter 1992)
# steering = gain * atan(2 * L * sin(alpha) / ld)
# L: wheelbase, alpha: bearing of the target, ld: distance to the target
# speed is reduced in proportion to the steering


class LaneFollowNode(Node):
    """
    Reactive lane-centering controller (Milestone 4: camera scan instead of LiDAR).

    Reads camera_scan (a LaserScan built from the colour image), finds the lane
    centreline from the walls and steers to a point on it with pure pursuit.
    Publishes steering and a requested speed on drive_request; the safety node
    caps the speed, applies AEB and stops the car if these messages stop.
    """

    def __init__(self) -> None:
        """Initialize parameters, state, subscriber, publisher."""
        super().__init__('lane_follow_node')

        # Launch parameters (set in milestone4_py.py):
        self.declare_parameter('lookahead', 1.0)
        self.declare_parameter('gain', 1.0)
        self.declare_parameter('max_speed', 0.7)


        # Lane Follow Parameters:

        self.LOOKAHEAD = float(self.get_parameter('lookahead').value)  # Distance to the target on the centreline (m)
        self.LANE_WIDTH = 1.0           # Starting guess of the lane width, replaced once measured (m)
        self.MIN_LANE = 0.6             # Narrowest lane width accepted from a measurement (m)
        self.MAX_LANE = 3.0             # Widest lane width accepted from a measurement (m)
        self.CHAIN_GAP = 0.25           # Largest distance between neighbouring points in one chain (m)
        self.OBSTACLE_SIZE = 1.0        # Chains shorter than this may be obstacles, longer chains are walls (m)
        self.NEAR_LIMIT = 0.75          # Nearest floor the camera sees (mount 0.187 m, pitch -5.3°, bottom 30 rows ignored) (m)
        self.MAX_RANGE = 4.0            # Scan value meaning nothing was seen, updated from each scan (m)


        # Obstacle passing parameters:
        # Note: the passing line is only planned on straights
        # AEB handled seperately in safety node.

        self.CAR_RADIUS = 0.3           # Car's half-width plus margin, added to each side of an obstacle (m)
        self.AVOID_RANGE = 3.0          # Obstacles farther than this are ignored (m)
        self.AVOID_HOLD = 1.0           # Keep the passing line this long after the obstacle leaves the view (s)
        self.FIT_HALF_WIDTH = 0.10      # Blocked check: car's half-width (0.15 m) minus 0.05 m for perception error (m)
        self.BLOCKED_FRAMES = 3         # Scans in a row with no gap the car fits through before requesting a stop


        # Steering parameters:

        self.STEERING_GAIN = float(self.get_parameter('gain').value)  # Scales the pure-pursuit steering angle
        self.WHEELBASE = 0.33           # Distance between the front and rear axles (m)
        self.MAX_STEERING = 0.4         # Max steering limit (±22.9°), the car's real steering range
        self.SMOOTHING = 0.3            # Weight of the previous steering command (higher = smoother, slower)


        # Speed parameters:

        self.MAX_SPEED = float(self.get_parameter('max_speed').value)  # Speed on straights (m/s), safety node caps it too
        self.CORNER_SPEED = 0.6         # Fraction of MAX_SPEED at full steering lock
        self.PASS_SPEED = 0.5           # Fraction of MAX_SPEED while passing an obstacle


        # What the node remembers between scans:

        self.previous_steering = 0.0    # previous steering cmd, initialized to straight ahead
        self.target_angle = 0.0         # bearing of the last target, remembers the lane direction (rad)
        self.lane_width = None          # measured lane width (m), None until measured
        self.width_samples = []         # recent lane-width measurements, the median is used
        self.avoid_offset = 0.0         # passing line, sideways from the centreline (m, left positive)
        self.avoid_time = 0.0           # time left to hold the passing line (s)
        self.lane_psi = None            # lane direction relative to the car (rad)
        self.lane_c0 = 0.0              # sideways position of the centreline at the car (m)
        self.lane_blocked = False       # this scan: no gap the car fits through, even without margin
        self.blocked_count = 0          # scans in a row with the lane blocked
        self.stop_requested = False     # latched stop sent to the safety node
        self.last_stamp = None          # arrival time of the previous scan
        self.last_mode = 'lane'         # previous mode, for logging changes
        self.last_width_log = None      # last lane width logged


        # create subscriber and publisher
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            'drive_request',
            10)  # steering + speed request, safety node caps the speed

        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)

        self.get_logger().info(
            f'lane_follow_node started: lookahead {self.LOOKAHEAD} m, '
            f'gain {self.STEERING_GAIN}, max speed {self.MAX_SPEED} m/s')


    # [ Step 1 ] find the walls

    def preprocess_scan(self, scan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert the scan ranges into 2D points, keeping only real hits.

        Args:
            scan: LaserScan message from camera_scan_node

        Returns:
            pts: Nx2 points (x ahead, y left) in meters
            idx: beam index of each point
            blind: True for every beam whose wall is in the camera's blind zone
        """

        # convert scans into numpy array (meters)
        ranges = np.asarray(scan.ranges, dtype=float)
        # calculate the angle corresponding to each scan index (radians)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        # walls closer than NEAR_LIMIT are reported at its edge (blind zone);
        # they still mark the walls beside the car, but say nothing about an obstacle's shape
        blind = ranges <= self.NEAR_LIMIT / np.cos(angles) + 0.03

        # keep only real hits: finite, positive and below MAX_RANGE (nothing seen)
        valid = np.isfinite(ranges) & (ranges < self.MAX_RANGE - 0.05) & (ranges > 0.0)
        idx = np.nonzero(valid)[0]

        # convert polar (range, angle) into Cartesian (x, y) points
        pts = np.column_stack([ranges[idx] * np.cos(angles[idx]), ranges[idx] * np.sin(angles[idx])])

        return pts, idx, blind


    def split_chains(self, pts, idx) -> list[np.ndarray]:
        """
        Split ordered points into chains of neighbours.

        Args:
            pts: Nx2 points in beam order
            idx: beam index of each point

        Returns:
            chains: arrays of positions into pts, one per chain
        """

        if len(pts) == 0:
            return []

        # a new chain starts when neighbouring points are farther apart than CHAIN_GAP,
        jump = np.hypot(*np.diff(pts, axis=0).T) > self.CHAIN_GAP
        # or the range jumps by more than 12% (an obstacle edge),
        r = np.hypot(pts[:, 0], pts[:, 1])
        step = np.abs(np.diff(r)) / np.minimum(r[:-1], r[1:]) > 0.12
        # or more than 3 beams in a row saw nothing
        skip = np.diff(idx) > 3
        cuts = np.nonzero(jump | step | skip)[0] + 1

        # split each chain again at sharp corners
        chains = []
        for c in np.split(np.arange(len(pts)), cuts):
            if len(c) == 0:
                continue
            chains.extend(x for x in np.split(c, self.corner_cuts(pts[c])) if len(x) > 0)

        return chains


    def corner_cuts(self, p, window=0.15, max_turn_deg=50.0) -> list[int]:
        """
        Find positions in a chain where the points make a sharp turn.

        Args:
            p: Mx2 chain points in order
            window: distance before and after each point used to compare directions (m);
                    grows with range beyond 1.5 m, where the points are noisier
            max_turn_deg: turning angle above which the chain is split (degrees)

        Returns:
            cuts: indices where a new chain should start
        """

        # not enough points or length to find a corner
        if len(p) < 5:
            return []
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])  # distance along the chain
        if s[-1] < 2.5 * window:
            return []

        # turning angle at each point: angle between the direction w metres before and w metres after
        turn = np.zeros(len(p))
        rng = np.hypot(p[:, 0], p[:, 1])
        for i in range(len(p)):
            w = window * max(1.0, rng[i] / 1.5)
            if s[i] < w or s[i] > s[-1] - w:
                continue                                    # too close to either end of the chain
            b = p[np.searchsorted(s, s[i] - w)]
            f = p[min(np.searchsorted(s, s[i] + w), len(p) - 1)]
            d1, d2 = p[i] - b, f - p[i]
            c = float(d1 @ d2) / max(float(np.linalg.norm(d1) * np.linalg.norm(d2)), 1e-9)
            turn[i] = math.degrees(math.acos(max(-1.0, min(1.0, c))))

        # cut once per sharp region, at its sharpest point
        cuts, i = [], 0
        while i < len(p):
            if turn[i] > max_turn_deg:
                j = i
                while j + 1 < len(p) and turn[j + 1] > max_turn_deg:
                    j += 1
                cuts.append(int(i + np.argmax(turn[i:j + 1])))
                i = j + 1
            else:
                i += 1

        return cuts


    def chain_length(self, p) -> float:
        """Length along a chain of points (m)."""
        return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


    def sort_chains(self, pts, idx, chains, num_beams) -> tuple:
        """
        Sort chains into walls and short chains (possible obstacles).

        Args:
            pts: Nx2 points
            idx: beam index of each point
            chains: chains from split_chains
            num_beams: number of beams in the scan

        Returns:
            walls: wall chains (Mx2 points each)
            shorts: short chains, as (points, positions into pts)
            right_chain: chain at the right edge of the view, or None
            left_chain: chain at the left edge of the view, or None
        """

        walls, shorts = [], []
        right_chain = left_chain = None
        for c in chains:
            p = pts[c]
            # remember the chains touching the edges of the view (for the lane width)
            if idx[c[0]] <= 2:
                right_chain = p
            if idx[c[-1]] >= num_beams - 3:
                left_chain = p
            # long chains are walls
            if self.chain_length(p) >= self.OBSTACLE_SIZE:
                walls.append(p)
            else:
                shorts.append((p, c))

        # short chains that join end to end into something spanning most of the lane
        # are a wall across the lane (the outer wall at a corner), not obstacles
        lane_w = self.lane_width or self.LANE_WIDTH
        groups = []
        for p, c in shorts:
            if groups and np.hypot(*(groups[-1][-1][0][-1] - p[0])) <= 0.12:
                groups[-1].append((p, c))
            else:
                groups.append([(p, c)])
        shorts = []
        for g in groups:
            if np.hypot(*(g[0][0][0] - g[-1][0][-1])) > 0.8 * lane_w:
                walls.extend(p for p, _ in g)
            else:
                shorts.extend(g)

        return walls, shorts, right_chain, left_chain


    def measure_lane_width(self, right, left) -> float | None:
        """
        Lane width from the right and left walls, if both look like straight parallel walls.

        Args:
            right: points of the chain at the right edge of the view
            left: points of the chain at the left edge of the view

        Returns:
            width: lane width (m), or None if the walls are not a clean straight
        """

        fits = []
        for p in (right, left):
            if len(p) < 8 or np.ptp(p[:, 0]) < 0.4:
                return None
            k, c = np.polyfit(p[:, 0], p[:, 1], 1)          # fit the wall as y = k x + c
            if np.percentile(np.abs(p[:, 1] - (k * p[:, 0] + c)), 90) > 0.08:
                return None                                   # curved: not a straight section
            fits.append((k, c))
        (k1, c1), (k2, c2) = fits

        # the walls must be parallel (within 8°)
        if abs(math.atan(k1) - math.atan(k2)) > math.radians(8.0):
            return None

        # width = gap between the two lines, measured perpendicular to them
        k = (k1 + k2) / 2.0
        return abs(c2 - c1) * math.cos(math.atan(k))


    def update_lane_width(self, right, left) -> float:
        """
        Add a new lane-width measurement (if there is one) and return half the lane width.

        Args:
            right: chain at the right edge of the view, or None
            left: chain at the left edge of the view, or None

        Returns:
            half: half the lane width (m)
        """

        if right is not None and left is not None and right is not left:
            w = self.measure_lane_width(right, left)
            if w is not None and self.MIN_LANE <= w <= self.MAX_LANE:
                # median of the last 30 measurements
                self.width_samples = (self.width_samples + [w])[-30:]
                self.lane_width = float(np.median(self.width_samples))

        return (self.lane_width or self.LANE_WIDTH) / 2.0


    def find_obstacles(self, walls, shorts, half, idx, blind) -> tuple[list, list]:
        """
        Decide which short chains are obstacles; the rest are pieces of wall.

        A short chain is an obstacle only on a straight and inside the lane. In a corner,
        along the lane edge, or when the walls give no centreline, short chains are pieces
        of wall (a ragged or partly hidden wall) and are used as such.

        Args:
            walls: wall chains
            shorts: short chains, as (points, positions into pts)
            half: half the lane width (m)
            idx: beam index of each point
            blind: True for every beam in the camera's blind zone

        Returns:
            walls: wall chains, including the short chains that are pieces of wall
            obstacles: obstacle chains (points outside the blind zone only)
        """

        # straight centreline ahead, if the lane is straight
        fit = self.fit_centreline(self.centreline_points(walls, half)) if walls else None
        straight = fit is not None and fit[2]

        obstacles = []
        for p, c in shorts:
            if straight:
                k, c_, _ = fit
                l_ = (p[:, 1] - (k * p[:, 0] + c_)) * math.cos(math.atan(k))   # distance from the centreline
                if not np.all(np.abs(l_) > half - 0.12):                          # inside the lane
                    seen = p[~blind[idx[c]]]
                    if len(seen) >= 3:
                        obstacles.append(seen)
                    continue                                                      # (a stray point or two: ignored)
            walls.append(p)

        return walls, obstacles


    # [ Step 2 ] find the centreline

    def offset_toward_car(self, p, d) -> np.ndarray:
        """
        Shift each point of a wall chain by d along its normal, toward the car.

        Args:
            p: Mx2 chain points (M >= 3)
            d: offset distance (m)

        Returns:
            shifted: Mx2 shifted points
        """

        # light smoothing (moving average of up to 5 points) before taking tangents
        k = min(5, len(p))
        if k >= 3:
            kern = np.ones(k) / k
            sx = np.convolve(np.pad(p[:, 0], k // 2, mode='edge'), kern, 'valid')
            sy = np.convolve(np.pad(p[:, 1], k // 2, mode='edge'), kern, 'valid')
            s = np.column_stack([sx, sy])[:len(p)]
        else:
            s = p

        # unit tangent along the wall, rotated 90° to get the normal
        t = np.gradient(s, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        n = np.column_stack([-t[:, 1], t[:, 0]])

        # the normal must point toward the car (origin)
        n[np.sum(n * (-s), axis=1) < 0] *= -1.0

        return s + d * n


    def centreline_points(self, walls, half) -> np.ndarray:
        """
        Centreline candidates: wall points shifted half a lane toward the car.

        Shifted points that land near some other wall (off a convex corner, or past
        the middle if the width is overestimated) are dropped.

        Args:
            walls: wall chains
            half: half the lane width (m)

        Returns:
            cand: Kx2 candidate points ahead of the car
        """

        cand = [self.offset_toward_car(p, half) for p in walls if len(p) >= 3]
        if not cand:
            return np.zeros((0, 2))
        cand = np.vstack(cand)

        # drop points closer than 0.6 * half to any wall
        wall_all = np.vstack(walls)
        dmin = np.sqrt(((cand[:, None, :] - wall_all[None, :, :]) ** 2).sum(-1)).min(1)
        cand = cand[dmin >= 0.6 * half]

        # keep points ahead of the car
        return cand[cand[:, 0] > 0.2]


    def fit_centreline(self, cand) -> tuple[float, float, bool] | None:
        """
        Fit a line y = k x + c through the centreline 0.3-3 m ahead.

        Args:
            cand: centreline candidates

        Returns:
            (k, c, straight): the line, and whether the lane there is straight
                              (fits to within 0.1 m and is within 25° of the car's heading)
            None: if there are too few points to fit
        """

        near = cand[(cand[:, 0] > 0.3) & (cand[:, 0] < 3.0)]
        if len(near) < 5 or np.ptp(near[:, 0]) <= 0.5:
            return None
        k, c = np.polyfit(near[:, 0], near[:, 1], 1)
        straight = bool(np.percentile(np.abs(near[:, 1] - (k * near[:, 0] + c)), 90) < 0.1
                        and abs(math.atan(k)) < math.radians(25.0))
        return k, c, straight


    # [ Step 3 ] pick a target

    def find_target(self, cand) -> np.ndarray:
        """
        Select the centreline point near the lookahead distance, continuing the previous choice.

        Args:
            cand: centreline candidates

        Returns:
            target: (x, y) of the target point
        """

        # candidates within 0.2 m of the lookahead distance (or the nearest ones if none)
        d = np.hypot(cand[:, 0], cand[:, 1])
        ang = np.arctan2(cand[:, 1], cand[:, 0])
        band = np.abs(d - self.LOOKAHEAD) < 0.2
        if not band.any():
            band = np.abs(d - self.LOOKAHEAD) <= np.abs(d - self.LOOKAHEAD).min() + 0.1

        # group them by bearing (a gap of more than 10° starts a new group)
        order = np.argsort(ang[band])
        sel_a, sel_p = ang[band][order], cand[band][order]
        groups = np.split(np.arange(len(sel_a)), np.nonzero(np.diff(sel_a) > math.radians(10.0))[0] + 1)

        # take the group closest to the previous target's bearing
        best = min(groups, key=lambda g: abs(float(np.mean(sel_a[g])) - self.target_angle))
        target = sel_p[best].mean(0)

        self.target_angle = math.atan2(target[1], target[0])   # remembered lane direction
        return target


    def lane_frame(self, cand, target) -> tuple[np.ndarray, float, bool]:
        """
        Lane direction and centre offset near the car, from the centreline points.

        The change from the previous frame is limited (the car cannot move sideways or
        turn faster than that between frames), so one noisy frame cannot flip the result.

        Args:
            cand: centreline candidates
            target: lane target, used for the direction if too few points

        Returns:
            n: unit normal of the lane, pointing left
            c0: sideways position of the centreline at the car, along n (m)
            straight: whether the lane ahead is straight
        """

        fit = self.fit_centreline(cand)
        if fit is not None:
            k, c, straight = fit
            psi = math.atan(k)
            c0 = c * math.cos(psi)
        else:
            psi, c0, straight = math.atan2(target[1], target[0]), 0.0, False

        # limit the change to 4° and 5 cm per frame
        if self.lane_psi is not None:
            psi = self.lane_psi + max(-math.radians(4.0), min(math.radians(4.0), psi - self.lane_psi))
            c0 = self.lane_c0 + max(-0.05, min(0.05, c0 - self.lane_c0))
        self.lane_psi, self.lane_c0 = psi, c0

        return np.array([-math.sin(psi), math.cos(psi)]), c0, straight


    def pass_offset(self, obstacles, n, c0, half, prev) -> float | None:
        """
        Passing line (sideways from the lane centre) that clears the obstacles ahead.

        Works in the lane's own frame, so the result does not depend on the car's heading.
        Each obstacle in the lane within AVOID_RANGE blocks the band it covers, widened by
        CAR_RADIUS on each side.

        Args:
            obstacles: obstacle chains
            n: lane normal (left positive)
            c0: sideways position of the centreline at the car
            half: half the lane width (m)
            prev: passing line chosen for the current obstacle (0 if none)

        Returns:
            offset: passing line (m, left positive)
            None: if no obstacle is in the way of the centreline
        """

        t = np.array([n[1], -n[0]])                  # along the lane
        bands = []
        for p in obstacles:
            s_, l_ = p @ t, p @ n - c0               # distance along and across the lane
            if s_.min() > self.AVOID_RANGE or s_.max() < 0.0 or l_.min() > half or l_.max() < -half:
                continue                              # too far, behind the car or outside the lane
            if np.all(np.abs(l_) > half - 0.12):
                continue                              # a piece of wall cut short by an obstacle's shadow
            bands.append((float(l_.min()) - self.CAR_RADIUS, float(l_.max()) + self.CAR_RADIUS))

        # nothing in the lane, or nothing on the centreline and not already passing
        if not bands:
            return None
        if not any(lo < 0.0 < hi for lo, hi in bands) and prev == 0.0:
            return None

        return self.free_nearest(bands, half, prev)


    def free_nearest(self, bands, half, prev) -> float:
        """
        Free sideways position nearest the centre, keeping the side already chosen if it is still free.

        Args:
            bands: blocked (low, high) sideways intervals (m)
            half: half the lane width (m)
            prev: passing line chosen for the current obstacle (0 if none)

        Returns:
            offset: passing line (m, left positive)
        """

        # candidate positions across the lane, staying 0.15 m (the car's half-width) off the walls
        edge = max(half - 0.15, 0.0)
        ls_ = np.linspace(-edge, edge, 41)
        free = np.array([all(not (lo < x < hi) for lo, hi in bands) for x in ls_])

        # no clear line: pass where the overlap is smallest,
        # on the side already chosen if there is one (the AEB backs this up)
        if not free.any():
            depth = np.array([max([min(x - lo, hi - x) for lo, hi in bands if lo < x < hi] or [0.0])
                              for x in ls_])

            # blocked: the overlap is deeper than the margin everywhere, so even the car's
            # real width (FIT_HALF_WIDTH, without margin) cannot fit; scan_callback requests a stop
            self.lane_blocked = bool(depth.min() > self.CAR_RADIUS - self.FIT_HALF_WIDTH)

            if prev != 0.0:
                depth = np.where(np.sign(ls_) == np.sign(prev), depth, np.inf)
            return float(ls_[int(np.argmin(depth))])

        # free positions, on the side already chosen if possible; take the one nearest the centre
        cand = ls_[free]
        same = cand[np.sign(cand) == np.sign(prev)] if prev != 0.0 else cand
        pool = same if len(same) else cand
        return float(pool[np.argmin(np.abs(pool))])


    def passing_line(self, obstacles, n, c0, half, straight, dt) -> float | None:
        """
        Update the passing line and keep it for AVOID_HOLD after the obstacle leaves the view
        (it is then beside the car, where the camera cannot see it).

        Obstacles are planned around on straights only: in a corner the lane frame is not
        a line, and the short chains seen there are the island's rounded end, which the
        centreline already keeps clear of.

        Args:
            obstacles: obstacle chains
            n: lane normal (left positive)
            c0: sideways position of the centreline at the car
            half: half the lane width (m)
            straight: whether the lane ahead is straight
            dt: time since the previous scan (s)

        Returns:
            offset: passing line (m, left positive), or None if not passing
        """

        off = None
        if straight or self.avoid_time > 0.0:
            off = self.pass_offset(obstacles, n, c0, half,
                                   self.avoid_offset if self.avoid_time > 0.0 else 0.0)
        if off is not None:
            self.avoid_offset, self.avoid_time = off, self.AVOID_HOLD
        elif self.avoid_time > 0.0:
            self.avoid_time -= dt
            off = self.avoid_offset

        return off


    # [ Step 4 ] pure pursuit

    def calculate_steering_angle(self, target) -> float:
        """
        Converts the target point into a clamped + smoothed steering command (pure pursuit).

        Args:
            target: (x, y) of the target point

        Returns:
            steering_angle: steering angle in radians (positive is left)
        """

        # pure pursuit: curvature 2 sin(alpha) / ld, scaled by the steering gain
        ld = max(float(np.linalg.norm(target)), 0.3)
        alpha = math.atan2(target[1], target[0])
        steering_angle = math.atan(2.0 * self.WHEELBASE * math.sin(alpha) / ld) * self.STEERING_GAIN

        # clamp the steering angle
        # MAX_STEERING = 0.4 rad (approximately ±22.9°)
        steering_angle = max(-self.MAX_STEERING, min(self.MAX_STEERING, steering_angle))

        # blend with the previous steering command
        # this prevents sudden changes in steering
        steering_angle = self.SMOOTHING * self.previous_steering + (1.0 - self.SMOOTHING) * steering_angle
        self.previous_steering = steering_angle

        return steering_angle


    def calculate_speed(self, steering_angle, passing) -> float:
        """
        Speed request: MAX_SPEED on straights, down to CORNER_SPEED at full lock,
        and at most PASS_SPEED while passing an obstacle.

        Args:
            steering_angle: steering angle in radians
            passing: True while following a passing line

        Returns:
            speed: requested speed (m/s)
        """

        frac = 1.0 - (1.0 - self.CORNER_SPEED) * abs(steering_angle) / self.MAX_STEERING
        if passing:
            frac = min(frac, self.PASS_SPEED)

        return self.MAX_SPEED * frac


    def publish_drive(self, steering_angle, speed, mode) -> None:
        """
        Publish a steering + speed request; the safety node caps the speed.

        Args:
            steering_angle: steering angle in radians (positive is left)
            speed: requested speed (m/s)
            mode: 'lane', 'pass' while passing an obstacle, 'hold' when nothing usable was seen
        """

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = float(speed)
        self.publisher_.publish(drive_msg)

        # log the measured lane width when it changes by more than 5 cm
        w = self.lane_width
        if w is not None and (self.last_width_log is None or abs(w - self.last_width_log) > 0.05):
            self.get_logger().info(f'measured lane width {w:.2f} m')
            self.last_width_log = w

        # log changes between following the centreline and passing an obstacle
        if mode != self.last_mode and 'hold' not in (mode, self.last_mode):
            self.get_logger().info('passing an obstacle' if mode == 'pass' else 'back on the centreline')
        self.last_mode = mode
        if mode == 'hold':
            self.get_logger().warn('no wall seen; easing off the last steering command',
                                   throttle_duration_sec=1.0)


    def publish_stop(self) -> None:
        """Send the latched-stop request (negative speed) to the safety node."""

        stop = AckermannDriveStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.drive.speed = -1.0          # negative speed = latched stop flag
        stop.drive.steering_angle = 0.0
        self.publisher_.publish(stop)


    # Processes incoming camera scan data to calculate and publish lane-following commands.
    def scan_callback(self, msg) -> None:
        """
        a. finds the walls and measures the lane width
        b. finds the centreline
        c. picks the target (or a passing line around an obstacle)
        d. publishes the steering angle and speed to drive_request

        Error Handling:
        if an obstacle leaves no gap the car fits through for BLOCKED_FRAMES scans in a row,
        requests a latched stop (negative speed) instead of trying to squeeze past;
        if no centreline is found, eases off the last steering command at CORNER_SPEED;
        if any error is raised, publishes nothing and the safety node's watchdog stops the car.

        Args:
            msg: LaserScan message from camera_scan_node; ranges at range_max mean nothing was seen.
        Returns:
            None
        """

        try:
            # lane was blocked: keep repeating the stop request, in case one message is lost
            if self.stop_requested:
                self.publish_stop()
                return

            # time since the previous scan, limited to 0.01-0.5 s
            now = self.get_clock().now()
            dt = 1.0 / 15.0 if self.last_stamp is None else (now - self.last_stamp).nanoseconds * 1e-9
            dt = min(max(dt, 0.01), 0.5)
            self.last_stamp = now
            self.MAX_RANGE = float(msg.range_max)

            # a. find the walls and measure the lane width
            pts, idx, blind = self.preprocess_scan(msg)
            chains = self.split_chains(pts, idx)
            walls, shorts, right_chain, left_chain = self.sort_chains(pts, idx, chains, len(msg.ranges))
            half = self.update_lane_width(right_chain, left_chain)
            walls, obstacles = self.find_obstacles(walls, shorts, half, idx, blind)

            # b. find the centreline
            cand = self.centreline_points(walls, half)
            if not len(cand):
                # nothing usable: ease off the last steering command
                self.previous_steering *= 0.9
                self.publish_drive(self.previous_steering, self.MAX_SPEED * self.CORNER_SPEED, 'hold')
                return

            # c. pick the target, or a point on the passing line around an obstacle
            target = self.find_target(cand)
            n, c0, straight = self.lane_frame(cand, target)
            self.lane_blocked = False
            off = self.passing_line(obstacles, n, c0, half, straight, dt)

            # no gap the car fits through for BLOCKED_FRAMES scans in a row:
            # stop instead of trying to "find somewhere to go"
            self.blocked_count = self.blocked_count + 1 if self.lane_blocked else 0
            if self.blocked_count >= self.BLOCKED_FRAMES:
                self.get_logger().warn('lane blocked: no gap the car fits through, requesting stop')
                self.stop_requested = True
                self.publish_stop()
                return
            if off:
                # a point LOOKAHEAD metres along the lane, off from the centreline
                target = self.LOOKAHEAD * np.array([n[1], -n[0]]) + (c0 + off) * n

            # d. calculate and publish the steering and speed
            steering_angle = self.calculate_steering_angle(target)
            speed = self.calculate_speed(steering_angle, bool(off))
            self.publish_drive(steering_angle, speed, 'pass' if off else 'lane')

        # error logging
        except Exception as exc:
            self.get_logger().error(f'lane follow failed: {exc}', throttle_duration_sec=1.0)


def main(args=None) -> None:
    rclpy.init(args=args)                   # initialize ROS 2
    lane_follow_node = LaneFollowNode()     # creates LaneFollowNode instance
    try:
        rclpy.spin(lane_follow_node)        # spin the node running + callback
    except KeyboardInterrupt:
        pass
    lane_follow_node.destroy_node()         # clears up the node
    if rclpy.ok():
        rclpy.shutdown()                    # shut down ROS2 (skip if Ctrl+C already did)


if __name__ == '__main__':
    main()