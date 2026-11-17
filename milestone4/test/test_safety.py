"""Tests for the path-aware AEB safety node."""

import json
import math
import unittest

import numpy as np
import ros_stubs

from milestone4.braking import car_motion, required_ttc, slow_speed_cap  # noqa: I100
from milestone4.safety_node import MotionTracker, SafetyNode  # noqa: I100


def corridor_world(half_width=0.8, wall_x=None):
    """
    Return dense wall points of a straight corridor (and a wall across it at wall_x).

    Args:
        half_width (float): Distance from the centreline to each wall (m).
        wall_x (float | None): Position of a wall across the corridor, or None.

    Returns:
        np.ndarray: Nx2 wall points.
    """
    walls = [ros_stubs.wall_points(-2.0, half_width, 12.0, half_width),
             ros_stubs.wall_points(-2.0, -half_width, 12.0, -half_width)]
    if wall_x is not None:
        walls.append(ros_stubs.wall_points(wall_x, -half_width, wall_x, half_width))
    return np.vstack(walls)


def drive(world_at, steering, speed, steps, step=0.025, on_scan=None):
    """
    Drive a simulated car at a constant steering and speed through a world.

    world_at(t) returns the world points at time t. on_scan(t, ranges, x, y, yaw) is
    called with each scan. Returns the final pose.
    """
    yaw_rate = speed * math.tan(steering) / 0.33
    x = y = yaw = 0.0
    # scan from the current pose, then move along the arc for one step
    for i in range(steps):
        t = i * step
        ranges = ros_stubs.world_scan(world_at(t), x, y, yaw)
        if on_scan is not None:
            on_scan(t, ranges, x, y, yaw)
        ahead, left, turn = car_motion(speed, yaw_rate, step)
        x += ahead * math.cos(yaw) - left * math.sin(yaw)
        y += ahead * math.sin(yaw) + left * math.cos(yaw)
        yaw += turn
    return x, y, yaw


class MotionTrackerTest(unittest.TestCase):
    """Obstacle speeds with the car's own motion removed."""

    def track(self, world_at, steering, speed, box=None):
        """
        Drive through a world and return the approach speeds measured on the last scan.

        Args:
            world_at: Function of time returning the world points.
            steering (float): Constant steering angle (rad).
            speed (float): Constant speed (m/s).
            box: Optional function (t, px, py, x, y, yaw) selecting the points to measure.

        Returns:
            np.ndarray: Approach speeds of the selected points (m/s).
        """
        tracker = MotionTracker()
        a = ros_stubs.lidar_angles()
        result = {}

        def on_scan(t, ranges, x, y, yaw):
            """Add one scan to the tracker and measure the selected points."""
            finite = np.isfinite(ranges)
            px = np.where(finite, ranges, 0.0) * np.cos(a)
            py = np.where(finite, ranges, 0.0) * np.sin(a)
            tracker.add(t, px[finite], py[finite], (x, y, yaw))
            pick = finite & (box(t, px, py, x, y, yaw) if box else np.ones(a.size, bool))
            result['speeds'] = tracker.approach_speeds(px[pick], py[pick])

        drive(world_at, steering, speed, 16, on_scan=on_scan)
        return result['speeds']

    def test_static_walls_stay_still_while_turning(self):
        # comparing beam indices made these walls look like they approached at ~1.7 m/s
        """Check static walls get no approach speed while turning."""
        for steering in (0.05, 0.2, 0.4):
            speeds = self.track(lambda t: corridor_world(), steering, 1.5)
            self.assertLess(float(np.percentile(speeds, 99)), 0.5)

    def test_approaching_box_is_measured_while_turning(self):
        """Check a box approaching at 2 m/s is measured, turning or not."""
        ys = np.arange(-0.2, 0.2, 0.005)

        def world(t):
            """Return the corridor plus the box face at time t."""
            face = np.stack([np.full_like(ys, 4.0 - 2.0 * t), ys], axis=1)   # 2 m/s towards us
            return np.vstack([corridor_world(), face])

        def on_box(t, px, py, x, y, yaw):
            # scan points within 5 cm of the box face, in the car frame
            """Return which scan points lie within 5 cm of the box face."""
            d = np.stack([np.full_like(ys, 4.0 - 2.0 * t), ys], axis=1) - [x, y]
            c, s = math.cos(-yaw), math.sin(-yaw)
            fx, fy = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
            near = np.hypot(px[:, None] - fx[None, :], py[:, None] - fy[None, :]).min(axis=1)
            return near < 0.05

        for steering in (0.0, 0.2):
            speeds = self.track(world, steering, 1.0, box=on_box)
            self.assertGreater(speeds.size, 10)
            self.assertAlmostEqual(float(np.median(speeds)), 2.0, delta=0.2)

    def test_new_points_have_no_speed(self):
        """Check a point that just appeared gets no speed."""
        tracker = MotionTracker()
        tracker.add(0.0, np.array([3.0]), np.array([0.0]), (0.0, 0.0, 0.0))
        tracker.add(0.1, np.array([3.0, 0.8]), np.array([0.0, 0.0]), (0.0, 0.0, 0.0))  # new
        speeds = tracker.approach_speeds(np.array([3.0, 0.8]), np.array([0.0, 0.0]))
        self.assertEqual(speeds.tolist(), [0.0, 0.0])


class SafetyNodeTest(unittest.TestCase):
    """AEB, the soft cap, latching, the watchdog and the scan fallback."""

    def setUp(self):
        """Reset the clock and overrides and build a safety node."""
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()
        self.node = SafetyNode()

    def commands(self):
        """Return every command published on /drive."""
        return self.node.publishers_by_topic['drive'].messages

    def step(self, t, ranges, speed, request=None, source='fused'):
        """
        Advance the clock and feed odometry, one scan and optionally a drive request.

        Args:
            t (float): Time (s).
            ranges: Scan ranges (m).
            speed (float): Odometry speed (m/s).
            request: (steering, speed) of a drive request, or None.
            source (str): 'fused' or 'raw' scan callback.
        """
        ros_stubs.CLOCK.seconds = t
        self.node.odom_callback(ros_stubs.make_odom(speed))
        scan = ros_stubs.make_scan(ranges, t=t)
        if source == 'fused':
            self.node.fused_scan_callback(scan)
        else:
            self.node.raw_scan_callback(scan)
        if request is not None:
            self.node.drive_request_callback(ros_stubs.make_request(*request))

    def test_box_in_path_triggers_aeb_at_speed(self):
        """Check a box 0.8 m ahead at 1.5 m/s triggers AEB."""
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 0.8, -0.2, 0.2)
        self.step(0.0, box, speed=1.5)
        self.assertEqual(self.node.safety_mode, 'aeb')

    def test_same_box_is_fine_at_low_speed(self):
        """Check the same box at 0.3 m/s passes the request through."""
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 0.8, -0.2, 0.2)
        self.step(0.0, box, speed=0.3, request=(0.0, 0.3))
        self.assertNotEqual(self.node.safety_mode, 'aeb')
        self.assertAlmostEqual(self.commands()[-1].drive.speed, 0.3)

    def test_soft_cap_slows_before_aeb(self):
        """Check the soft stage caps the speed at slow_speed_cap()."""
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 1.2, -0.2, 0.2)
        self.step(0.0, box, speed=1.0, request=(0.0, 1.0))
        speed = self.commands()[-1].drive.speed
        self.assertEqual(self.node.safety_mode, 'slow')
        self.assertGreater(speed, 0.5)
        self.assertLess(speed, 1.0)
        expected = float(slow_speed_cap(1.2 - 0.25, 0.0, 2.0, 0.15, 2.0, 0.1))
        self.assertAlmostEqual(speed, expected, places=2)

    def test_corner_walls_do_not_trigger_aeb(self):
        # the Milestone 3 false AEB: driving fast down a narrow section with walls beside it
        """Check walls beside a narrow straight do not trigger AEB."""
        walls = ros_stubs.corridor_ranges(0.5)
        for i in range(10):
            self.step(i * 0.025, walls, speed=2.0, request=(0.0, 1.0))
        self.assertEqual(self.node.safety_mode, 'normal')
        self.assertAlmostEqual(self.commands()[-1].drive.speed, 1.0)

    def test_turning_at_speed_does_not_trigger_aeb(self):
        # static corridor, gentle left turn at 1.5 m/s: the per-beam range rate of an earlier
        # version saw the walls approaching here and braked
        """Check a gentle turn in a static corridor does not trigger AEB."""
        steering, speed = 0.1, 1.5
        yaw_rate = speed * math.tan(steering) / 0.33

        def on_scan(t, ranges, x, y, yaw):
            """Feed odometry, the scan and a drive request at time t."""
            ros_stubs.CLOCK.seconds = t
            self.node.odom_callback(ros_stubs.make_odom(speed, yaw_rate))
            self.node.fused_scan_callback(ros_stubs.make_scan(ranges, t=t))
            self.node.drive_request_callback(ros_stubs.make_request(steering, speed))

        drive(lambda t: corridor_world(0.8), steering, speed, 24, on_scan=on_scan)
        self.assertNotIn(self.node.safety_mode, ('aeb', 'stopped'))
        self.assertGreater(self.commands()[-1].drive.speed, 0.5)   # slowed for the wall, no AEB

    def test_wall_ahead_while_driving_triggers_aeb(self):
        """Check a wall across the corridor triggers AEB."""
        def on_scan(t, ranges, x, y, yaw):
            """Feed odometry and the scan at time t."""
            ros_stubs.CLOCK.seconds = t
            self.node.odom_callback(ros_stubs.make_odom(1.5))
            self.node.fused_scan_callback(ros_stubs.make_scan(ranges, t=t))

        drive(lambda t: corridor_world(0.8, wall_x=2.0), 0.0, 1.5, 40, on_scan=on_scan)
        self.assertEqual(self.node.safety_mode, 'aeb')

    def trigger_distance(self, speed):
        """
        Return the box distance at which AEB engages for a car at a speed.

        Args:
            speed (float): Car speed (m/s).

        Returns:
            float | None: First distance (stepping in by 5 mm) that triggers AEB, or None.
        """
        for i in range(440):
            x = 2.5 - 0.005 * i                                # 2.5 m down to 0.3 m
            node = SafetyNode()
            ros_stubs.CLOCK.seconds = 0.0
            node.odom_callback(ros_stubs.make_odom(speed))
            node.fused_scan_callback(ros_stubs.make_scan(
                ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), x, -0.2, 0.2)))
            if node.safety_mode == 'aeb':
                return x
        return None

    def test_variable_ttc_brakes_earlier_at_speed(self):
        # trigger distance = stop_margin + v x required TTC(v), which grows faster than v
        """Check the trigger distance follows the variable TTC."""
        at_1, at_2 = self.trigger_distance(1.0), self.trigger_distance(2.0)
        self.assertAlmostEqual(at_1, 0.25 + 1.0 * float(required_ttc(1.0, 0, 0.15, 2.0, 0.1)),
                               delta=0.01)
        self.assertAlmostEqual(at_2, 0.25 + 2.0 * float(required_ttc(2.0, 0, 0.15, 2.0, 0.1)),
                               delta=0.01)
        self.assertGreater(at_2 - 0.25, 2.0 * (at_1 - 0.25))   # more than twice as far

    def test_crawling_into_the_margin_triggers_aeb(self):
        # at 0.2 m/s the TTC rule alone would allow a few cm; the gap to the margin does not
        """Check crawling into the stop margin triggers AEB."""
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 0.22, -0.2, 0.2)
        self.step(0.0, box, speed=0.2)
        self.assertEqual(self.node.safety_mode, 'aeb')

    def test_state_shows_the_variable_threshold(self):
        """Check /safety/state shows the AEB and slow thresholds."""
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 2.0, -0.2, 0.2)
        self.step(0.0, box, speed=1.0, request=(0.0, 1.0))
        self.node.watchdog_callback()
        state = self.node.publishers_by_topic['safety/state'].messages[-1].data
        self.assertIn('aeb<0.50s', state)                      # required TTC at 1 m/s
        self.assertIn('slow<1.00s', state)                     # slow_factor x that

    def test_error_in_a_callback_brakes(self):
        """Check an error in a callback brakes and is logged."""
        self.step(0.0, ros_stubs.corridor_ranges(1.0), speed=1.0, request=(0.0, 1.0))
        broken = ros_stubs.make_scan(ros_stubs.corridor_ranges(1.0), t=0.025)
        broken.angle_increment = None                          # makes the callback raise
        self.node.fused_scan_callback(broken)
        self.assertEqual(self.commands()[-1].drive.speed, 0.0)
        self.assertTrue(any(level == 'error' for level, _ in self.node.get_logger().records))

    def test_only_the_aeb_is_logged(self):
        # stale -> resume -> slow -> aeb: only the latched stop is logged, once
        """Check stale, slow and resume are not logged; AEB is, once."""
        clear = ros_stubs.corridor_ranges(1.0)
        self.step(0.0, clear, speed=0.0)
        self.node.watchdog_callback()
        self.assertEqual(self.node.safety_mode, 'stale')
        self.step(0.025, clear, speed=0.0, request=(0.0, 1.0))
        self.assertEqual(self.node.safety_mode, 'normal')
        box = ros_stubs.add_box(clear, 1.2, -0.2, 0.2)
        self.step(0.05, box, speed=0.5, request=(0.0, 1.0))
        self.assertEqual(self.node.safety_mode, 'slow')
        self.step(0.075, ros_stubs.add_box(clear, 0.6, -0.2, 0.2), speed=1.5)
        self.assertEqual(self.node.safety_mode, 'aeb')
        self.step(0.1, ros_stubs.add_box(clear, 0.5, -0.2, 0.2), speed=1.5, request=(0.0, 1.0))
        records = self.node.get_logger().records
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], 'error')
        self.assertTrue(records[0][1].startswith('AEB engaged: obstacle on the path (TTC '))

    def test_approaching_obstacle_brakes_while_stopped(self):
        # the car is still; an obstacle comes towards it at 2 m/s (track_obstacles:=true)
        # Milestone 3 TTC used only our own speed, so it would never brake here
        """Check an obstacle coming at a stopped car triggers AEB."""
        ros_stubs.PARAM_OVERRIDES['safety_node'] = {'track_obstacles': True}
        self.node = SafetyNode()
        ros_stubs.PARAM_OVERRIDES.clear()
        walls = ros_stubs.corridor_ranges(1.0)
        engaged_at = None
        for i in range(40):
            t = i * 0.025
            x = 2.5 - 2.0 * t
            self.step(t, ros_stubs.add_box(walls, x, -0.2, 0.2), speed=0.0)
            if self.node.safety_mode == 'aeb':
                engaged_at = x
                break
        self.assertIsNotNone(engaged_at)
        self.assertGreater(engaged_at, 0.5)

    def test_static_obstacle_does_not_brake_a_stopped_car(self):
        """Check a static box does not trigger AEB for a stopped car."""
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 0.6, -0.2, 0.2)
        for i in range(20):
            self.step(i * 0.025, box, speed=0.0)
        self.assertNotEqual(self.node.safety_mode, 'aeb')

    def test_aeb_releases_once_the_path_clears(self):
        """Check AEB is not latched: the car drives on once the path clears."""
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 0.8, -0.2, 0.2)
        self.step(0.0, box, speed=1.5)
        self.assertEqual(self.node.safety_mode, 'aeb')
        self.assertEqual(self.commands()[-1].drive.speed, 0.0)
        self.step(0.025, ros_stubs.corridor_ranges(1.0), speed=0.0, request=(0.0, 1.5))
        self.assertEqual(self.node.safety_mode, 'normal')      # not latched: drives on
        self.assertGreater(self.commands()[-1].drive.speed, 0.0)

    def test_negative_speed_request_latches_a_stop(self):
        """Check a negative speed request latches a logged stop."""
        clear = ros_stubs.corridor_ranges(1.0)
        self.step(0.0, clear, speed=1.0, request=(0.0, -1.0))
        self.step(0.025, clear, speed=0.0, request=(0.0, 1.5))
        self.assertEqual(self.node.safety_mode, 'stopped')
        self.assertEqual(self.commands()[-1].drive.speed, 0.0)
        self.assertEqual(self.node.get_logger().records,
                         [('error', 'AEB engaged: stop requested (laps done)')])

    def test_watchdog_brakes_when_requests_stop(self):
        """Check the watchdog brakes on stale requests and resumes."""
        clear = ros_stubs.corridor_ranges(1.0)
        self.step(0.0, clear, speed=1.0, request=(0.0, 1.0))
        self.assertGreater(self.commands()[-1].drive.speed, 0.0)

        self.step(0.5, clear, speed=1.0)                     # scans keep coming, requests do not
        self.node.watchdog_callback()
        self.assertEqual(self.node.safety_mode, 'stale')
        self.assertEqual(self.commands()[-1].drive.speed, 0.0)

        self.step(0.525, clear, speed=0.5, request=(0.0, 1.0))   # requests are back: resume
        self.assertEqual(self.node.safety_mode, 'normal')
        self.assertGreater(self.commands()[-1].drive.speed, 0.0)

    def test_falls_back_to_the_raw_scan(self):
        """Check the raw scan is used only while the fused scan is stale."""
        clear = ros_stubs.corridor_ranges(1.0)
        self.step(0.0, clear, speed=0.0, source='raw')
        self.assertIs(self.node.active_source(), self.node.sources['raw'])
        self.step(0.025, clear, speed=0.0, source='fused')
        self.assertIs(self.node.active_source(), self.node.sources['fused'])

    def test_blocked_path_waits_then_drives_on(self):
        """Check the car waits at a blocked path and goes once it opens."""
        blocked = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 0.2, -0.3, 0.3)
        for i in range(30):                                    # held 1.5 s in front of it
            self.step(i * 0.05, blocked, speed=0.0, request=(0.0, 1.0))
        self.assertEqual(self.commands()[-1].drive.speed, 0.0)
        self.assertFalse(self.node.is_latched())
        self.step(1.5, ros_stubs.corridor_ranges(1.0), speed=0.0, request=(0.0, 1.0))
        self.assertGreater(self.commands()[-1].drive.speed, 0.0)   # path open: go again

    def test_state_topic(self):
        """Check /safety/state starts with the mode and names the scan."""
        self.step(0.0, ros_stubs.corridor_ranges(1.0), speed=0.0, request=(0.0, 1.0))
        self.node.watchdog_callback()
        state = self.node.publishers_by_topic['safety/state'].messages[-1].data
        self.assertTrue(state.startswith('normal'))
        self.assertIn('scan=fused', state)


class HelpersTest(unittest.TestCase):
    """Sanity checks of the test helpers themselves."""

    def test_corridor_side_beam(self):
        """Check corridor_ranges() puts the side wall at half_width."""
        a = ros_stubs.lidar_angles()
        r = ros_stubs.corridor_ranges(0.5)
        left = int(np.argmin(np.abs(a - math.pi / 2)))
        self.assertAlmostEqual(float(r[left]), 0.5, places=3)


if __name__ == '__main__':
    unittest.main()


class ExitTest(unittest.TestCase):
    """Ctrl+C (or any other exit) stops the car before the node ends."""

    def run_main(self, raise_in_spin):
        """
        Run safety_node.main() with spin raising an exception.

        Args:
            raise_in_spin (BaseException): Exception raised from the fake spin.

        Returns:
            SafetyNode: The node main() created.
        """
        from milestone4 import safety_node
        made = []

        class Recording(safety_node.SafetyNode):
            """SafetyNode that records each instance it creates."""

            def __init__(self):
                """Build the node and remember it."""
                super().__init__()
                made.append(self)

        def spin(node):
            """Publish a moving command, then raise the given exception."""
            node.publish_drive_command(1.0, 0.2)               # the car is moving
            raise raise_in_spin

        # swap in the recording node and fake spin, restoring them afterwards
        original = safety_node.SafetyNode, safety_node.rclpy.spin
        safety_node.SafetyNode, safety_node.rclpy.spin = Recording, spin
        try:
            safety_node.main()
        finally:
            safety_node.SafetyNode, safety_node.rclpy.spin = original
        return made[0]

    def test_ctrl_c_stops_the_car_then_ends(self):
        """Check Ctrl+C sends zero-speed commands and logs nothing."""
        node = self.run_main(KeyboardInterrupt())              # main() returns normally
        drive = node.publishers_by_topic['drive'].messages
        self.assertEqual(drive[0].drive.speed, 1.0)
        self.assertGreaterEqual(len(drive), 6)
        self.assertTrue(all(m.drive.speed == 0.0 for m in drive[1:]))
        self.assertEqual(node.get_logger().records, [])        # nothing logged on the way out

    def test_any_other_exit_also_stops_the_car(self):
        """Check any other exception still sends a zero-speed command."""
        from milestone4 import safety_node
        made = []
        original_init = safety_node.SafetyNode.__init__

        def init(node):
            """Run the original __init__ and remember the node."""
            original_init(node)
            made.append(node)

        safety_node.SafetyNode.__init__ = init
        try:
            with self.assertRaises(RuntimeError):
                self.run_main(RuntimeError('boom'))
        finally:
            safety_node.SafetyNode.__init__ = original_init
        self.assertEqual(made[-1].publishers_by_topic['drive'].messages[-1].drive.speed, 0.0)


class ObstacleSpeedRobustnessTest(unittest.TestCase):
    """Short-lived obstacle-speed readings must not set off AEB (the car-test failure)."""

    def setUp(self):
        """Reset the clock and overrides."""
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()

    def tearDown(self):
        """Clear the parameter overrides."""
        ros_stubs.PARAM_OVERRIDES.clear()

    def node(self, **params):
        """Return a safety node with these parameters (tracking on by default)."""
        params.setdefault('track_obstacles', True)            # these tests are about tracking
        ros_stubs.PARAM_OVERRIDES['safety_node'] = params
        node = SafetyNode()
        ros_stubs.PARAM_OVERRIDES.clear()
        return node

    def run_glitch(self, node):
        """
        Drive at 0.86 m/s towards a static box; for two scans it seems 0.4 m closer.

        Args:
            node (SafetyNode): Node under test.

        Returns:
            list: Safety mode after each scan.
        """
        ys = np.arange(-0.25, 0.25, 0.005)
        modes = []

        def world(t):
            """Return the corridor plus the box, 0.4 m closer during the glitch."""
            glitch = 0.90 <= t < 0.95                          # two scans, box ~1.8 m away
            x = 2.6 - (0.4 if glitch else 0.0)
            return np.vstack([corridor_world(0.8), np.stack([np.full_like(ys, x), ys], 1)])

        def on_scan(t, ranges, x, y, yaw):
            """Feed odometry, the scan and a drive request; record the mode."""
            ros_stubs.CLOCK.seconds = t
            node.odom_callback(ros_stubs.make_odom(0.86))
            node.fused_scan_callback(ros_stubs.make_scan(ranges, t=t))
            node.drive_request_callback(ros_stubs.make_request(0.0, 0.86))
            modes.append(node.safety_mode)

        drive(world, 0.0, 0.86, 48, on_scan=on_scan)          # the box stays over 1.5 m away
        return modes

    def test_short_glitch_does_not_trigger_aeb(self):
        """Check a two-scan range glitch does not trigger AEB."""
        self.assertNotIn('aeb', self.run_glitch(self.node()))

    def test_glitch_did_trigger_aeb_without_persistence(self):
        # the old behaviour (one scan was enough): this is the false AEB from the car log
        """Check the glitch did trigger AEB with persistence 1 (old behaviour)."""
        self.assertIn('aeb', self.run_glitch(self.node(obstacle_persist_scans=1)))

    def test_track_obstacles_false_ignores_obstacle_speed(self):
        """Check track_obstacles:=false ignores an approaching box."""
        node = self.node(track_obstacles=False)
        walls = ros_stubs.corridor_ranges(1.0)
        for i in range(30):                                    # 2.5 m -> 1.05 m, car stopped
            t = i * 0.025
            ros_stubs.CLOCK.seconds = t
            node.odom_callback(ros_stubs.make_odom(0.0))
            node.fused_scan_callback(ros_stubs.make_scan(
                ros_stubs.add_box(walls, 2.5 - 2.0 * t, -0.2, 0.2), t=t))
        self.assertNotEqual(node.safety_mode, 'aeb')

    def test_yaw_comes_from_the_commanded_steering(self):
        # odometry that publishes angular.z flipped, zero or right gives the same pose
        """Check the pose's yaw ignores the odometry angular.z."""
        yaws = []
        for angular in (1.0, -1.0, 0.0):
            ros_stubs.CLOCK.seconds = 0.0
            node = self.node()
            node.drive_request_callback(ros_stubs.make_request(0.2, 1.0))   # steering 0.2
            for i in range(3):
                ros_stubs.CLOCK.seconds = i * 0.05
                node.odom_callback(ros_stubs.make_odom(1.0, angular))
            yaws.append(node.pose[2])
        expected = 1.0 * math.tan(0.2) / 0.33 * 0.1
        for yaw in yaws:
            self.assertAlmostEqual(yaw, expected)

    def test_aeb_log_names_the_obstacle_speed(self):
        """Check the AEB log line names the obstacle as static."""
        node = self.node()
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 0.8, -0.2, 0.2)
        node.odom_callback(ros_stubs.make_odom(1.5))
        node.fused_scan_callback(ros_stubs.make_scan(box))
        errors = [m for level, m in node.get_logger().records if level == 'error']
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith('AEB engaged: obstacle on the path'))
        self.assertIn('obstacle static', errors[0])


class FacingGateTest(unittest.TestCase):
    """Barriers seen at a shallow angle must not look like they move (the third car test)."""

    def setUp(self):
        """Reset the clock and overrides."""
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()

    def tearDown(self):
        """Clear the parameter overrides."""
        ros_stubs.PARAM_OVERRIDES.clear()

    def node(self, track_obstacles=True):
        """Return a safety node with track_obstacles set."""
        ros_stubs.PARAM_OVERRIDES['safety_node'] = {'track_obstacles': track_obstacles}
        node = SafetyNode()
        ros_stubs.PARAM_OVERRIDES.clear()
        return node

    def steering_error(self, node, commanded=0.25):
        """
        Drive straight down a 1.2 m corridor while the node is told the car steers.

        Args:
            node (SafetyNode): Node under test.
            commanded (float): Steering angle sent to the node (rad).

        Returns:
            float: Largest obstacle speed cap seen (m/s).
        """
        world, worst = corridor_world(0.6), []

        def on_scan(t, ranges, x, y, yaw):
            """Feed a steering request, odometry and the scan; record the obstacle cap."""
            ros_stubs.CLOCK.seconds = t
            node.drive_request_callback(ros_stubs.make_request(commanded, 0.65))
            node.odom_callback(ros_stubs.make_odom(0.65))
            node.fused_scan_callback(ros_stubs.make_scan(ranges, t=t))
            worst.append(node.sources['fused'].obstacle_cap)

        drive(lambda t: world, 0.0, 0.65, 40, on_scan=on_scan)   # really goes straight
        return max(worst)

    def test_tracking_is_off_by_default(self):
        """Check tracking is off by default and gives no obstacle speed."""
        node = SafetyNode()
        self.assertFalse(node.TRACK_OBSTACLES)
        walls = ros_stubs.corridor_ranges(1.0)
        for i in range(30):                                    # a box comes at a stopped car
            t = i * 0.025
            ros_stubs.CLOCK.seconds = t
            node.odom_callback(ros_stubs.make_odom(0.0))
            node.fused_scan_callback(ros_stubs.make_scan(
                ros_stubs.add_box(walls, 2.5 - 2.0 * t, -0.2, 0.2), t=t))
            self.assertEqual(node.sources['fused'].obstacle_cap, 0.0)

    def test_only_surfaces_facing_the_car_count(self):
        """Check the wall across counts as facing and the side walls do not."""
        node = self.node()
        ranges = ros_stubs.world_scan(corridor_world(0.5, wall_x=2.0), 0.0, 0.0, 0.0)
        node.fused_scan_callback(ros_stubs.make_scan(ranges))
        facing = node.facing(node.sources['fused'])
        bearing = np.degrees(ros_stubs.lidar_angles())
        self.assertTrue(np.all(facing[np.abs(bearing) <= 10.0]))        # the wall across
        shallow = (np.abs(bearing) >= 18.0) & (np.abs(bearing) <= 40.0)  # the side walls
        self.assertFalse(np.any(facing[shallow]))

    def test_steering_error_does_not_make_the_barriers_move(self):
        # the node thinks the car turns (0.25 rad commanded) while it goes straight, like a
        # servo that does not follow the command: no obstacle speed and no AEB
        """Check a steering mismatch gives no barrier speed and no AEB."""
        node = self.node()
        self.assertLess(self.steering_error(node), 0.1)
        self.assertNotEqual(node.safety_mode, 'aeb')

    def test_steering_error_did_make_the_barriers_move_without_the_gate(self):
        # the old behaviour: the side walls seemed to rush at the car (car test: 1.2-1.5 m/s)
        """Check the barriers did seem to move without the facing gate."""
        node = self.node()
        node.facing = lambda source: np.ones(source.ranges.size, dtype=bool)
        self.assertGreater(self.steering_error(node), 1.0)

    def test_a_box_coming_at_the_car_keeps_its_speed(self):
        """Check a box coming at the car still gets its speed."""
        node = self.node()
        walls = ros_stubs.corridor_ranges(1.0)
        worst = 0.0
        for i in range(30):                                    # 2 m/s at a stopped car
            t = i * 0.025
            ros_stubs.CLOCK.seconds = t
            node.odom_callback(ros_stubs.make_odom(0.0))
            node.fused_scan_callback(ros_stubs.make_scan(
                ros_stubs.add_box(walls, 2.5 - 2.0 * t, -0.2, 0.2), t=t))
            worst = max(worst, node.sources['fused'].obstacle_cap)
        self.assertGreater(worst, 1.2)                         # 2.0 minus the 0.5 deadband


class ScanLagTest(unittest.TestCase):
    """Scans that arrive late must be matched with where the car was when they were taken."""

    def setUp(self):
        """Build a safety node with tracking on."""
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()
        ros_stubs.PARAM_OVERRIDES['safety_node'] = {'track_obstacles': True}
        self.node = SafetyNode()
        ros_stubs.PARAM_OVERRIDES.clear()

    def odom(self, t, speed=1.0):
        """Feed one stamped odometry message at time t."""
        ros_stubs.CLOCK.seconds = t
        msg = ros_stubs.make_odom(speed)
        msg.header.stamp = ros_stubs.stamp(t)
        self.node.odom_callback(msg)

    def test_pose_at_interpolates_and_extrapolates(self):
        """Check pose_at() interpolates, extrapolates and clamps."""
        for i in range(5):                                     # 1 m/s straight, 0.1 s apart
            self.odom(1.0 + 0.1 * i)
        self.assertAlmostEqual(self.node.pose_at(1.15)[0], 0.15, places=6)   # between
        self.assertAlmostEqual(self.node.pose_at(1.40)[0], 0.40, places=6)   # newest
        self.assertAlmostEqual(self.node.pose_at(1.50)[0], 0.50, places=6)   # ahead of it
        self.assertAlmostEqual(self.node.pose_at(0.50)[0], 0.00, places=6)   # before: oldest

    def test_lag_spike_does_not_create_an_obstacle_speed(self):
        # the second car test: a static scene, scans reaching the node up to 0.4 s late and
        # then in a burst. Matching them with the pose on arrival gave about 2.4 m/s.
        """Check late scans in a burst give no obstacle speed."""
        ys = np.arange(-0.25, 0.25, 0.005)
        world = np.vstack([corridor_world(0.8), np.stack([np.full_like(ys, 4.0), ys], 1)])
        queue, worst = [], 0.0
        for i in range(70):
            t = 1.0 + i * 0.025
            self.odom(t)
            self.node.drive_request_callback(ros_stubs.make_request(0.0, 1.0))
            # 0.4 s lag spike from 1.5 to 1.8 s; scans are delivered in arrival order
            lag = 0.4 * (t - 1.5) / 0.3 + 0.02 if 1.5 <= t < 1.8 else 0.02
            ranges = ros_stubs.world_scan(world, t - 1.0, 0.0, 0.0)   # car at 1 m/s from x=0
            queue.append((t + lag, ros_stubs.make_scan(ranges, t=t)))
            while queue and queue[0][0] <= t + 1e-9:
                self.node.fused_scan_callback(queue.pop(0)[1])
                worst = max(worst, self.node.sources['fused'].obstacle_cap)
        self.assertLess(worst, 0.1)
        self.assertNotEqual(self.node.safety_mode, 'aeb')

    def test_old_scan_counts_as_closer(self):
        # a box 1.0 m ahead seen 0.3 s ago at 1 m/s is really about 0.7 m ahead now
        """Check a late scan counts the distance driven since it was taken."""
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 1.0, -0.2, 0.2)
        self.odom(1.0)
        fresh = SafetyNode()
        ros_stubs.CLOCK.seconds = 1.0
        fresh.odom_callback(ros_stubs.make_odom(1.0))
        fresh.fused_scan_callback(ros_stubs.make_scan(box, t=1.0))
        ros_stubs.CLOCK.seconds = 1.3
        self.node.fused_scan_callback(ros_stubs.make_scan(box, t=1.0))     # 0.3 s late
        self.assertEqual(fresh.safety_mode, 'normal')          # 0.75 m gap at 1 m/s: fine
        self.assertEqual(self.node.safety_mode, 'aeb')         # only 0.45 m really: brake

    def test_lag_is_reported(self):
        """Check the scan lag is on /safety/state and not logged."""
        ros_stubs.CLOCK.seconds = 1.3
        self.node.fused_scan_callback(ros_stubs.make_scan(ros_stubs.corridor_ranges(1.0), t=1.0))
        self.node.watchdog_callback()
        state = self.node.publishers_by_topic['safety/state'].messages[-1].data
        self.assertIn('lag=300ms', state)                     # on the topic, not in the log
        self.assertEqual(self.node.get_logger().records, [])

    def test_drive_requests_reuse_the_obstacle_speeds(self):
        """Check obstacle speeds are computed once per scan, not per request."""
        calls = []
        tracker = self.node.sources['fused'].tracker
        original = tracker.approach_speeds
        tracker.approach_speeds = lambda px, py: calls.append(1) or original(px, py)
        ros_stubs.CLOCK.seconds = 1.0
        self.node.fused_scan_callback(ros_stubs.make_scan(ros_stubs.corridor_ranges(1.0), t=1.0))
        for _ in range(5):
            self.node.drive_request_callback(ros_stubs.make_request(0.1, 1.0))
        self.assertEqual(len(calls), 1)                        # once per scan, not per request


class DebugTopicTest(unittest.TestCase):
    """safety/debug: one line of JSON per drive request, only while someone listens."""

    def setUp(self):
        """Build a safety node and keep its debug publisher."""
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()
        self.node = SafetyNode()
        self.debug = self.node.publishers_by_topic['safety/debug']

    def request(self, t, ranges, speed, requested):
        """Feed odometry, a scan and a drive request at time t."""
        ros_stubs.CLOCK.seconds = t
        self.node.odom_callback(ros_stubs.make_odom(speed))
        self.node.fused_scan_callback(ros_stubs.make_scan(ranges, t=t))
        self.node.drive_request_callback(ros_stubs.make_request(0.0, requested))

    def test_nothing_is_published_without_a_subscriber(self):
        """Check no debug line is built without a subscriber."""
        self.request(1.0, ros_stubs.corridor_ranges(1.0), 0.5, 1.0)
        self.assertEqual(len(self.node.publishers_by_topic['drive'].messages), 1)
        self.assertEqual(self.debug.messages, [])

    def test_clear_path(self):
        """Check the debug line on a clear path."""
        self.debug.subscribers = 1
        self.request(1.0, ros_stubs.corridor_ranges(1.0), 0.5, 0.8)
        line = json.loads(self.debug.messages[-1].data)
        self.assertEqual((line['t'], line['mode'], line['scan']), (1.0, 'normal', 'fused'))
        self.assertEqual((line['requested'], line['commanded'], line['v']), (0.8, 0.8, 0.5))

    def test_soft_cap_and_aeb_are_explained(self):
        """Check the debug line explains the soft cap and AEB."""
        self.debug.subscribers = 1
        box = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 1.0, -0.2, 0.2)
        self.request(1.0, box, 0.3, 1.0)               # slow car, box 1 m ahead: capped
        line = json.loads(self.debug.messages[-1].data)
        self.assertEqual(line['mode'], 'slow')
        self.assertAlmostEqual(line['path'], 1.0, places=2)
        self.assertLess(line['commanded'], 1.0)
        self.assertAlmostEqual(line['commanded'], line['cap'], places=3)
        self.request(1.025, box, 1.5, 1.0)             # the same box at 1.5 m/s: AEB
        line = json.loads(self.debug.messages[-1].data)
        self.assertEqual((line['mode'], line['commanded']), ('aeb', 0.0))
        self.assertLess(line['ttc'], line['ttc_needed'])
