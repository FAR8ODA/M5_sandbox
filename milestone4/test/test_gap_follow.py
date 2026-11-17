"""Tests for the disparity-extender gap follow node."""

import math
import unittest

import numpy as np
import ros_stubs

from milestone4.braking import slow_speed_cap  # noqa: I100  (after ros_stubs)
from milestone4.gap_follow_node import GapFollowNode  # noqa: I100
from milestone4.safety_node import SafetyNode  # noqa: I100


class GapFollowTest(unittest.TestCase):
    """Disparity extension, target choice, cornering check, speed and scan fallback."""

    def setUp(self) -> None:
        """Reset the clock and overrides; build a node and the LiDAR beam angles."""
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()
        self.node = GapFollowNode()
        self.angles = ros_stubs.lidar_angles()

    def requests(self) -> list:
        """Return every drive request the node has published."""
        return self.node.publishers_by_topic['drive_request'].messages

    def test_disparity_is_extended_by_half_width_plus_margin(self) -> None:
        """Check a disparity is extended by exactly half the car plus the margin."""
        ranges = np.full(200, 4.0)
        ranges[100:120] = 1.0                                 # a box 1 m away
        extended = self.node.extend_disparities(ranges, np.zeros(200, bool), 0.01)

        self.assertAlmostEqual(self.node.MARGIN, 0.20)       # 0.35 m from the car's centre
        count = math.ceil(math.atan2(0.15 + 0.20, 1.0) / 0.01)  # beams covering 0.35 m at 1 m
        self.assertTrue(np.all(extended[100 - count:120 + count] == 1.0))
        self.assertEqual(extended[100 - count - 1], 4.0)
        self.assertEqual(extended[120 + count], 4.0)

    def test_missing_returns_at_an_edge_do_not_hide_it(self) -> None:
        # black barriers often give no return right at their edge: 3 invalid beams between
        # the barrier (1 m) and the open track (4 m). Comparing only adjacent beams found no
        # disparity there and extended nothing.
        """Check invalid beams at a barrier edge do not hide the disparity."""
        inc = math.radians(0.25)
        count = math.ceil(math.atan2(0.15 + 0.20, 1.0) / inc)
        for far_side_up in (True, False):
            ranges = np.r_[np.full(80, 1.0), np.zeros(3), np.full(120, 4.0)]
            if not far_side_up:
                ranges = ranges[::-1].copy()
            extended = self.node.extend_disparities(ranges, np.zeros(ranges.size, bool), inc)
            changed = int(np.sum((extended < 4.0) & (ranges == 4.0)))
            self.assertEqual(changed, count - 3)               # counted from the edge
            self.assertTrue(np.all(extended[ranges == 0.0] == 0.0))   # still invalid

    def test_confirmed_box_gets_the_wider_margin(self) -> None:
        """Check a camera-confirmed box is extended by box_margin."""
        ranges = np.full(200, 4.0)
        ranges[100:120] = 1.0
        boxes = np.zeros(200, bool)
        boxes[100:120] = True                                 # the camera says it is a box
        extended = self.node.extend_disparities(ranges, boxes, 0.01)

        count = math.ceil(math.atan2(0.15 + 0.25, 1.0) / 0.01)
        self.assertTrue(np.all(extended[100 - count:100] == 1.0))
        self.assertEqual(extended[100 - count - 1], 4.0)

    def test_near_corner_of_a_box_seen_from_the_side_keeps_its_margin(self) -> None:
        # a box seen from the side: its side face runs from the disparity (1.2 m, beam 100)
        # to the near corner (0.8 m, beam 104), then the front face. The margin is 0.35 m at
        # the near corner too, which needs more beams than at the far edge of the side face
        """Check the extension covers the margin at a box's near corner."""
        ranges = np.full(200, 4.0)
        ranges[100:105] = np.linspace(1.2, 0.8, 5)
        ranges[105:140] = 0.8
        extended = self.node.extend_disparities(ranges, np.zeros(200, bool), 0.01)

        at_edge = 100 - math.ceil(math.atan2(0.35, 1.2) / 0.01)
        at_corner = 104 - math.ceil(math.atan2(0.35, 0.8) / 0.01)
        self.assertLess(at_corner, at_edge)                   # the corner reaches further
        self.assertTrue(np.all(extended[at_corner:100] == 1.2))
        self.assertEqual(extended[at_corner - 1], 4.0)

    def test_invalid_beams_are_not_disparities(self) -> None:
        """Check a single invalid beam does not create a disparity."""
        ranges = np.full(200, 4.0)
        ranges[50] = 0.0                                      # one invalid reading
        extended = self.node.extend_disparities(ranges, np.zeros(200, bool), 0.01)
        self.assertEqual(int((extended == 4.0).sum()), 199)

    def drive_corridor(self, seconds) -> list:
        """
        Feed a straight corridor for some seconds at 40 Hz.

        Args:
            seconds (float): How long to drive (s).

        Returns:
            list: The drive part of every request published.
        """
        scan = ros_stubs.make_scan(ros_stubs.corridor_ranges(0.6))
        for i in range(int(seconds / 0.025)):
            ros_stubs.CLOCK.seconds = i * 0.025
            self.node.fused_callback(scan)
        return [m.drive for m in self.requests()]

    def test_straight_corridor_drives_straight_at_full_speed(self) -> None:
        """Check a straight corridor gives straight steering at max_speed."""
        requests = self.drive_corridor(1.5)
        self.assertLess(abs(requests[-1].steering_angle), 0.02)
        self.assertAlmostEqual(requests[-1].speed, 1.5, places=3)        # max_speed

    def test_speed_ramps_up_at_a_accel(self) -> None:
        """Check the requested speed rises by at most a_accel per second."""
        speeds = [r.speed for r in self.drive_corridor(1.0)]
        rises = np.diff([0.0] + speeds)
        self.assertLessEqual(float(rises.max()), 2.0 * 0.025 + 1e-9)   # a_accel x dt
        self.assertAlmostEqual(speeds[int(0.5 / 0.025) - 1], 1.0, places=2)  # 1 m/s after 0.5 s

    def test_target_avoids_a_box_on_the_right(self) -> None:
        """Check the target and steering go left of a box on the right."""
        ranges = ros_stubs.add_box(ros_stubs.corridor_ranges(0.6), 1.5, -0.3, 0.1)
        r, a = self.node.preprocess_scan(ros_stubs.make_scan(ranges))
        extended = self.node.extend_disparities(r, np.zeros(r.size, bool), a[1] - a[0])
        index = self.node.find_target(extended, a)

        self.assertIsNotNone(index)
        self.assertGreater(a[index], math.atan2(0.1, 1.5))    # left of the box's left edge

        self.node.fused_callback(ros_stubs.make_scan(ranges))
        self.assertGreater(self.requests()[-1].drive.steering_angle, 0.0)   # steers left

    def corner(self, ranges, steering) -> float:
        """Run side_distance() and corner_check() on ranges, as drive_on() does."""
        beside = self.node.side_distance(ranges, self.angles, steering)
        return self.node.corner_check(steering, beside)

    def test_corner_check_blocks_a_turn_into_the_side(self) -> None:
        """Check corner_check cancels a turn towards something beside the car."""
        r = np.full(self.angles.size, 3.0)
        r[np.abs(self.angles - math.radians(100.0)) < math.radians(2.0)] = 0.15  # left side
        self.assertEqual(self.corner(r, 0.3), 0.0)
        self.assertEqual(self.corner(r, -0.3), -0.3)

    def test_slows_for_the_straight_path_when_the_turn_may_be_cancelled(self) -> None:
        # a wall 1 m ahead and to the right, open to the left, and the car already turning
        # hard left at 1 m/s: the turn's own path is free. With something 0.3 m beside the
        # car on the left (nearer than side_clear + 0.15 m), corner_check may cancel the turn
        # on the next scan, so the speed must already suit the straight path into the wall.
        # (It was the turn's speed, and AEB fired the moment the turn was cancelled.)
        """Check the speed suits the straight path when the turn may be cancelled."""
        ranges = np.where(self.angles > math.radians(12.0), 4.0, 1.0)
        beside = np.abs(self.angles - math.radians(100.0)) < math.radians(2.0)
        speeds = []
        ros_stubs.PARAM_OVERRIDES['gap_follow_node'] = {'max_speed': 1.0}
        for side in (3.0, 0.3):
            ros_stubs.CLOCK.seconds = 0.0
            node = GapFollowNode()
            node.previous_speed, node.previous_steering = 1.0, 0.4
            ranges[beside] = side
            node.fused_callback(ros_stubs.make_scan(ranges))
            drive = node.publishers_by_topic['drive_request'].messages[-1].drive
            self.assertGreater(drive.steering_angle, 0.3)     # still turning left
            speeds.append(drive.speed)
        ros_stubs.PARAM_OVERRIDES.clear()
        self.assertAlmostEqual(speeds[0], 1.0)                # nothing beside: the turn's speed
        wall = math.cos(math.asin(self.node.CORRIDOR))        # nearest wall point on the path
        self.assertAlmostEqual(speeds[1], self.node.braking_limit(wall), places=2)

    def test_corner_check_ignores_the_car_itself(self) -> None:
        """Check readings of the car's own body are ignored by corner_check."""
        r = np.full(self.angles.size, 3.0)
        r[np.abs(self.angles - math.radians(100.0)) < math.radians(2.0)] = 0.05   # own body
        r[np.abs(self.angles - math.radians(130.0)) < math.radians(2.0)] = 0.15   # behind
        self.assertEqual(self.corner(r, 0.3), 0.3)

    def test_clear_straight_gives_max_speed(self) -> None:
        """Check a clear straight path gives max_speed."""
        self.assertAlmostEqual(self.node.calculate_speed(math.inf, 0.0, False), 1.5)
        self.assertAlmostEqual(self.node.calculate_speed(20.0, 0.0, False), 1.5)

    def test_braking_limit(self) -> None:
        # keep plan_factor (2.5) x the TTC needed to stop for whatever is 1 m along the path
        """Check the braking limit keeps plan_factor x the needed TTC."""
        expected = float(slow_speed_cap(1.0 - 0.25, 0.0, 2.5, 0.15, 2.0, 0.1))
        self.assertAlmostEqual(self.node.calculate_speed(1.0, 0.0, False), expected)
        self.assertLess(expected, 0.75)                        # about 0.70 m/s for 1 m
        self.assertGreater(self.node.calculate_speed(3.0, 0.0, False), expected)

    def test_braking_limit_below_min_speed_wins(self) -> None:
        # min_speed keeps the car moving, but never faster than it can stop
        """Check a braking limit below min_speed is not lifted."""
        brake = self.node.braking_limit(0.4)
        self.assertLess(brake, self.node.MIN_SPEED)
        self.assertAlmostEqual(self.node.calculate_speed(0.4, 0.0, False), brake)

    def test_cornering_limit(self) -> None:
        """Check the cornering limit sqrt(a_lat / curvature) and when it applies."""
        curvature = math.tan(0.4) / 0.33
        limit = math.sqrt(2.5 / curvature)                               # about 1.4 m/s
        self.assertAlmostEqual(self.node.cornering_limit(0.4), limit)
        self.assertAlmostEqual(self.node.cornering_limit(-0.4), limit)
        # at the default 1.5 m/s top speed full lock is the limit, a gentle turn is not
        self.assertAlmostEqual(self.node.calculate_speed(math.inf, 0.4, False), limit)
        self.assertAlmostEqual(self.node.calculate_speed(math.inf, 0.1, False), 1.5)
        # with a 1.0 m/s top speed the corners are never the limit
        ros_stubs.PARAM_OVERRIDES['gap_follow_node'] = {'max_speed': 1.0}
        slow = GapFollowNode()
        ros_stubs.PARAM_OVERRIDES.clear()
        self.assertAlmostEqual(slow.calculate_speed(math.inf, 0.4, False), 1.0)

    def test_box_pass_cap(self) -> None:
        """Check passing a box caps the speed at pass_speed."""
        self.assertAlmostEqual(self.node.calculate_speed(math.inf, 0.0, True), 0.8)

    def test_planner_never_asks_more_than_safety_allows(self) -> None:
        # same scans, same braking model: the planner's request is inside the safety band,
        # so the safety node should not have to cut it
        """Check the planner's request is never above the safety node's cap."""
        for x in (0.8, 1.2, 2.0, 3.0, 5.0):
            ranges = ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), x, -0.2, 0.2)
            scan = ros_stubs.make_scan(ranges)
            r, a = self.node.preprocess_scan(scan)
            free = self.node.path_clearance(scan, a, 0.0)
            request = self.node.calculate_speed(free, 0.0, False)
            safety = SafetyNode()
            safety.fused_scan_callback(scan)
            _, cap, _, _, _ = safety.check_path(safety.sources['fused'], 0.0, request)
            self.assertLessEqual(request, cap + 1e-9, f'box at {x} m')

    def test_steering_is_smoothed(self) -> None:
        # MS3 drove this track with 0.7 on the previous command; 0.3 twitched on the car
        """Check the steering is pure pursuit blended with the previous command."""
        self.assertAlmostEqual(self.node.SMOOTHING, 0.6)
        # pure pursuit from the rear axle, 0.27 m behind the LiDAR; lookahead at its 0.5 m minimum
        gx, gy = 0.5 * math.cos(0.1) + 0.27, 0.5 * math.sin(0.1)
        raw = math.atan(0.33 * 2.0 * gy / (gx * gx + gy * gy))
        first = self.node.calculate_steering_angle(0.1, 5.0)   # from straight ahead
        self.assertAlmostEqual(first, 0.4 * raw)
        second = self.node.calculate_steering_angle(0.1, 5.0)
        self.assertAlmostEqual(second, 0.6 * first + 0.4 * raw)

    def test_pure_pursuit_from_the_rear_axle(self) -> None:
        # treating the LiDAR as the rear axle (lidar_x 0, the old formula) turns harder
        """Check pure pursuit from the rear axle turns less than from the LiDAR."""
        self.node.SMOOTHING = 0.0
        axle = self.node.calculate_steering_angle(0.3, 5.0)
        self.node.LIDAR_X = 0.0
        lidar = self.node.calculate_steering_angle(0.3, 5.0)
        self.assertAlmostEqual(lidar, math.atan(2.0 * 0.33 * math.sin(0.3) / 0.5))
        self.assertGreater(lidar, axle)
        self.assertGreater(axle, 0.0)

    def test_lookahead_follows_the_measured_speed(self) -> None:
        # asking to slow down must not shorten the lookahead before the car has slowed
        """Check the lookahead uses the measured, not the requested, speed."""
        self.node.odom_callback(ros_stubs.make_odom(1.5))
        self.node.previous_speed = 0.3
        fast = self.node.calculate_steering_angle(0.3, 5.0)
        self.node.previous_steering = 0.0
        self.node.previous_speed = 1.5
        self.assertAlmostEqual(fast, self.node.calculate_steering_angle(0.3, 5.0))

    def test_box_ahead_uses_camera_labels(self) -> None:
        """Check box_ahead() needs camera box labels."""
        ranges = ros_stubs.add_box(ros_stubs.corridor_ranges(0.6), 1.0, -0.2, 0.2)
        r, a = self.node.preprocess_scan(ros_stubs.make_scan(ranges))
        boxes = np.zeros(r.size, bool)
        self.assertFalse(self.node.box_ahead(r, a, boxes))
        boxes[np.abs(a) < math.radians(5.0)] = True
        self.assertTrue(self.node.box_ahead(r, a, boxes))

    def test_raw_scan_used_only_when_fused_is_stale(self) -> None:
        """Check the raw scan drives only while the fused scan is stale."""
        scan = ros_stubs.make_scan(ros_stubs.corridor_ranges(0.6))
        self.node.fused_callback(scan)
        ros_stubs.CLOCK.seconds = 0.05
        self.node.raw_callback(scan)
        self.assertEqual(len(self.requests()), 1)             # fused is fresh: raw ignored

        ros_stubs.CLOCK.seconds = 1.0
        self.node.raw_callback(scan)
        self.assertEqual(len(self.requests()), 2)             # fused stale: drive on raw

    def test_no_gap_publishes_nothing(self) -> None:
        """Check no request is published when there is no gap."""
        self.node.fused_callback(ros_stubs.make_scan(np.full(ros_stubs.LIDAR_BEAMS, 0.2)))
        self.assertEqual(len(self.requests()), 0)


class BoxLabelDebounceTest(unittest.TestCase):
    """Flickering camera box labels must not toggle the box margin or the passing speed."""

    def setUp(self) -> None:
        """Build a node and labelled / empty /obstacle_scan ranges."""
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()
        self.node = GapFollowNode()
        self.labelled = np.full(ros_stubs.LIDAR_BEAMS, np.inf)
        self.labelled[530:550] = 1.2                           # a box 1.2 m ahead
        self.empty = np.full(ros_stubs.LIDAR_BEAMS, np.inf)

    def label(self, t, on) -> bool:
        """
        Feed one /obstacle_scan at time t.

        Args:
            t (float): Time (s).
            on (bool): True to label the box, False for no labels.

        Returns:
            bool: Whether the box's centre beam counts as a box afterwards.
        """
        ros_stubs.CLOCK.seconds = t
        self.node.obstacle_callback(ros_stubs.make_scan(self.labelled if on else self.empty, t=t))
        return bool(self.node.box_beams(ros_stubs.LIDAR_BEAMS)[540])

    def test_one_label_in_three_is_not_a_box(self) -> None:
        """Check one label in three scans is not a box."""
        self.label(0.000, False)
        self.label(0.025, False)
        self.assertFalse(self.label(0.050, True))

    def test_two_labels_in_three_are_a_box_held_for_box_hold(self) -> None:
        """Check two labels in three make a box, held for box_hold."""
        self.label(0.000, False)
        self.label(0.025, True)
        self.assertTrue(self.label(0.050, True))
        self.assertTrue(self.label(0.075, False))              # held ...
        self.assertTrue(self.label(0.300, False))
        self.assertFalse(self.label(0.400, False))             # ... for 0.3 s

    def test_passing_is_held_for_pass_hold(self) -> None:
        """Check passing mode is held for pass_hold after the box."""
        r = np.full(ros_stubs.LIDAR_BEAMS, 3.0)
        a = ros_stubs.lidar_angles()
        boxes = np.zeros(r.size, bool)
        boxes[530:550] = True
        r[530:550] = 1.0
        self.assertTrue(self.node.box_ahead(r, a, boxes))
        ros_stubs.CLOCK.seconds = 0.4
        self.assertTrue(self.node.box_ahead(r, a, np.zeros(r.size, bool)))   # still passing
        ros_stubs.CLOCK.seconds = 0.6
        self.assertFalse(self.node.box_ahead(r, a, np.zeros(r.size, bool)))  # 0.5 s later

    def test_flickering_labels_start_one_pass(self) -> None:
        # labels on in every other scan for a second, as in the car log
        """Check labels flickering every other scan start only one pass."""
        scan = ros_stubs.make_scan(ros_stubs.add_box(ros_stubs.corridor_ranges(1.0), 1.2,
                                                     -0.2, 0.2))
        passing = []
        for i in range(40):
            t = i * 0.025
            self.label(t, i % 2 == 0)
            self.node.fused_callback(scan)
            passing.append(t < self.node.pass_until)
        self.assertEqual(sum(now and not before for before, now
                             in zip([False] + passing, passing)), 1)
        self.assertEqual(self.node.get_logger().records, [])   # nothing logged


if __name__ == '__main__':
    unittest.main()
