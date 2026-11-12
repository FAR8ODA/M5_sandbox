"""Tests for the fused (camera + LiDAR + odometry) lap counter."""

import unittest

import numpy as np
import ros_stubs

from milestone4.lap_counter_node import LapCounterNode, unit_signature  # noqa: I100


def start_view():
    """Return the LiDAR ranges at the start line: a corridor with a box on the left."""
    return ros_stubs.add_box(ros_stubs.corridor_ranges(0.8), 1.5, 0.2, 0.6)


def away_view():
    """Return the LiDAR ranges in a corner: wall across the track ahead, open to the left."""
    a = ros_stubs.lidar_angles()
    with np.errstate(divide='ignore'):
        right_wall = np.where(np.sin(a) < 0.0, 0.8 / np.abs(np.sin(a)), 10.0)
    return ros_stubs.add_box(np.minimum(right_wall, 10.0), 1.2, -5.0, 0.6)


def straight_view():
    """Return the LiDAR ranges on another straight, which looks a lot like the start."""
    return ros_stubs.corridor_ranges(0.8)


class SignatureTest(unittest.TestCase):
    """LiDAR signatures and similarity."""

    def setUp(self):
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()
        self.node = LapCounterNode()

    def signature(self, ranges):
        return self.node.lidar_signature(self.node.lidar_profile(ros_stubs.make_scan(ranges)))

    def test_same_view_scores_one(self):
        s = self.signature(start_view())
        self.assertAlmostEqual(float(np.dot(s, s)), 1.0)

    def test_corner_scores_low_and_straight_scores_high(self):
        start = self.signature(start_view())
        self.assertLess(float(np.dot(start, self.signature(away_view()))), 0.5)   # re-arms
        # another straight looks alike: this is why the odometry gate and the peak are needed
        self.assertGreater(float(np.dot(start, self.signature(straight_view()))), 0.8)

    def test_flat_profile_has_no_signature(self):
        self.assertFalse(unit_signature(np.full(10, 3.0)).any())


class LateCameraTest(unittest.TestCase):
    """The camera reference when the RealSense starts after the LiDAR."""

    def setUp(self):
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()
        self.node = LapCounterNode()
        # stand in for cv_bridge and OpenCV: every frame becomes the same thumbnail
        self.node.bridge.imgmsg_to_cv2 = lambda msg, desired_encoding='bgr8': np.zeros((4, 4, 3))
        self.node.make_thumbnail = lambda bgr: np.arange(64 * 48 * 3, dtype=np.float32).reshape(
            48, 64, 3) % 17

    def run_start(self, camera_from, speed=0.0, until=1.5):
        """Run the start: LiDAR from t=0, camera frames from camera_from seconds."""
        node, t = self.node, 0.0
        while t < until:
            ros_stubs.CLOCK.seconds = t
            node.odom_callback(ros_stubs.make_odom(speed))
            node.lidar_callback(ros_stubs.make_scan(start_view(), t=t))
            if t >= camera_from:
                node.image_callback(ros_stubs.Image())
            node.tick()
            t += 0.05

    def test_late_camera_still_gets_a_reference(self):
        self.run_start(camera_from=1.0)                       # LiDAR reference at 0.3 s
        self.assertIsNotNone(self.node.lidar_reference)
        self.assertIsNotNone(self.node.cam_reference)

    def test_camera_after_the_car_left_is_not_used(self):
        self.run_start(camera_from=1.0, speed=1.0)            # 1 m away by then
        self.assertIsNone(self.node.cam_reference)            # laps on the LiDAR only
        self.assertIsNotNone(self.node.lidar_reference)
        self.assertEqual(self.node.get_logger().records, [])  # and nothing logged


class CounterTest(unittest.TestCase):
    """The counting state machine and the odometry distance gate."""

    def setUp(self):
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()
        ros_stubs.PARAM_OVERRIDES['lap_counter_node'] = {'use_camera': False, 'num_laps': 2}
        self.node = LapCounterNode()

    def tearDown(self):
        ros_stubs.PARAM_OVERRIDES.clear()

    def search(self, distance):
        """Put the counter in the armed search phase with some distance driven."""
        self.node.phase = 'search'
        self.node.armed = True
        self.node.streak = 0
        self.node.last_odom_time = 0.0
        self.node.distance = distance

    def test_distance_gate_blocks_an_early_lap(self):
        self.search(distance=2.0)                             # below min_lap_distance (5 m)
        laps = [self.node.update_counter(0.95, t) for t in (0.0, 0.05, 0.1)]
        self.assertFalse(any(laps))
        self.assertFalse(self.node.armed)                     # must leave the start view again

    def feed(self, scores):
        """Feed scores 0.05 s apart; return which ticks counted a lap."""
        return [self.node.update_counter(s, i * 0.05) for i, s in enumerate(scores)]

    def test_lap_counts_at_the_peak(self):
        self.search(distance=6.0)
        # confirmed on the 3rd tick, still rising, counted once the score falls past its peak
        laps = self.feed([0.85, 0.85, 0.86, 0.90, 0.97, 0.99, 0.93])
        self.assertEqual(laps, [False] * 6 + [True])
        self.assertAlmostEqual(self.node.first_lap_distance, 6.0)
        self.assertAlmostEqual(self.node.lap_gate(), 5.1)     # max(5, 0.85 x 6)

    def test_flat_peak_counts_after_the_timeout(self):
        self.search(distance=6.0)
        laps = self.feed([0.9] * 30)
        self.assertTrue(any(laps))
        self.assertEqual(sum(laps), 1)
        self.assertGreaterEqual(laps.index(True) * 0.05, 1.0)  # PEAK_TIMEOUT after confirming

    def test_gate_follows_the_first_lap(self):
        self.node.first_lap_distance = 20.0
        self.assertAlmostEqual(self.node.lap_gate(), 17.0)

    def test_no_odometry_skips_the_gate(self):
        self.search(distance=0.0)
        self.node.last_odom_time = None
        laps = self.feed([0.95, 0.95, 0.95, 0.5])
        self.assertTrue(laps[-1])

    def test_arms_on_distance_when_the_score_stays_high(self):
        # between solid walls the LiDAR score never drops below rearm_threshold (0.64 at its
        # lowest in the closed-loop simulator), so the score alone never armed the counter
        self.node.phase = 'search'
        self.node.last_odom_time = 0.0
        self.node.distance = 2.0                              # under half the 5 m lap gate
        self.node.update_counter(0.65, 0.0)
        self.assertFalse(self.node.armed)
        self.node.distance = 6.0                              # past the gate
        laps = self.feed([0.65, 0.9, 0.9, 0.9, 0.95, 0.9])
        self.assertTrue(self.node.armed or any(laps))
        self.assertEqual(sum(laps), 1)

    def test_no_odometry_still_needs_the_score_drop(self):
        self.node.phase = 'search'
        self.node.distance = 100.0                            # meaningless without odometry
        self.assertFalse(any(self.feed([0.65, 0.9, 0.9, 0.9, 0.95, 0.9])))
        self.assertFalse(self.node.armed)

    def test_full_run_on_lidar_only(self):
        node = self.node
        t = 0.0

        def tick(view, speed):
            nonlocal t
            ros_stubs.CLOCK.seconds = t
            node.odom_callback(ros_stubs.make_odom(speed))
            node.lidar_callback(ros_stubs.make_scan(view, t=t))
            node.tick()
            t += 0.05

        for _ in range(10):                                   # reference at the start line
            tick(start_view(), 0.0)
        self.assertEqual(node.phase, 'blind')
        for _ in range(140):                                  # 7 s away from the start, 2 m/s
            tick(away_view(), 2.0)
        self.assertTrue(node.armed)
        for _ in range(5):                                    # back at the start
            tick(start_view(), 2.0)
        self.assertEqual(node.laps, 0)                        # still at the peak
        tick(straight_view(), 2.0)                            # just past the start line
        self.assertEqual(node.laps, 1)
        self.assertEqual(node.publishers_by_topic['lap_count'].messages[-1].data, 1)

        for _ in range(140):
            tick(away_view(), 2.0)
        for _ in range(5):
            tick(start_view(), 2.0)
        tick(straight_view(), 2.0)
        self.assertEqual(node.laps, 2)
        self.assertTrue(node.done)

        stops = node.publishers_by_topic['drive_request'].messages
        self.assertLess(stops[-1].drive.speed, 0.0)           # latched stop requested

        # the only log lines: one per lap, the last one saying the car stops
        logs = node.get_logger().records
        self.assertEqual([level for level, _ in logs], ['info', 'info'])
        self.assertRegex(logs[0][1], r'^Lap 1/2 \(score \d\.\d\d, \d+\.\d m driven\)$')
        self.assertRegex(logs[1][1], r'^Lap 2/2 \(score \d\.\d\d, \d+\.\d m driven\): stopping$')


if __name__ == '__main__':
    unittest.main()
