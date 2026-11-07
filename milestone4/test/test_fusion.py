"""Tests for the LiDAR-camera fusion rules (fusion.py) and fusion_node."""

import math
import unittest

import numpy as np
import ros_stubs

from milestone4 import fusion_node  # noqa: I100  (after ros_stubs, which fakes ROS)
from milestone4.fusion import (camera_to_lidar, CameraPoints, clean_lidar,  # noqa: I100
                                       FusionParams, points_to_beams, scan_to_points,
                                       ScanFuser)


def face_points(x, y_min, y_max, is_box, count=40):
    """Return camera points along a face x metres ahead, placed on the LiDAR beams."""
    ys = np.linspace(y_min, y_max, count)
    beams, ranges = points_to_beams(np.full(count, x), ys, ros_stubs.LIDAR_ANGLE_MIN,
                                    ros_stubs.LIDAR_INCREMENT, ros_stubs.LIDAR_BEAMS)
    return CameraPoints(beams=beams, ranges=ranges, is_box=is_box)


def camera_scan(x, y_min, y_max, t, range_max=4.0):
    """Return a camera-style LaserScan (180 beams, +/-35 degrees) seeing a face x m ahead."""
    a = np.linspace(-math.radians(35.0), math.radians(35.0), 180)
    r_face = x / np.cos(a)
    y_face = r_face * np.sin(a)
    ranges = np.where((y_face >= y_min) & (y_face <= y_max), r_face, range_max)
    return ros_stubs.make_scan(ranges, angle_min=float(a[0]), angle_increment=float(a[1] - a[0]),
                               range_min=0.05, range_max=range_max, t=t)


class FusionRulesTest(unittest.TestCase):
    """The association and persistence rules, without ROS."""

    def setUp(self):
        self.angles = ros_stubs.lidar_angles()

    def test_box_on_a_lidar_return_is_confirmed(self):
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.2, 0.2)
        fused, obstacle = ScanFuser(FusionParams()).fuse(
            lidar, 0.06, 10.0, [face_points(1.5, -0.2, 0.2, is_box=True)])

        self.assertTrue(np.allclose(fused, lidar))           # the LiDAR scan is unchanged
        confirmed = np.isfinite(obstacle)
        self.assertGreater(confirmed.sum(), 10)
        self.assertTrue(np.allclose(obstacle[confirmed], lidar[confirmed]))  # LiDAR range kept
        y = obstacle[confirmed] * np.sin(self.angles[confirmed])
        self.assertTrue(np.all(np.abs(y) <= 0.25))            # only beams on the box face

    def test_depth_point_on_a_lidar_return_is_not_a_box(self):
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.2, 0.2)
        fused, obstacle = ScanFuser(FusionParams()).fuse(
            lidar, 0.06, 10.0, [face_points(1.5, -0.2, 0.2, is_box=False)])
        self.assertTrue(np.allclose(fused, lidar))
        self.assertFalse(np.isfinite(obstacle).any())

    def test_camera_only_point_needs_persistence(self):
        lidar = ros_stubs.corridor_ranges()                  # the LiDAR misses a low object
        low = face_points(0.8, -0.1, 0.1, is_box=False)
        fuser = ScanFuser(FusionParams(persist_frames=3))

        for _ in range(2):
            fused, obstacle = fuser.fuse(lidar, 0.06, 10.0, [low])
            self.assertTrue(np.allclose(fused, lidar))
            self.assertFalse(np.isfinite(obstacle).any())

        fused, obstacle = fuser.fuse(lidar, 0.06, 10.0, [low])
        self.assertTrue(np.allclose(fused[low.beams], low.ranges, atol=0.02))
        # a depth-only object is not a box: no box margin or pass speed for it (on the car,
        # floor that slipped past the floor rejection made the planner slow all lap)
        self.assertFalse(np.isfinite(obstacle).any())
        self.assertTrue(fuser.trusted[low.beams].all())

    def test_reused_camera_frame_does_not_count(self):
        lidar = ros_stubs.corridor_ranges()
        low = face_points(0.8, -0.1, 0.1, is_box=False)
        fuser = ScanFuser(FusionParams(persist_frames=3))
        fuser.fuse(lidar, 0.06, 10.0, [low], new_camera_frame=True)
        for _ in range(5):                                   # LiDAR 40 Hz, same camera frame
            fused, _ = fuser.fuse(lidar, 0.06, 10.0, [low], new_camera_frame=False)
        self.assertTrue(np.allclose(fused, lidar))

    def test_missing_frame_resets_persistence(self):
        lidar = ros_stubs.corridor_ranges()
        low = face_points(0.8, -0.1, 0.1, is_box=False)
        elsewhere = face_points(1.0, 0.4, 0.5, is_box=False)  # 22-27 degrees left, far from it
        fuser = ScanFuser(FusionParams(persist_frames=3))
        fuser.fuse(lidar, 0.06, 10.0, [low])
        fuser.fuse(lidar, 0.06, 10.0, [low])
        fuser.fuse(lidar, 0.06, 10.0, [elsewhere])            # the low object not seen
        fuser.fuse(lidar, 0.06, 10.0, [low])
        fused, _ = fuser.fuse(lidar, 0.06, 10.0, [low])
        self.assertTrue(np.allclose(fused[low.beams], lidar[low.beams]))

    def test_colour_alone_never_adds_an_object(self):
        # e.g. orange tape on the floor: the colour scan sees a "box base" the LiDAR cannot
        lidar = ros_stubs.corridor_ranges()
        tape = face_points(0.8, -0.1, 0.1, is_box=True)
        fuser = ScanFuser(FusionParams(persist_frames=3))
        for _ in range(5):
            fused, obstacle = fuser.fuse(lidar, 0.06, 10.0, [tape])
        self.assertTrue(np.allclose(fused, lidar))
        self.assertFalse(np.isfinite(obstacle).any())

    def test_point_behind_a_lidar_return_is_ignored(self):
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.0, -0.2, 0.2)
        fuser = ScanFuser(FusionParams(persist_frames=1))
        fused, obstacle = fuser.fuse(lidar, 0.06, 10.0, [face_points(2.0, -0.1, 0.1, False)])
        self.assertTrue(np.allclose(fused, lidar))
        self.assertFalse(np.isfinite(obstacle).any())

    def test_camera_to_lidar_transform(self):
        params = FusionParams(cam_x=0.1, cam_y=0.0, cam_yaw=math.radians(90.0))
        x, y = camera_to_lidar(np.array([1.0]), np.array([0.0]), params)
        self.assertAlmostEqual(float(x[0]), 0.1)
        self.assertAlmostEqual(float(y[0]), 1.0)
        x, _ = camera_to_lidar(np.array([1.0]), np.array([0.0]), FusionParams(), shift_x=0.05)
        self.assertAlmostEqual(float(x[0]), 0.95)

    def test_scan_to_points_drops_nothing_seen(self):
        x, y = scan_to_points([4.0, 1.0, 4.0, float('nan')], -0.1, 0.1, 4.0)
        self.assertEqual(x.size, 1)
        self.assertAlmostEqual(float(np.hypot(x[0], y[0])), 1.0)

    def test_clean_lidar_marks_invalid_as_inf(self):
        r = clean_lidar([float('nan'), 0.01, 20.0, 2.0], 0.06, 10.0)
        self.assertTrue(np.isinf(r[:3]).all())
        self.assertEqual(r[3], 2.0)


class FusionNodeTest(unittest.TestCase):
    """The node: freshness, persistence across real camera frames, and the mode topic."""

    def setUp(self):
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()

    def tearDown(self):
        ros_stubs.PARAM_OVERRIDES.clear()

    def last(self, node, topic):
        return node.publishers_by_topic[topic].messages[-1]

    def test_stale_camera_passes_lidar_through(self):
        node = fusion_node.FusionNode()
        node.depth_callback(camera_scan(0.8, -0.1, 0.1, t=0.0))
        ros_stubs.CLOCK.seconds = 1.0
        lidar = ros_stubs.corridor_ranges()
        node.lidar_callback(ros_stubs.make_scan(lidar, t=1.0))

        self.assertTrue(np.allclose(self.last(node, 'fused_scan').ranges, lidar))
        self.assertEqual(self.last(node, 'fusion/mode').data, 'lidar_only')

    def depth_run(self, **params):
        """Feed 4 depth frames of an object 0.8 m ahead that the LiDAR misses."""
        ros_stubs.PARAM_OVERRIDES['fusion_node'] = params
        node = fusion_node.FusionNode()
        ros_stubs.PARAM_OVERRIDES.clear()
        lidar = ros_stubs.corridor_ranges()                  # the LiDAR misses this object
        for i in range(4):
            t = i * 0.033
            ros_stubs.CLOCK.seconds = t
            node.depth_callback(camera_scan(0.8, -0.1, 0.1, t=t, range_max=2.5))
            node.lidar_callback(ros_stubs.make_scan(lidar, t=t))
        ahead = np.abs(ros_stubs.lidar_angles()) < 0.05
        return node, np.array(self.last(node, 'fused_scan').ranges)[ahead]

    def test_fresh_depth_frames_are_fused(self):
        node, ahead = self.depth_run(use_depth=True)
        self.assertLess(ahead.min(), 0.9)                     # the object is now in the scan
        obstacle = np.array(self.last(node, 'obstacle_scan').ranges)
        self.assertFalse(np.isfinite(obstacle).any())         # but it is not a box
        self.assertEqual(self.last(node, 'fusion/mode').data, 'fused')
        logs = [m for _, m in node.get_logger().records]
        self.assertTrue(any('depth camera adds an object at 0.8 m, 0 deg straight ahead' in m
                            for m in logs))

    def test_depth_is_off_by_default(self):
        node, ahead = self.depth_run()
        self.assertGreater(ahead.min(), 5.0)                  # LiDAR only: nothing added
        self.assertTrue(any('depth off' in m for _, m in node.get_logger().records))

    def test_fresh_box_frames_label_lidar_returns(self):
        node = fusion_node.FusionNode()
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.2, 0.2)
        for i in range(2):
            t = i * 0.033
            ros_stubs.CLOCK.seconds = t
            node.box_callback(camera_scan(1.5, -0.2, 0.2, t=t))
            node.lidar_callback(ros_stubs.make_scan(lidar, t=t))
        obstacle = np.array(self.last(node, 'obstacle_scan').ranges)
        self.assertTrue(np.allclose(self.last(node, 'fused_scan').ranges, lidar))
        self.assertGreater(int(np.isfinite(obstacle).sum()), 10)
        logs = [m for _, m in node.get_logger().records]
        self.assertTrue(any(m.startswith('box label at 1.5 m') for m in logs))

    def test_camera_disabled_ignores_camera(self):
        ros_stubs.PARAM_OVERRIDES['fusion_node'] = {'use_camera': False}
        node = fusion_node.FusionNode()
        lidar = ros_stubs.corridor_ranges()
        for i in range(4):
            node.depth_callback(camera_scan(0.8, -0.1, 0.1, t=0.0))
            node.lidar_callback(ros_stubs.make_scan(lidar, t=0.0))
        self.assertTrue(np.allclose(self.last(node, 'fused_scan').ranges, lidar))
        self.assertEqual(self.last(node, 'fusion/mode').data, 'lidar_only')


if __name__ == '__main__':
    unittest.main()
