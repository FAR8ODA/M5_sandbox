"""Tests for the LiDAR-camera fusion rules (fusion.py) and fusion_node."""

import math
import unittest

import numpy as np
import ros_stubs

from milestone4 import fusion_node  # noqa: I100  (after ros_stubs, which fakes ROS)
from milestone4.camera_model import CameraModel, ground_points, image_points  # noqa: I100
from milestone4.fusion import (box_labels, camera_to_lidar, CameraPoints,  # noqa: I100
                               clean_lidar, column_counts, FusionParams, lidar_to_camera,
                               points_to_beams, scan_to_points, ScanFuser)

CAMERA_HEIGHT = 0.187           # camera above the floor (m)
MASK_SHAPE = (240, 320)         # the box mask: half of a 640 x 480 colour image


def face_points(x, y_min, y_max, count=40):
    """
    Return depth points along a face x metres ahead, placed on the LiDAR beams.

    Args:
        x (float): Distance ahead (m).
        y_min, y_max (float): Extent of the face, left positive (m).
        count (int): Number of points.

    Returns:
        CameraPoints: Beam index and range of each point.
    """
    ys = np.linspace(y_min, y_max, count)
    beams, ranges = points_to_beams(np.full(count, x), ys, ros_stubs.LIDAR_ANGLE_MIN,
                                    ros_stubs.LIDAR_INCREMENT, ros_stubs.LIDAR_BEAMS)
    return CameraPoints(beams=beams, ranges=ranges)


def camera_scan(x, y_min, y_max, t, range_max=4.0):
    """
    Return a camera-style LaserScan (180 beams, +/-35 degrees) seeing a face x m ahead.

    Args:
        x (float): Distance to the face (m).
        y_min, y_max (float): Extent of the face, left positive (m).
        t (float): Header stamp (s).
        range_max (float): Range reported where the face is not seen (m).

    Returns:
        LaserScan: The simulated scan.
    """
    a = np.linspace(-math.radians(35.0), math.radians(35.0), 180)
    r_face = x / np.cos(a)
    y_face = r_face * np.sin(a)
    ranges = np.where((y_face >= y_min) & (y_face <= y_max), r_face, range_max)
    return ros_stubs.make_scan(ranges, angle_min=float(a[0]), angle_increment=float(a[1] - a[0]),
                               range_min=0.05, range_max=range_max, t=t)


def mask_camera(pitch_deg=-5.3):
    """
    Return the colour camera at the mask's resolution (half of 640 x 480, f = 600).

    Args:
        pitch_deg (float): Camera pitch (degrees, positive = tilted down).

    Returns:
        CameraModel: The camera model.
    """
    return CameraModel(fx=300.0, fy=300.0, cx=160.0, cy=120.0, width=MASK_SHAPE[1],
                       height_px=MASK_SHAPE[0], mount_height=CAMERA_HEIGHT,
                       pitch_rad=math.radians(pitch_deg))


def paint(mask, cam, x_near, x_far, y_min, y_max, low, high, height=CAMERA_HEIGHT):
    """
    Paint what the camera sees of a cardboard surface into a box mask.

    The surface spans y_min..y_max (left positive); its top edge is x_far ahead at height
    high above the floor and its bottom edge x_near ahead at height low. A box face has
    x_near = x_far; tape on the floor has low = high = 0. height is how high the camera
    really is above the floor. Returns the bottom edge's row.
    """
    # columns from the side edges, rows from the top and bottom edges
    u, _, _ = image_points(np.full(2, x_near), np.array([-y_max, -y_min]), 0.0, cam)
    _, v_top, _ = image_points(np.array([x_far]), np.zeros(1), height - high, cam)
    _, v_bottom, _ = image_points(np.array([x_near]), np.zeros(1), height - low, cam)
    rows = slice(max(0, int(round(v_top[0]))), max(0, int(round(v_bottom[0])) + 1))
    cols = slice(max(0, int(round(u[0]))), max(0, int(round(u[1])) + 1))
    mask[rows, cols] = 255
    return float(v_bottom[0])


def labels_for(lidar, mask, cam=None, **params):
    """
    Run box_labels on simulated LiDAR ranges and a painted mask.

    Args:
        lidar: Raw LiDAR ranges (m).
        mask (np.ndarray): Box mask at MASK_SHAPE.
        cam (CameraModel | None): Camera the node assumes; None = mask_camera().
        **params: FusionParams overrides.

    Returns:
        np.ndarray: Per-beam box labels.
    """
    return box_labels(clean_lidar(lidar, 0.06, 10.0), ros_stubs.LIDAR_ANGLE_MIN,
                      ros_stubs.LIDAR_INCREMENT, column_counts(mask), cam or mask_camera(),
                      FusionParams(**params))


class BoxLabelTest(unittest.TestCase):
    """Box labels: LiDAR returns projected into the camera's box mask."""

    def setUp(self):
        """Store the simulated LiDAR beam angles."""
        self.angles = ros_stubs.lidar_angles()

    def test_lidar_returns_on_a_box_are_labelled(self):
        """Check box-face returns are labelled, walls are not, and LiDAR ranges are kept."""
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.2, 0.2)
        mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
        paint(mask, mask_camera(), 1.5, 1.5, -0.2, 0.2, low=0.0, high=0.3)
        labels = labels_for(lidar, mask)

        on_face = (np.abs(lidar * np.cos(self.angles) - 1.5) < 1e-6) & (np.abs(self.angles) < 0.2)
        self.assertGreater(int(on_face.sum()), 40)
        self.assertGreater(labels[on_face].mean(), 0.9)       # the face is labelled ...
        self.assertFalse(labels[~on_face].any())              # ... and the walls are not

        fused, obstacle = ScanFuser(FusionParams()).fuse(lidar, 0.06, 10.0, [], labels=labels)
        self.assertTrue(np.allclose(fused, lidar))            # the LiDAR scan is unchanged
        self.assertTrue(np.allclose(obstacle[labels], lidar[labels]))   # LiDAR range kept
        self.assertFalse(np.isfinite(obstacle[~labels]).any())

    def test_a_degree_of_pitch_does_not_lose_the_label(self):
        # braking or a bump pitches the car. The labels must survive it: they dropped out
        # with the old rule (the box's floor distance had to match the LiDAR within 0.25 m)
        """Check labels survive a 1 degree pitch error that broke the old range rule."""
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 2.0, -0.2, 0.2)
        on_face = (np.abs(lidar * np.cos(self.angles) - 2.0) < 1e-6) & (np.abs(self.angles) < 0.15)
        for pitch_error in (-1.0, 1.0):
            real = mask_camera(-5.3 + pitch_error)            # how the camera really points
            mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
            base_row = paint(mask, real, 2.0, 2.0, -0.2, 0.2, low=0.0, high=0.3)
            labels = labels_for(lidar, mask)                  # the node still assumes -5.3
            self.assertGreater(labels[on_face].mean(), 0.9)
            self.assertFalse(labels[~on_face].any())

            # the old rule on the same frame: the floor distance of the box's base is
            # off by more than its 0.25 m tolerance
            forward, _, _ = ground_points(np.array([160.0]), np.array([base_row]), mask_camera())
            self.assertGreater(abs(float(forward[0]) - 2.0), 0.25)

    def test_camera_anywhere_from_15_to_22_cm_above_the_floor(self):
        # the camera sits 15 to 22 cm above the floor; cam_z (its height above the LiDAR's
        # scan plane, here 12.7 cm up) is only known roughly. With cam_z left at its default
        # the labels hold over that whole range of real heights, near and far
        """Check labels hold for camera heights of 15-22 cm with the default cam_z."""
        for height in (0.15, 0.187, 0.22):
            for distance in (0.5, 1.5, 3.0):
                lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), distance, -0.2, 0.2)
                mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
                paint(mask, mask_camera(), distance, distance, -0.2, 0.2, low=0.0, high=0.3,
                      height=height)
                labels = labels_for(lidar, mask)
                on_face = ((np.abs(lidar * np.cos(self.angles) - distance) < 1e-6)
                           & (np.abs(self.angles) < math.atan2(0.18, distance)))
                self.assertGreater(labels[on_face].mean(), 0.9, (height, distance))
                self.assertFalse(labels[np.abs(self.angles) > math.atan2(0.3, distance)].any())

    def test_tan_furniture_behind_the_ducting_is_not_a_box(self):
        # the LiDAR sees the ducting 1.5 m ahead; a tan desk 2.5 m away shows above it
        """Check tan furniture seen above the ducting is not labelled."""
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.6, 0.6)
        mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
        paint(mask, mask_camera(), 2.5, 2.5, -0.5, 0.5, low=0.35, high=0.8)
        self.assertGreater(int(mask.sum()), 0)
        self.assertFalse(labels_for(lidar, mask).any())

    def test_tape_on_the_floor_is_not_a_box(self):
        # orange tape 0.7-0.9 m ahead; the LiDAR looks over it at a wall 3 m away
        """Check orange floor tape is not labelled and adds nothing to the scan."""
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 3.0, -0.6, 0.6)
        mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
        paint(mask, mask_camera(), 0.7, 0.9, -0.1, 0.1, low=0.0, high=0.0)
        self.assertGreater(int(mask.sum()), 0)
        labels = labels_for(lidar, mask)
        self.assertFalse(labels.any())
        fused, obstacle = ScanFuser(FusionParams()).fuse(lidar, 0.06, 10.0, [], labels=labels)
        self.assertTrue(np.allclose(fused, lidar))            # colour never adds an object
        self.assertFalse(np.isfinite(obstacle).any())

    def test_far_returns_are_not_labelled(self):
        """Check returns beyond label_max_range are not labelled."""
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 3.0, -0.2, 0.2)
        mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
        paint(mask, mask_camera(), 3.0, 3.0, -0.2, 0.2, low=0.0, high=0.3)
        self.assertTrue(labels_for(lidar, mask).any())
        self.assertFalse(labels_for(lidar, mask, label_max_range=2.0).any())

    def test_label_needs_a_lidar_return(self):
        """Check a label on a beam with no LiDAR return gives no obstacle."""
        lidar = np.full(ros_stubs.LIDAR_BEAMS, np.inf)        # no returns at all
        labels = np.ones(ros_stubs.LIDAR_BEAMS, dtype=bool)
        _, obstacle = ScanFuser(FusionParams()).fuse(lidar, 0.06, 10.0, [], labels=labels)
        self.assertFalse(np.isfinite(obstacle).any())

    def test_image_points_undoes_ground_points(self):
        """Check image_points() inverts ground_points()."""
        cam = mask_camera(pitch_deg=4.0)
        us, vs = np.array([20.0, 160.0, 300.0]), np.array([150.0, 200.0, 230.0])
        forward, lateral, valid = ground_points(us, vs, cam)
        self.assertTrue(valid.all())
        u, v, ok = image_points(forward, lateral, CAMERA_HEIGHT, cam)
        self.assertTrue(ok.all())
        self.assertTrue(np.allclose(u, us) and np.allclose(v, vs))

    def test_lidar_to_camera_undoes_camera_to_lidar(self):
        """Check lidar_to_camera() inverts camera_to_lidar()."""
        params = FusionParams(cam_x=0.1, cam_y=-0.02, cam_yaw=math.radians(3.0))
        x, y = np.array([1.0, 2.5]), np.array([0.3, -0.4])
        xl, yl = camera_to_lidar(x, y, params, shift_x=0.05)
        xc, yc = lidar_to_camera(xl, yl, params, shift_x=0.05)
        self.assertTrue(np.allclose(xc, x) and np.allclose(yc, y))


class FusionRulesTest(unittest.TestCase):
    """The depth association and persistence rules, without ROS."""

    def test_depth_point_on_a_lidar_return_changes_nothing(self):
        """Check a depth point on a LiDAR return leaves the scan unchanged."""
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.2, 0.2)
        fused, obstacle = ScanFuser(FusionParams()).fuse(
            lidar, 0.06, 10.0, [face_points(1.5, -0.2, 0.2)])
        self.assertTrue(np.allclose(fused, lidar))
        self.assertFalse(np.isfinite(obstacle).any())

    def test_camera_only_point_needs_persistence(self):
        """Check a depth-only point is added after persist_frames frames, not as a box."""
        lidar = ros_stubs.corridor_ranges()                  # the LiDAR misses a low object
        low = face_points(0.8, -0.1, 0.1)
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
        """Check the same camera frame reused by later scans does not build persistence."""
        lidar = ros_stubs.corridor_ranges()
        low = face_points(0.8, -0.1, 0.1)
        fuser = ScanFuser(FusionParams(persist_frames=3))
        fuser.fuse(lidar, 0.06, 10.0, [low], new_camera_frame=True)
        for _ in range(5):                                   # LiDAR 40 Hz, same camera frame
            fused, _ = fuser.fuse(lidar, 0.06, 10.0, [low], new_camera_frame=False)
        self.assertTrue(np.allclose(fused, lidar))

    def test_missing_frame_resets_persistence(self):
        """Check a frame without the point resets its persistence."""
        lidar = ros_stubs.corridor_ranges()
        low = face_points(0.8, -0.1, 0.1)
        elsewhere = face_points(1.0, 0.4, 0.5)                # 22-27 degrees left, far from it
        fuser = ScanFuser(FusionParams(persist_frames=3))
        fuser.fuse(lidar, 0.06, 10.0, [low])
        fuser.fuse(lidar, 0.06, 10.0, [low])
        fuser.fuse(lidar, 0.06, 10.0, [elsewhere])            # the low object not seen
        fuser.fuse(lidar, 0.06, 10.0, [low])
        fused, _ = fuser.fuse(lidar, 0.06, 10.0, [low])
        self.assertTrue(np.allclose(fused[low.beams], lidar[low.beams]))

    def test_point_behind_a_lidar_return_is_ignored(self):
        """Check a depth point farther than the LiDAR return is ignored."""
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.0, -0.2, 0.2)
        fuser = ScanFuser(FusionParams(persist_frames=1))
        fused, obstacle = fuser.fuse(lidar, 0.06, 10.0, [face_points(2.0, -0.1, 0.1)])
        self.assertTrue(np.allclose(fused, lidar))
        self.assertFalse(np.isfinite(obstacle).any())

    def test_camera_to_lidar_transform(self):
        """Check the camera-to-LiDAR rotation, offset and ego-motion shift."""
        params = FusionParams(cam_x=0.1, cam_y=0.0, cam_yaw=math.radians(90.0))
        x, y = camera_to_lidar(np.array([1.0]), np.array([0.0]), params)
        self.assertAlmostEqual(float(x[0]), 0.1)
        self.assertAlmostEqual(float(y[0]), 1.0)
        x, _ = camera_to_lidar(np.array([1.0]), np.array([0.0]), FusionParams(), shift_x=0.05)
        self.assertAlmostEqual(float(x[0]), 0.95)

    def test_scan_to_points_drops_nothing_seen(self):
        """Check range_max and NaN beams are dropped by scan_to_points()."""
        x, y = scan_to_points([4.0, 1.0, 4.0, float('nan')], -0.1, 0.1, 4.0)
        self.assertEqual(x.size, 1)
        self.assertAlmostEqual(float(np.hypot(x[0], y[0])), 1.0)

    def test_clean_lidar_marks_invalid_as_inf(self):
        """Check NaN and out-of-range readings become inf."""
        r = clean_lidar([float('nan'), 0.01, 20.0, 2.0], 0.06, 10.0)
        self.assertTrue(np.isinf(r[:3]).all())
        self.assertEqual(r[3], 2.0)


class FusionNodeTest(unittest.TestCase):
    """The node: freshness, persistence across real camera frames, and the mode topic."""

    def setUp(self):
        """Reset the fake clock and parameter overrides."""
        ros_stubs.CLOCK.seconds = 0.0
        ros_stubs.PARAM_OVERRIDES.clear()

    def tearDown(self):
        """Clear the parameter overrides."""
        ros_stubs.PARAM_OVERRIDES.clear()

    def last(self, node, topic):
        """Return the last message a node published on a topic."""
        return node.publishers_by_topic[topic].messages[-1]

    def camera_info(self):
        """Return the colour camera_info: 640 x 480, twice the mask's resolution."""
        info = ros_stubs.CameraInfo()
        info.k, info.width, info.height = [600.0, 0, 320.0, 0, 600.0, 240.0, 0, 0, 1], 640, 480
        return info

    def box_mask(self, x, t):
        """Return the /camera_box_mask message of a box face x m ahead, taken at time t."""
        mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
        paint(mask, mask_camera(), x, x, -0.2, 0.2, low=0.0, high=0.3)
        # pack it as a mono8 Image, as camera_scan_node does
        msg = ros_stubs.Image()
        msg.header.stamp = ros_stubs.stamp(t)
        msg.height, msg.width = mask.shape
        msg.encoding, msg.step, msg.data = 'mono8', mask.shape[1], mask.tobytes()
        return msg

    def test_stale_camera_passes_lidar_through(self):
        """Check a stale depth scan leaves the LiDAR scan unchanged (lidar_only)."""
        node = fusion_node.FusionNode()
        node.depth_callback(camera_scan(0.8, -0.1, 0.1, t=0.0))
        ros_stubs.CLOCK.seconds = 1.0
        lidar = ros_stubs.corridor_ranges()
        node.lidar_callback(ros_stubs.make_scan(lidar, t=1.0))

        self.assertTrue(np.allclose(self.last(node, 'fused_scan').ranges, lidar))
        self.assertEqual(self.last(node, 'fusion/mode').data, 'lidar_only')

    def depth_run(self, **params):
        """
        Feed 4 depth frames of an object 0.8 m ahead that the LiDAR misses.

        Args:
            **params: fusion_node parameter overrides.

        Returns:
            tuple: (node, fused ranges of the beams within 0.05 rad of straight ahead).
        """
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
        """Check fresh depth frames add the object, not as a box, without logging."""
        node, ahead = self.depth_run(use_depth=True)
        self.assertLess(ahead.min(), 0.9)                     # the object is now in the scan
        obstacle = np.array(self.last(node, 'obstacle_scan').ranges)
        self.assertFalse(np.isfinite(obstacle).any())         # but it is not a box
        self.assertEqual(self.last(node, 'fusion/mode').data, 'fused')
        self.assertEqual(node.get_logger().records, [])       # shown in /fused_scan, not logged

    def test_depth_is_off_by_default(self):
        """Check depth is not fused unless use_depth is set."""
        node, ahead = self.depth_run()
        self.assertGreater(ahead.min(), 5.0)                  # LiDAR only: nothing added
        self.assertFalse(node.USE_DEPTH)

    def test_fresh_box_mask_labels_lidar_returns(self):
        """Check a fresh box mask labels the box on /obstacle_scan."""
        node = fusion_node.FusionNode()
        node.info_callback(self.camera_info())
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.2, 0.2)
        for i in range(2):
            t = i * 0.033
            ros_stubs.CLOCK.seconds = t
            node.mask_callback(self.box_mask(1.5, t))
            node.lidar_callback(ros_stubs.make_scan(lidar, t=t))
        obstacle = np.array(self.last(node, 'obstacle_scan').ranges)
        self.assertTrue(np.allclose(self.last(node, 'fused_scan').ranges, lidar))
        self.assertGreater(int(np.isfinite(obstacle).sum()), 40)
        self.assertAlmostEqual(float(obstacle[np.isfinite(obstacle)].min()), 1.5, delta=0.05)
        self.assertEqual(self.last(node, 'fusion/mode').data, 'fused')
        self.assertEqual(node.get_logger().records, [])       # shown in /obstacle_scan

    def test_box_mask_follows_the_car_between_frames(self):
        # the mask shows the box 1.5 m ahead; 0.1 s later the car (2 m/s) is 0.2 m closer,
        # and the LiDAR sees it at 1.3 m: still the same pixels once the motion is undone
        """Check the car's motion since the mask frame is undone when labelling."""
        node = fusion_node.FusionNode()
        node.info_callback(self.camera_info())
        node.odom_callback(ros_stubs.make_odom(2.0))
        node.mask_callback(self.box_mask(1.5, 0.0))
        ros_stubs.CLOCK.seconds = 0.1
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.3, -0.2, 0.2)
        node.lidar_callback(ros_stubs.make_scan(lidar, t=0.1))
        obstacle = np.array(self.last(node, 'obstacle_scan').ranges)
        labelled = np.isfinite(obstacle)
        self.assertGreater(int(labelled.sum()), 40)
        y = obstacle[labelled] * np.sin(ros_stubs.lidar_angles()[labelled])
        self.assertTrue(np.all(np.abs(y) <= 0.21))             # the face, not the walls

    def test_stale_box_mask_is_not_used(self):
        """Check a stale box mask labels nothing (lidar_only)."""
        node = fusion_node.FusionNode()
        node.info_callback(self.camera_info())
        node.mask_callback(self.box_mask(1.5, 0.0))
        ros_stubs.CLOCK.seconds = 1.0                         # the camera stopped
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.2, 0.2)
        node.lidar_callback(ros_stubs.make_scan(lidar, t=1.0))
        self.assertFalse(np.isfinite(self.last(node, 'obstacle_scan').ranges).any())
        self.assertEqual(self.last(node, 'fusion/mode').data, 'lidar_only')

    def test_box_mask_waits_for_camera_info(self):
        """Check a mask before camera_info is ignored."""
        node = fusion_node.FusionNode()
        node.mask_callback(self.box_mask(1.5, 0.0))           # intrinsics not known yet
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.2, 0.2)
        node.lidar_callback(ros_stubs.make_scan(lidar, t=0.0))
        self.assertFalse(np.isfinite(self.last(node, 'obstacle_scan').ranges).any())
        self.assertEqual(self.last(node, 'fusion/mode').data, 'lidar_only')

    def test_camera_disabled_ignores_camera(self):
        """Check use_camera:=false ignores the depth scan and the mask."""
        ros_stubs.PARAM_OVERRIDES['fusion_node'] = {'use_camera': False}
        node = fusion_node.FusionNode()
        node.info_callback(self.camera_info())
        lidar = ros_stubs.add_box(ros_stubs.corridor_ranges(), 1.5, -0.2, 0.2)
        for i in range(4):
            node.depth_callback(camera_scan(0.8, -0.1, 0.1, t=0.0))
            node.mask_callback(self.box_mask(1.5, 0.0))
            node.lidar_callback(ros_stubs.make_scan(lidar, t=0.0))
        self.assertTrue(np.allclose(self.last(node, 'fused_scan').ranges, lidar))
        self.assertFalse(np.isfinite(self.last(node, 'obstacle_scan').ranges).any())
        self.assertEqual(self.last(node, 'fusion/mode').data, 'lidar_only')


if __name__ == '__main__':
    unittest.main()
