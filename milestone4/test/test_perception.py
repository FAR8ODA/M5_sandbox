"""Tests for the camera helpers: boundary scan, depth floor rejection and colour masks."""

import math
import unittest

import numpy as np
import ros_stubs

from milestone4.boundary import _bin_points, _first_hit_rows, mask_scan  # noqa: I100
from milestone4.camera_model import CameraModel, ground_points  # noqa: I100
from milestone4.depth import depth_ranges, DepthBandParams  # noqa: I100


def loop_first_hit_rows(mask, horizon_row, ignore_bottom_rows):
    """Milestone 3 version (a Python loop), to check the vectorised one against."""
    height, width = mask.shape
    row_top = max(0, int(np.floor(horizon_row)) + 1)
    row_bot = height - max(0, ignore_bottom_rows)
    hits = np.full(width, -1, dtype=np.int32)
    if row_top >= row_bot:
        return hits
    for col in range(width):
        rows = np.nonzero(mask[row_top:row_bot, col])[0]
        if len(rows) > 0:
            hits[col] = row_top + rows[-1]
    return hits


def camera(width=640, height=480, f=380.0, pitch_deg=10.0):
    """
    Return a camera model like the D435i colour camera, tilted down to see the floor.

    Args:
        width, height (int): Image size (px).
        f (float): Focal length (px).
        pitch_deg (float): Downward tilt (degrees).

    Returns:
        CameraModel: The camera.
    """
    return CameraModel(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, width=width,
                       height_px=height, mount_height=0.187, pitch_rad=math.radians(pitch_deg))


class BoundaryTest(unittest.TestCase):
    """The colour-scan geometry."""

    def test_vectorised_hit_rows_match_the_loop(self):
        """Check _first_hit_rows() matches the Milestone 3 loop."""
        rng = np.random.default_rng(1)
        mask = (rng.random((120, 160)) > 0.97).astype(np.uint8) * 255
        for horizon, bottom in ((20.5, 10), (-5.0, 0), (130.0, 0)):
            self.assertTrue(np.array_equal(_first_hit_rows(mask, horizon, bottom),
                                           loop_first_hit_rows(mask, horizon, bottom)))

    def test_bin_points_keeps_the_nearest(self):
        """Check _bin_points() keeps the nearest point per beam."""
        ranges = np.full(3, 4.0, dtype=np.float32)
        _bin_points(np.array([1.0, 2.0]), np.array([0.0, 0.0]), ranges, -0.1, 0.1)
        self.assertAlmostEqual(float(ranges[1]), 1.0)
        self.assertEqual(float(ranges[0]), 4.0)

    def test_mask_row_becomes_its_floor_distance(self):
        """Check a masked row's range equals its floor distance."""
        cam = camera()
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[300, :] = 255                                     # a wall base along one row
        result = mask_scan(mask, cam, num_beams=90, range_max=4.0)

        forward, lateral, _ = ground_points(np.array([320]), np.array([300]), cam)
        centre = int(np.argmin(np.abs(result.angle_min
                                      + np.arange(90) * result.angle_increment)))
        self.assertAlmostEqual(float(result.ranges[centre]),
                               float(np.hypot(forward[0], lateral[0])), places=2)

    def test_box_must_stand_on_the_floor(self):
        # tan furniture behind the track's black ducting has the colour of a box, but in the
        # image it stands on the ducting, not on the floor
        """Check box pixels standing on the ducting are dropped."""
        cam = camera()
        wall = np.zeros((480, 640), dtype=np.uint8)
        wall[330:360, :] = 255                                 # the ducting
        behind = np.zeros_like(wall)
        behind[300:330, 100:300] = 255                         # tan, right on top of it
        on_floor = np.zeros_like(wall)
        on_floor[380:420, 300:400] = 255                       # a box on the floor

        def seen(box, below):
            """Return how many beams of mask_scan() saw something."""
            result = mask_scan(box, cam, 90, 4.0, not_below=below, contact_rows=4)
            return int(np.sum(result.ranges < 4.0))

        self.assertGreater(seen(behind, None), 0)              # was taken for a box
        self.assertEqual(seen(behind, wall), 0)
        self.assertGreater(seen(on_floor, wall), 0)


class DepthTest(unittest.TestCase):
    """Floor rejection in the depth scan."""

    W, H, F = 640, 480, 380.0
    MOUNT, PITCH = 0.187, 10.0

    def floor_depth(self):
        """Return the depth image (m) of a flat floor seen by the camera tilted down by PITCH."""
        p = math.radians(self.PITCH)
        rows = np.arange(self.H, dtype=float)[:, None]
        drop = (rows - self.H / 2.0) / self.F * math.cos(p) + math.sin(p)   # per metre of z
        with np.errstate(divide='ignore'):
            z = np.where(drop > 1e-6, self.MOUNT / drop, 0.0)
        return np.repeat(z, self.W, axis=1)

    def params(self, floor_reject):
        """Return depth band settings for this camera, with or without floor rejection."""
        return DepthBandParams(band_top=-40, band_bottom=200, floor_reject=floor_reject,
                               mount_height=self.MOUNT, pitch_deg=self.PITCH)

    def test_floor_is_rejected(self):
        """Check an empty floor reads as nothing seen once rejected."""
        depth = self.floor_depth()
        c = self.W / 2.0
        kept, _ = depth_ranges(depth, self.F, c, self.H / 2.0, self.params(False), fy=self.F)
        rejected, _ = depth_ranges(depth, self.F, c, self.H / 2.0, self.params(True), fy=self.F)
        self.assertLess(kept.min(), 2.0)                       # Milestone 3: floor looks solid
        self.assertTrue(np.all(rejected == DepthBandParams().range_max))   # nothing seen

    def test_box_is_still_seen(self):
        """Check a box 1 m ahead is still seen with floor rejection on."""
        depth = self.floor_depth()
        p = math.radians(self.PITCH)
        rows = np.arange(self.H, dtype=float)[:, None]
        # depth and height of a vertical face 1 m ahead, kept where it is 0-0.3 m high
        yn = (rows - self.H / 2.0) / self.F
        z_box = np.repeat(1.0 / (math.cos(p) - yn * math.sin(p)), self.W, axis=1)  # 1 m ahead
        height = self.MOUNT - z_box * (yn * math.cos(p) + math.sin(p))
        face = (height >= 0.0) & (height <= 0.3)
        face[:, :280] = False
        face[:, 360:] = False                                  # a box in the middle columns
        depth = np.where(face & (z_box < np.where(depth > 0, depth, np.inf)), z_box, depth)

        ranges, angles = depth_ranges(depth, self.F, self.W / 2.0, self.H / 2.0,
                                      self.params(True), fy=self.F)
        centre = int(np.argmin(np.abs(angles)))
        self.assertAlmostEqual(float(ranges[centre]), 1.0, delta=0.05)
        self.assertEqual(float(ranges[0]), DepthBandParams().range_max)   # floor beside: no


@unittest.skipUnless(ros_stubs.HAVE_CV2, 'OpenCV not installed')
class MaskTest(unittest.TestCase):
    """Wall and box masks (need OpenCV)."""

    def test_wall_and_box_masks_are_separate(self):
        """Check ducting and cardboard go to separate, non-overlapping masks."""
        from milestone4.masks import MaskParams, obstacle_mask, split_masks
        img = np.full((100, 100, 3), 180, dtype=np.uint8)      # light grey floor
        img[60:80, :] = (20, 20, 20)                           # black ducting
        img[10:40, 30:60] = (40, 120, 200)                     # cardboard (BGR orange-brown)
        wall, box = split_masks(img, MaskParams())

        self.assertTrue(wall[70, 50] and not box[70, 50])
        self.assertTrue(box[25, 45] and not wall[25, 45])
        self.assertFalse(wall[5, 5] or box[5, 5])
        self.assertFalse(np.any((wall > 0) & (box > 0)))
        self.assertTrue(np.array_equal(obstacle_mask(img, MaskParams()) > 0,
                                       (wall > 0) | (box > 0)))


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(ros_stubs.HAVE_CV2, 'OpenCV not installed')
class CameraScanNodeTest(unittest.TestCase):
    """The box mask is always published; the scans only while someone subscribes."""

    def frame(self, node, subscribers):
        """
        Feed one synthetic colour frame (ducting and a box) to the node.

        Args:
            node (CameraScanNode): Node under test.
            subscribers (int): Subscriber count set on both scan publishers.
        """
        img = np.full((480, 640, 3), 180, dtype=np.uint8)
        img[300:340, :] = (20, 20, 20)                         # black ducting
        img[380:420, 280:360] = (40, 120, 200)                 # a cardboard box (BGR)
        node.bridge.imgmsg_to_cv2 = lambda msg, desired_encoding='bgr8': img
        node.publishers_by_topic['camera_scan'].subscribers = subscribers
        node.publishers_by_topic['camera_box_scan'].subscribers = subscribers
        node.color_callback(ros_stubs.Image())

    def test_scans_only_when_subscribed(self):
        """Check the mask is always published and the scans only with subscribers."""
        from milestone4.camera_scan_node import CameraScanNode
        info = ros_stubs.CameraInfo()
        info.k, info.width, info.height = [380.0, 0, 320.0, 0, 380.0, 240.0, 0, 0, 1], 640, 480
        node = CameraScanNode()
        node.color_info_callback(info)
        self.frame(node, subscribers=0)
        self.assertEqual(len(node.publishers_by_topic['camera_scan'].messages), 0)
        self.assertEqual(len(node.publishers_by_topic['camera_box_scan'].messages), 0)
        self.assertEqual(len(node.publishers_by_topic['camera_box_mask'].messages), 1)
        self.frame(node, subscribers=1)
        self.assertEqual(len(node.publishers_by_topic['camera_scan'].messages), 1)
        self.assertEqual(len(node.publishers_by_topic['camera_box_scan'].messages), 1)
        self.assertEqual(len(node.publishers_by_topic['camera_box_mask'].messages), 2)

    def test_box_mask_is_the_cardboard_at_half_resolution(self):
        """Check the mask is mono8 at half resolution and marks only the box."""
        from milestone4.camera_scan_node import CameraScanNode
        info = ros_stubs.CameraInfo()
        info.k, info.width, info.height = [380.0, 0, 320.0, 0, 380.0, 240.0, 0, 0, 1], 640, 480
        node = CameraScanNode()
        node.color_info_callback(info)
        self.frame(node, subscribers=0)
        msg = node.publishers_by_topic['camera_box_mask'].messages[-1]
        self.assertEqual((msg.height, msg.width, msg.step, msg.encoding), (240, 320, 320, 'mono8'))
        mask = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        self.assertTrue(np.all(mask[192:208, 142:178] == 255))  # the box (rows 380-420 / 2)
        self.assertEqual(int(mask[:185].sum()), 0)             # the ducting and floor are not
        self.assertEqual(int(mask[:, :135].sum()), 0)

    def test_box_scan_ignores_tan_above_the_ducting(self):
        """Check tan above the ducting is not in the box scan, a box on the floor is."""
        from milestone4.camera_scan_node import CameraScanNode
        info = ros_stubs.CameraInfo()
        info.k, info.width, info.height = [380.0, 0, 320.0, 0, 380.0, 240.0, 0, 0, 1], 640, 480
        node = CameraScanNode()
        node.color_info_callback(info)
        node.publishers_by_topic['camera_box_scan'].subscribers = 1   # e.g. rviz
        tan = (40, 120, 200)                                   # cardboard colour (BGR)

        def box_ranges(img):
            """Feed one image and return the published box scan ranges."""
            node.bridge.imgmsg_to_cv2 = lambda msg, desired_encoding='bgr8': img
            node.color_callback(ros_stubs.Image())
            return np.array(node.publishers_by_topic['camera_box_scan'].messages[-1].ranges)

        behind = np.full((480, 640, 3), 180, dtype=np.uint8)   # light grey floor
        behind[300:340, :] = (20, 20, 20)                      # black ducting
        behind[280:300, 100:300] = tan                         # a desk behind it
        self.assertTrue(np.all(box_ranges(behind) >= 4.0))     # nothing labelled

        on_floor = np.full((480, 640, 3), 180, dtype=np.uint8)
        on_floor[380:420, 280:360] = tan                       # a box on the track
        self.assertTrue(np.any(box_ranges(on_floor) < 4.0))
