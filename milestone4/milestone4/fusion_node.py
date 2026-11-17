"""
ROS 2 node: fuse the camera box mask and depth scan into each LiDAR scan.

Runs once per /scan; the fusion rules live in fusion.py. If camera data goes stale,
the LiDAR scan passes through unchanged.
"""

from __future__ import annotations

import math

from milestone4.camera_model import CameraModel
from milestone4.fusion import (box_labels, camera_to_lidar, CameraPoints, clean_lidar,
                               column_counts, FusionParams, points_to_beams, scan_to_points,
                               ScanFuser)
from milestone4.node_utils import run_node, seconds_since, stamp_seconds
from nav_msgs.msg import Odometry
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String


class FusionNode(Node):
    """
    Fuse the camera's box mask and depth scan into the LiDAR scan.

    Subscribes to scan, camera_box_mask, color_info_topic, depth_scan and
    ego_racecar/odom. Publishes, per LiDAR scan, fused_scan (LiDAR geometry, depth
    points added), obstacle_scan (ranges of beams confirmed as boxes, inf elsewhere)
    and fusion/mode ('fused' or 'lidar_only').
    """

    def __init__(self) -> None:
        """Declare parameters and create the fuser, subscriptions and publishers."""
        super().__init__('fusion_node')

        # Main tuning parameters (values in config/milestone4.yaml):
        self.declare_parameter('use_camera', True)       # False: LiDAR passes straight through
        self.declare_parameter('use_depth', False)       # also fuse the depth scan
        self.declare_parameter('camera_timeout', 0.2)    # older camera scans are ignored (s)
        self.declare_parameter('max_shift', 0.3)         # cap on the ego-motion correction (m)

        # Camera mounting relative to the LiDAR (camera sits on top of the LiDAR):
        self.declare_parameter('cam_x', 0.0)             # ahead of the LiDAR (m)
        self.declare_parameter('cam_y', 0.0)             # left of the LiDAR (m)
        self.declare_parameter('cam_yaw_deg', 0.0)       # heading offset (degrees, left positive)
        self.declare_parameter('cam_z', 0.06)            # above the LiDAR's scan plane (m)
        self.declare_parameter('pitch_deg', -5.3)        # camera pitch (positive = tilted down)
        self.declare_parameter('color_info_topic', '/camera/camera/color/camera_info')

        # Box labels: the pixels checked around each projected LiDAR return (see fusion.py):
        self.declare_parameter('label_strip_deg', 2.0)   # strip half-height, as an angle
        self.declare_parameter('label_fraction', 0.5)    # share of the strip that must be box
        self.declare_parameter('label_max_range', 4.0)   # farther returns are not labelled (m)

        # Depth association and persistence (see fusion.py):
        defaults = FusionParams()
        self.declare_parameter('assoc_tol', defaults.assoc_tol)
        self.declare_parameter('assoc_window', defaults.assoc_window)
        self.declare_parameter('persist_frames', defaults.persist_frames)

        # read the parameters once; changes need a restart
        p = self.get_parameter
        self.USE_CAMERA = bool(p('use_camera').value)
        self.USE_DEPTH = bool(p('use_depth').value)
        self.CAMERA_TIMEOUT = float(p('camera_timeout').value)
        self.MAX_SHIFT = float(p('max_shift').value)
        self.params = FusionParams(
            cam_x=float(p('cam_x').value),
            cam_y=float(p('cam_y').value),
            cam_yaw=math.radians(float(p('cam_yaw_deg').value)),
            cam_z=float(p('cam_z').value),
            assoc_tol=float(p('assoc_tol').value),
            assoc_window=int(p('assoc_window').value),
            persist_frames=int(p('persist_frames').value),
            label_strip=math.radians(float(p('label_strip_deg').value)),
            label_fraction=float(p('label_fraction').value),
            label_max_range=float(p('label_max_range').value))
        self.PITCH_DEG = float(p('pitch_deg').value)
        self.fuser = ScanFuser(self.params)

        # What the node remembers between scans:
        self.speed = 0.0                      # forward speed from odometry (m/s)
        self.camera_info: CameraInfo | None = None   # colour intrinsics (full resolution)
        self.mask_counts: np.ndarray | None = None   # column_counts() of the latest box mask
        self.mask_cam: CameraModel | None = None     # intrinsics at the mask's resolution
        self.mask_stamp = 0.0                 # when the mask's colour frame was taken (s)
        self.mask_time: Time | None = None    # arrival time of the latest box mask
        self.depth_scan: LaserScan | None = None
        self.depth_time: Time | None = None
        self.used_stamps = None               # camera stamps used on the previous LiDAR scan

        # create subscribers and publishers
        self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)
        self.create_subscription(Image, 'camera_box_mask', self.mask_callback, 10)
        self.create_subscription(CameraInfo, str(p('color_info_topic').value),
                                 self.info_callback, qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'depth_scan', self.depth_callback, 10)
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.fused_publisher = self.create_publisher(LaserScan, 'fused_scan', 10)
        self.obstacle_publisher = self.create_publisher(LaserScan, 'obstacle_scan', 10)
        self.mode_publisher = self.create_publisher(String, 'fusion/mode', 10)

    def odom_callback(self, msg: Odometry) -> None:
        """
        Store the forward speed (abs: the car's odometry can read negative going forward).

        Args:
            msg (Odometry): Odometry of the car.
        """
        self.speed = abs(msg.twist.twist.linear.x)

    def info_callback(self, msg: CameraInfo) -> None:
        """
        Store the colour camera's intrinsics (needed to project LiDAR points).

        Args:
            msg (CameraInfo): Full-resolution colour camera intrinsics.
        """
        self.camera_info = msg

    def mask_callback(self, msg: Image) -> None:
        """
        Store the latest box mask, ready for looking up LiDAR points in it.

        The mask comes from camera_scan_node at a reduced resolution (every mask_step-th
        pixel), so the intrinsics are scaled down by the same factor.

        Args:
            msg (Image): mono8 mask, non-zero on cardboard pixels, stamped as its colour frame.
        """
        try:
            info = self.camera_info
            if info is None or info.width <= 0 or msg.width <= 0 or msg.height <= 0:
                return                        # no intrinsics yet: LiDAR only until they arrive
            # rows may be padded to msg.step bytes; keep only the first width columns
            mask = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            mask = mask.reshape(msg.height, msg.step)[:, :msg.width]
            # scale the full-resolution intrinsics down to the mask's resolution
            scale = msg.width / float(info.width)
            self.mask_cam = CameraModel(
                fx=info.k[0] * scale, fy=info.k[4] * scale, cx=info.k[2] * scale,
                cy=info.k[5] * scale, width=int(msg.width), height_px=int(msg.height),
                mount_height=0.0, pitch_rad=math.radians(self.PITCH_DEG))
            self.mask_counts = column_counts(mask)
            self.mask_stamp = stamp_seconds(msg.header.stamp)
            self.mask_time = self.get_clock().now()
        except Exception as exc:
            self.get_logger().error(f'box mask failed: {exc}', once=True)

    def depth_callback(self, msg: LaserScan) -> None:
        """
        Store the latest depth scan and when it arrived.

        Args:
            msg (LaserScan): Scan from depth_scan_node.
        """
        self.depth_scan = msg
        self.depth_time = self.get_clock().now()

    def fresh(self, arrival: Time | None) -> bool:
        """
        Check whether a camera scan arrived recently enough to use.

        Args:
            arrival: Arrival time of the scan, or None if it never arrived.

        Returns:
            bool: True if it arrived within camera_timeout.
        """
        return seconds_since(self, arrival) <= self.CAMERA_TIMEOUT

    def shift_since(self, camera_stamp: float, lidar: LaserScan) -> float:
        """
        Estimate how far the car drove between a camera frame and a LiDAR scan.

        Args:
            camera_stamp (float): When the camera frame was taken (s).
            lidar (LaserScan): The LiDAR scan being fused.

        Returns:
            float: Odometry speed x time difference, between 0 and max_shift (m).
        """
        # never negative, and capped so a bad stamp cannot move points too far
        dt = stamp_seconds(lidar.header.stamp) - camera_stamp
        return min(max(self.speed * dt, 0.0), self.MAX_SHIFT)

    def camera_points(self, cam_scan: LaserScan, lidar: LaserScan) -> CameraPoints:
        """
        Place the depth scan's points on the LiDAR beams.

        Args:
            cam_scan (LaserScan): Depth scan.
            lidar (LaserScan): The LiDAR scan being fused.

        Returns:
            CameraPoints: Beam index and range of each point in the LiDAR frame.
        """
        x, y = scan_to_points(cam_scan.ranges, cam_scan.angle_min, cam_scan.angle_increment,
                              cam_scan.range_max)

        # the car drove forward between the camera frame and this LiDAR scan
        shift = self.shift_since(stamp_seconds(cam_scan.header.stamp), lidar)

        xl, yl = camera_to_lidar(x, y, self.params, shift)
        beams, ranges = points_to_beams(xl, yl, lidar.angle_min, lidar.angle_increment,
                                        len(lidar.ranges))
        return CameraPoints(beams=beams, ranges=ranges)

    def label_boxes(self, lidar: LaserScan) -> np.ndarray | None:
        """
        Look up every LiDAR return in the latest box mask.

        Args:
            lidar (LaserScan): The LiDAR scan being fused.

        Returns:
            np.ndarray | None: Per beam, True if its return is on a cardboard box; None if
            there is no fresh mask (camera off, stale, or no camera_info yet).
        """
        if self.mask_counts is None or self.mask_cam is None or not self.fresh(self.mask_time):
            return None
        # invalid returns become inf and are never labelled
        ranges = clean_lidar(lidar.ranges, lidar.range_min, lidar.range_max)
        return box_labels(ranges, lidar.angle_min, lidar.angle_increment, self.mask_counts,
                          self.mask_cam, self.params, self.shift_since(self.mask_stamp, lidar))

    def make_scan(self, template: LaserScan, ranges) -> LaserScan:
        """
        Build a LaserScan with the LiDAR's header and geometry and new ranges.

        Args:
            template (LaserScan): The LiDAR scan.
            ranges: New range values (m).

        Returns:
            LaserScan: The new scan.
        """
        scan = LaserScan()
        scan.header = template.header
        scan.angle_min = template.angle_min
        scan.angle_max = template.angle_max
        scan.angle_increment = template.angle_increment
        scan.time_increment = template.time_increment
        scan.scan_time = template.scan_time
        scan.range_min = template.range_min
        scan.range_max = template.range_max
        scan.ranges = [float(r) for r in ranges]
        return scan

    def set_mode(self, mode: str) -> None:
        """
        Publish the fusion mode on /fusion/mode (it is not logged).

        Args:
            mode (str): 'fused' or 'lidar_only'.
        """
        self.mode_publisher.publish(String(data=mode))

    def lidar_callback(self, msg: LaserScan) -> None:
        """
        Fuse the latest camera scans into one LiDAR scan and publish the result.

        a. looks up the LiDAR returns in the fresh box mask, and gathers the fresh depth scan
        b. places the depth points on the LiDAR beams
        c. fuses both into the scan
        d. publishes the fused scan, the obstacle scan and the mode

        Error Handling:
        if anything fails, the LiDAR scan is republished unchanged as the fused scan,
        so the planner and safety node keep a valid scan.

        Args:
            msg (LaserScan): Scan from the LiDAR.
        """
        try:
            # a. the camera data that is recent enough to trust
            labels = None
            sources = []
            stamps = []
            if self.USE_CAMERA:
                labels = self.label_boxes(msg)
                if (self.USE_DEPTH and self.depth_scan is not None
                        and self.fresh(self.depth_time)):
                    sources.append(self.camera_points(self.depth_scan, msg))
                    stamps.append(stamp_seconds(self.depth_scan.header.stamp))

            # persistence only counts depth frames not used on an earlier LiDAR scan
            new_frame = bool(stamps) and stamps != self.used_stamps
            self.used_stamps = stamps

            # b./c. fuse
            fused, obstacle = self.fuser.fuse(msg.ranges, msg.range_min, msg.range_max,
                                              sources, new_frame, labels)

            # d. publish
            self.fused_publisher.publish(self.make_scan(msg, fused))
            self.obstacle_publisher.publish(self.make_scan(msg, obstacle))
            self.set_mode('fused' if sources or labels is not None else 'lidar_only')

        except Exception as exc:
            # fail safe: downstream nodes still get the unmodified LiDAR scan
            self.get_logger().error(f'fusion failed, passing LiDAR through: {exc}', once=True)
            self.fused_publisher.publish(msg)


def main() -> None:
    """Run the fusion node until Ctrl+C (console entry point)."""
    run_node(FusionNode)


if __name__ == '__main__':
    main()
