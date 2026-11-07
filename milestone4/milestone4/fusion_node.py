"""
LiDAR and camera fusion for Milestone 4.

Our fusion_node runs once per LiDAR scan (40 Hz). The LiDAR is the primary sensor; the
camera adds box labels and objects the LiDAR plane misses. The rules live in fusion.py.

The fusion node:
    a. takes the latest /camera_box_scan (colour, boxes only) and /depth_scan (floor rejected)
    b. moves their points into the LiDAR frame, correcting for how far the car drove since
       the camera frame (odometry speed x time difference)
    c. fuses them into the LiDAR scan:
         - a box point on a LiDAR return marks that beam as a confirmed box
         - a depth point clearly nearer than the LiDAR is added after persist_frames frames
           (colour points only label: they cannot measure height, so they never add)
    d. publishes /fused_scan (LiDAR geometry, drop-in for /scan), /obstacle_scan (beams the
       colour camera confirmed as boxes, inf elsewhere) and /fusion/mode ('fused' or
       'lidar_only')
    e. logs (every 2 s at most) where a box label or a depth-only object is, so a false one
       can be traced to its source on the car

The depth scan is off by default (use_depth:=true turns it on): if the camera pitch or
height is slightly off, the floor gets past the floor rejection and appears as obstacles.
Check /depth_scan on an empty floor first (README tuning step 4).

If the camera scans stop arriving, the output is the LiDAR scan unchanged, so the car
keeps driving on the LiDAR alone.
"""

from __future__ import annotations

import math

from milestone4.fusion import (camera_to_lidar, CameraPoints, FusionParams,
                                       points_to_beams, scan_to_points, ScanFuser)
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def stamp_seconds(stamp) -> float:
    """Convert a message header stamp to seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class FusionNode(Node):
    """Fuse camera box and depth scans into the LiDAR scan."""

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

        # Association and persistence (see fusion.py):
        defaults = FusionParams()
        self.declare_parameter('assoc_tol', defaults.assoc_tol)
        self.declare_parameter('assoc_window', defaults.assoc_window)
        self.declare_parameter('persist_frames', defaults.persist_frames)

        p = self.get_parameter
        self.USE_CAMERA = bool(p('use_camera').value)
        self.USE_DEPTH = bool(p('use_depth').value)
        self.CAMERA_TIMEOUT = float(p('camera_timeout').value)
        self.MAX_SHIFT = float(p('max_shift').value)
        self.params = FusionParams(
            cam_x=float(p('cam_x').value),
            cam_y=float(p('cam_y').value),
            cam_yaw=math.radians(float(p('cam_yaw_deg').value)),
            assoc_tol=float(p('assoc_tol').value),
            assoc_window=int(p('assoc_window').value),
            persist_frames=int(p('persist_frames').value))
        self.fuser = ScanFuser(self.params)

        # What the node remembers between scans:
        self.speed = 0.0                      # forward speed from odometry (m/s)
        self.box_scan: LaserScan | None = None
        self.box_time: Time | None = None     # arrival time of the latest box scan
        self.depth_scan: LaserScan | None = None
        self.depth_time: Time | None = None
        self.used_stamps = None               # camera stamps used on the previous LiDAR scan
        self.mode = None                      # 'fused' or 'lidar_only'

        # create subscribers and publishers
        self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)
        self.create_subscription(LaserScan, 'camera_box_scan', self.box_callback, 10)
        self.create_subscription(LaserScan, 'depth_scan', self.depth_callback, 10)
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.fused_publisher = self.create_publisher(LaserScan, 'fused_scan', 10)
        self.obstacle_publisher = self.create_publisher(LaserScan, 'obstacle_scan', 10)
        self.mode_publisher = self.create_publisher(String, 'fusion/mode', 10)

        self.get_logger().info(
            f'fusion_node started: camera {"on" if self.USE_CAMERA else "off"}, '
            f'depth {"on" if self.USE_DEPTH else "off"}')

    def odom_callback(self, msg: Odometry) -> None:
        """Store the forward speed (abs: the car's odometry can read negative going forward)."""
        self.speed = abs(msg.twist.twist.linear.x)

    def box_callback(self, msg: LaserScan) -> None:
        """Store the latest camera box scan and when it arrived."""
        self.box_scan = msg
        self.box_time = self.get_clock().now()

    def depth_callback(self, msg: LaserScan) -> None:
        """Store the latest depth scan and when it arrived."""
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
        if arrival is None:
            return False
        return (self.get_clock().now() - arrival).nanoseconds * 1e-9 <= self.CAMERA_TIMEOUT

    def camera_points(self, cam_scan: LaserScan, lidar: LaserScan, is_box: bool) -> CameraPoints:
        """
        Place one camera scan's points on the LiDAR beams.

        Args:
            cam_scan (LaserScan): Camera box scan or depth scan.
            lidar (LaserScan): The LiDAR scan being fused.
            is_box (bool): True for the box scan, False for the depth scan.

        Returns:
            CameraPoints: Beam index and range of each point in the LiDAR frame.
        """
        x, y = scan_to_points(cam_scan.ranges, cam_scan.angle_min, cam_scan.angle_increment,
                              cam_scan.range_max)

        # the car drove forward between the camera frame and this LiDAR scan
        dt = stamp_seconds(lidar.header.stamp) - stamp_seconds(cam_scan.header.stamp)
        shift = min(max(self.speed * dt, 0.0), self.MAX_SHIFT)

        xl, yl = camera_to_lidar(x, y, self.params, shift)
        beams, ranges = points_to_beams(xl, yl, lidar.angle_min, lidar.angle_increment,
                                        len(lidar.ranges))
        return CameraPoints(beams=beams, ranges=ranges, is_box=is_box)

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
        Publish the fusion mode and log when it changes.

        Args:
            mode (str): 'fused' or 'lidar_only'.
        """
        if mode != self.mode:
            if mode == 'fused':
                self.get_logger().info('camera scans arriving: fusing camera with LiDAR')
            elif self.USE_CAMERA:
                self.get_logger().warn('camera scans stale or missing: driving on LiDAR only')
            else:
                self.get_logger().info('camera disabled: driving on LiDAR only')
            self.mode = mode
        self.mode_publisher.publish(String(data=mode))

    def lidar_callback(self, msg: LaserScan) -> None:
        """
        Fuse the latest camera scans into one LiDAR scan and publish the result.

        a. gathers the fresh camera scans
        b. places their points on the LiDAR beams
        c. fuses them into the scan
        d. publishes the fused scan, the obstacle scan and the mode

        Error Handling:
        if anything fails, the LiDAR scan is republished unchanged as the fused scan,
        so the planner and safety node keep a valid scan.

        Args:
            msg (LaserScan): Scan from the LiDAR.
        """
        try:
            # a. the camera scans that are recent enough to trust
            sources = []
            stamps = []
            if self.USE_CAMERA:
                if self.box_scan is not None and self.fresh(self.box_time):
                    sources.append(self.camera_points(self.box_scan, msg, is_box=True))
                    stamps.append(stamp_seconds(self.box_scan.header.stamp))
                if (self.USE_DEPTH and self.depth_scan is not None
                        and self.fresh(self.depth_time)):
                    sources.append(self.camera_points(self.depth_scan, msg, is_box=False))
                    stamps.append(stamp_seconds(self.depth_scan.header.stamp))

            # persistence only counts camera frames not used on an earlier LiDAR scan
            new_frame = bool(stamps) and stamps != self.used_stamps
            self.used_stamps = stamps

            # b./c. fuse
            fused, obstacle = self.fuser.fuse(msg.ranges, msg.range_min, msg.range_max,
                                              sources, new_frame)

            # d. publish
            self.fused_publisher.publish(self.make_scan(msg, fused))
            self.obstacle_publisher.publish(self.make_scan(msg, obstacle))
            self.set_mode('fused' if sources else 'lidar_only')

            # e. say where the camera changed the scan (each log line has its own call site)
            angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
            boxes = np.isfinite(obstacle)
            if boxes.any():
                i = int(np.argmin(np.where(boxes, obstacle, np.inf)))
                self.get_logger().info(f'box label at {where(obstacle[i], angles[i])}',
                                       throttle_duration_sec=2.0)
            added = self.fuser.trusted
            if added is not None and added.size == fused.size and added.any():
                i = int(np.argmin(np.where(added, fused, np.inf)))
                self.get_logger().info(
                    f'depth camera adds an object at {where(fused[i], angles[i])}',
                    throttle_duration_sec=2.0)

        except Exception as exc:
            self.get_logger().error(f'fusion failed, passing LiDAR through: {exc}',
                                    throttle_duration_sec=1.0)
            self.fused_publisher.publish(msg)


def where(distance: float, angle: float) -> str:
    """
    Describe where a point is, for the logs, e.g. '1.4 m, 20 deg right'.

    Args:
        distance (float): Range (m).
        angle (float): Bearing (rad, positive is left).

    Returns:
        str: Distance and side.
    """
    degrees = math.degrees(float(angle))
    side = 'straight ahead' if abs(degrees) < 2.0 else ('left' if degrees > 0.0 else 'right')
    return f'{float(distance):.1f} m, {abs(degrees):.0f} deg {side}'


def main(args=None) -> None:
    """Run the fusion node."""
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
