"""ROS 2 node: depth image to a LaserScan of low obstacles and boxes on ``depth_scan``.

Subscribes to the RealSense depth image and camera_info, runs depth.depth_ranges with
floor rejection, and publishes a sensor_msgs/LaserScan. The fusion node adds these
points to the LiDAR scan when they are nearer than anything the LiDAR sees and persist
for a few frames. Depth is 16-bit millimetres (0 = no reading); cv_bridge passes the
values through.
"""

from typing import Optional

from cv_bridge import CvBridge
from milestone4.depth import depth_ranges, DepthBandParams
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan


class DepthScanNode(Node):
    """Publish a LaserScan from the depth image, with the floor removed.

    Subscribes to the depth image and camera_info. Publishes on ``depth_scan``.
    Frames that arrive before camera_info are ignored.
    """

    def __init__(self) -> None:
        """Declare parameters, subscribe to depth and camera_info, and publish scans."""
        super().__init__('depth_scan_node')

        # Topics (the defaults match the RealSense wrapper on the car).
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('depth_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('scan_topic', 'depth_scan')
        self.declare_parameter('scan_frame', 'camera_depth_frame')

        # Row band, scan shape and floor rejection. Defaults live in DepthBandParams.
        d = DepthBandParams()
        self.declare_parameter('band_top', d.band_top)              # rows above centre (negative)
        self.declare_parameter('band_bottom', d.band_bottom)        # rows below centre
        self.declare_parameter('num_beams', d.num_beams)            # bearings in the scan
        self.declare_parameter('percentile', d.percentile)          # low = nearest object
        self.declare_parameter('min_valid_frac', d.min_valid_frac)  # else report nothing seen
        self.declare_parameter('range_min', d.range_min)            # metres; nearer is invalid
        self.declare_parameter('range_max', d.range_max)            # metres; also nothing seen
        self.declare_parameter('floor_reject', d.floor_reject)      # drop floor pixels
        self.declare_parameter('mount_height', d.mount_height)      # camera height (m)
        self.declare_parameter('pitch_deg', d.pitch_deg)            # positive = tilted down
        self.declare_parameter('min_height', d.min_height)          # lower = floor (m)
        self.declare_parameter('max_height', d.max_height)          # higher = ignored (m)

        # cv_bridge turns each ROS image into a NumPy array without rescaling.
        self.bridge = CvBridge()
        # Filled by info_callback. fx and cx turn a column into a bearing;
        # fy and cy place each row's ray for the floor test.
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # Depth topics use the sensor-data QoS (best effort) on the RealSense.
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter('depth_info_topic').value),
            self.info_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('depth_topic').value),
            self.depth_callback,
            qos_profile_sensor_data
        )
        self.scan_publisher = self.create_publisher(
            LaserScan,
            str(self.get_parameter('scan_topic').value),
            10
        )

    def info_callback(self, msg: CameraInfo) -> None:
        """Store fx, fy, cx and cy from the depth camera intrinsics.

        Args:
            msg (CameraInfo): Depth camera intrinsics.
        """
        # CameraInfo.k is the 3x3 intrinsic matrix, row-major: [fx 0 cx; 0 fy cy; 0 0 1].
        self.fx = float(msg.k[0])
        self.cx = float(msg.k[2])
        self.fy = float(msg.k[4])
        self.cy = float(msg.k[5])

    def band_params(self) -> DepthBandParams:
        """Read the depth-band settings used by depth_ranges.

        Returns:
            DepthBandParams: Current parameter values, read on each frame.
        """
        p = self.get_parameter
        return DepthBandParams(
            band_top=int(p('band_top').value),
            band_bottom=int(p('band_bottom').value),
            num_beams=int(p('num_beams').value),
            percentile=float(p('percentile').value),
            min_valid_frac=float(p('min_valid_frac').value),
            range_min=float(p('range_min').value),
            range_max=float(p('range_max').value),
            floor_reject=bool(p('floor_reject').value),
            mount_height=float(p('mount_height').value),
            pitch_deg=float(p('pitch_deg').value),
            min_height=float(p('min_height').value),
            max_height=float(p('max_height').value),
        )

    def depth_callback(self, msg: Image) -> None:
        """Build and publish one depth scan.

        Args:
            msg (Image): 16UC1 depth in millimetres (0 = no reading).
        """
        if self.fx is None:
            return  # intrinsics not received yet

        try:
            # passthrough keeps the 16-bit millimetre image; depth_ranges wants metres.
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_m = depth.astype(np.float32) / 1000.0
            prm = self.band_params()
            ranges, angles = depth_ranges(depth_m, self.fx, self.cx, self.cy, prm, fy=self.fy)

            # One LaserScan per depth frame. range_max means nothing was seen.
            scan = LaserScan()
            scan.header.stamp = msg.header.stamp          # same time as the image
            scan.header.frame_id = str(self.get_parameter('scan_frame').value)
            scan.angle_min = float(angles[0])
            scan.angle_max = float(angles[-1])
            scan.angle_increment = float((angles[-1] - angles[0]) / (len(angles) - 1))
            scan.range_min = prm.range_min
            scan.range_max = prm.range_max
            scan.ranges = ranges.astype(float).tolist()
            self.scan_publisher.publish(scan)

        except Exception as exc:
            # Publishing nothing makes the fusion node ignore depth until it recovers.
            self.get_logger().error(f'depth scan failed: {exc}', once=True)


def main(args=None) -> None:
    """Run the depth scan node until interrupted."""
    rclpy.init(args=args)
    node = DepthScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
