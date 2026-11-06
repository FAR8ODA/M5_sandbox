"""ROS 2 node: depth image to a LaserScan on ``depth_scan``, for AEB.

Subscribes to the RealSense depth image and camera_info, runs
perception.depth.depth_ranges, and publishes a sensor_msgs/LaserScan for
the safety node's time-to-collision check. Depth is 16-bit millimetres
(0 = no reading); cv_bridge passes the values through.
"""

from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from milestone4.perception.depth import DepthBandParams, depth_ranges


class DepthScanNode(Node):
    """Publish a LaserScan from the depth image for the safety node's AEB.

    Subscribes to the depth image and camera_info. Publishes on
    ``depth_scan``. Frames that arrive before camera_info are ignored.
    """

    def __init__(self) -> None:
        """Declare parameters, subscribe to depth and camera_info, and publish scans.

        Side effects:
            Creates the depth-image and camera_info subscriptions and the
            LaserScan publisher.
        """
        super().__init__('depth_scan_node')

        # Topics (the defaults match the RealSense wrapper on the car).
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('depth_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('scan_topic', 'depth_scan')
        self.declare_parameter('scan_frame', 'camera_depth_frame')

        # Row band and scan shape. Defaults live in DepthBandParams: a band
        # around the image centre, 90 beams, the 10th percentile, 0.15-3.0 m.
        d = DepthBandParams()
        self.declare_parameter('band_top', d.band_top)              # rows above centre (negative)
        self.declare_parameter('band_bottom', d.band_bottom)        # rows below centre
        self.declare_parameter('num_beams', d.num_beams)            # bearings in the scan
        self.declare_parameter('percentile', d.percentile)          # low = nearest object
        self.declare_parameter('min_valid_frac', d.min_valid_frac)  # else report nothing seen
        self.declare_parameter('range_min', d.range_min)            # metres; nearer is invalid
        self.declare_parameter('range_max', d.range_max)            # metres; also nothing seen

        # cv_bridge turns each ROS image into a NumPy array without rescaling.
        self.bridge = CvBridge()
        # Filled by info_callback. fx and cx turn a column into a bearing;
        # cy places the row band on the image centre.
        self.fx: Optional[float] = None
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

        self.get_logger().info('depth_scan_node started; waiting for camera_info')

    def info_callback(self, msg: CameraInfo) -> None:
        """Store fx, cx and cy from the depth camera intrinsics.

        Args:
            msg (CameraInfo): Depth camera intrinsics.

        Side effects:
            depth_callback publishes only after this runs.
        """
        # CameraInfo.k is the 3x3 intrinsic matrix, row-major: fx, cx, then cy.
        self.fx = float(msg.k[0])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def band_params(self) -> DepthBandParams:
        """Read the depth-band settings used by depth_ranges.

        Returns:
            DepthBandParams: Current parameter values, read on each frame.
        """
        # Cast each ROS parameter into the field type DepthBandParams expects.
        p = self.get_parameter
        return DepthBandParams(
            band_top=int(p('band_top').value), 
            band_bottom=int(p('band_bottom').value),
            num_beams=int(p('num_beams').value), 
            percentile=float(p('percentile').value),
            min_valid_frac=float(p('min_valid_frac').value),
            range_min=float(p('range_min').value), 
            range_max=float(p('range_max').value)
            )

    def depth_callback(self, msg: Image) -> None:
        """Build and publish one depth scan.

        Args:
            msg (Image): 16UC1 depth in millimetres (0 = no reading).

        Side effects:
            Publishes one LaserScan once camera_info has arrived.
        """
        if self.fx is None:
            return  # intrinsics not received yet

        # passthrough keeps the 16-bit millimetre image; depth_ranges wants metres.
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth_m = depth.astype(np.float32) / 1000.0
        prm = self.band_params()
        # Nearest valid depth per bearing. angles increase to the left.
        ranges, angles = depth_ranges(depth_m, self.fx, self.cx, self.cy, prm)

        # One LaserScan per depth frame. range_max means nothing was seen.
        scan = LaserScan()
        scan.header.stamp = msg.header.stamp          # same time as the image
        scan.header.frame_id = str(self.get_parameter('scan_frame').value)
        # angles increase to the left, beam 0 on the right, as in a ROS LaserScan.
        scan.angle_min = float(angles[0])
        scan.angle_max = float(angles[-1])
        scan.angle_increment = float((angles[-1] - angles[0]) / (len(angles) - 1))
        scan.range_min = prm.range_min
        scan.range_max = prm.range_max                # also the value for "nothing seen"
        scan.ranges = ranges.astype(float).tolist()
        self.scan_publisher.publish(scan)


def main(args=None) -> None:
    """Run the depth scan node until interrupted.

    Args:
        args: Arguments for rclpy.init, or None for the process arguments.

    Side effects:
        Initializes ROS, spins the node, then shuts ROS down.
    """
    rclpy.init(args=args)
    node = DepthScanNode()
    try:
        rclpy.spin(node)  # block until Ctrl-C
    except KeyboardInterrupt:
        pass
    # Tear the node down even when spin was interrupted.
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()