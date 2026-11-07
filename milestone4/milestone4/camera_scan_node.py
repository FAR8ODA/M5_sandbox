"""ROS 2 node: colour camera to LaserScans of the track walls and boxes.

Subscribes to the RealSense colour image and camera_info. Each frame is masked
once into walls (black ducting) and boxes (cardboard), and two scans are
published:

    camera_scan       walls and boxes together (as in Milestone 3)
    camera_box_scan   boxes only; the fusion node uses it to label LiDAR points

A debug image is published on ``camera_scan/debug`` when someone is viewing it. The
walls-and-boxes scan and the debug image are only built while someone subscribes to them.
"""

from typing import Optional

import cv2
from cv_bridge import CvBridge
from milestone4.boundary import mask_scan, ScanResult
from milestone4.camera_model import CameraModel
from milestone4.debug_draw import draw_debug
from milestone4.masks import MaskParams, split_masks
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan


class CameraScanNode(Node):
    """Publish LaserScans computed from the colour camera image."""

    def __init__(self) -> None:
        """Declare parameters and create the subscriptions and publishers."""
        super().__init__('camera_scan_node')

        # RealSense topic defaults.
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('color_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('scan_topic', 'camera_scan')
        self.declare_parameter('box_scan_topic', 'camera_box_scan')
        self.declare_parameter('scan_frame', 'camera_link')

        # Camera mounting on our car.
        self.declare_parameter('mount_height', 0.187)   # metres above the floor
        self.declare_parameter('pitch_deg', -5.3)       # positive = tilted down

        # Scan shape. range_max is also reported when nothing is seen.
        self.declare_parameter('num_beams', 180)
        self.declare_parameter('range_min', 0.05)
        self.declare_parameter('range_max', 4.0)
        self.declare_parameter('ignore_bottom_rows', 30)  # rows showing the car body
        self.declare_parameter('box_contact_rows', 4)     # a box must stand on the floor here

        # HSV thresholds for ducting and cardboard (H 0-179, S and V 0-255).
        defaults = MaskParams()
        self.declare_parameter('dark_v_max', defaults.dark_v_max)
        self.declare_parameter('use_box_color', defaults.use_box_color)
        self.declare_parameter('box_h_min', defaults.box_h_min)
        self.declare_parameter('box_h_max', defaults.box_h_max)
        self.declare_parameter('box_s_min', defaults.box_s_min)
        self.declare_parameter('box_v_min', defaults.box_v_min)
        self.declare_parameter('open_kernel', defaults.open_kernel)

        # Debug overlay is optional; drawn only when this is true and someone is watching.
        self.declare_parameter('publish_debug', True)

        # ROS images become OpenCV arrays. The camera model waits for camera_info.
        self.bridge = CvBridge()
        self.color_cam: Optional[CameraModel] = None

        # Sensor-data QoS (best effort), matching the RealSense.
        self.create_subscription(
            CameraInfo,
            self.get_parameter('color_info_topic').value,
            self.color_info_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Image,
            self.get_parameter('color_topic').value,
            self.color_callback,
            qos_profile_sensor_data
        )

        # The two scans, and the debug image beside the main one.
        scan_topic = str(self.get_parameter('scan_topic').value)
        self.scan_publisher = self.create_publisher(LaserScan, scan_topic, 10)
        self.box_scan_publisher = self.create_publisher(
            LaserScan, str(self.get_parameter('box_scan_topic').value), 10)
        self.debug_publisher = self.create_publisher(Image, f'{scan_topic}/debug', 1)

        self.get_logger().info('camera_scan_node started; waiting for camera_info')

    def color_info_callback(self, msg: CameraInfo) -> None:
        """Store the colour camera model. Rebuilt on every camera_info message.

        Args:
            msg (CameraInfo): Colour camera intrinsics.
        """
        first = self.color_cam is None
        self.color_cam = CameraModel.from_k(
            msg.k, msg.width, msg.height,
            float(self.get_parameter('mount_height').value),
            float(self.get_parameter('pitch_deg').value))
        if first:
            self.get_logger().info(f'camera_info received ({msg.width}x{msg.height})')

    def mask_params(self) -> MaskParams:
        """Read the current colour thresholds from the parameters.

        Returns:
            MaskParams: Thresholds for the colour masks.
        """
        # Read every frame, so a parameter change applies on the next frame.
        p = self.get_parameter
        return MaskParams(
            dark_v_max=int(p('dark_v_max').value),
            use_box_color=bool(p('use_box_color').value),
            box_h_min=int(p('box_h_min').value),
            box_h_max=int(p('box_h_max').value),
            box_s_min=int(p('box_s_min').value),
            box_v_min=int(p('box_v_min').value),
            open_kernel=int(p('open_kernel').value))

    def to_laserscan(self, result: ScanResult, stamp) -> LaserScan:
        """Pack a scan result into a LaserScan message.

        Args:
            result (ScanResult): Output of mask_scan.
            stamp: Header stamp to use (the image's, so the scan has the frame's time).

        Returns:
            LaserScan: Beam 0 is the right edge of the view; angles increase to the left.
        """
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = str(self.get_parameter('scan_frame').value)
        scan.angle_min = float(result.angle_min)
        scan.angle_increment = float(result.angle_increment)
        scan.angle_max = float(result.angle_min
                               + (result.ranges.size - 1) * result.angle_increment)
        scan.range_min = float(self.get_parameter('range_min').value)
        scan.range_max = float(self.get_parameter('range_max').value)  # also "nothing seen"
        scan.ranges = result.ranges.astype(float).tolist()
        return scan

    def color_callback(self, msg: Image) -> None:
        """Compute and publish both camera scans for one colour frame.

        Args:
            msg (Image): Colour image from the camera.
        """
        if self.color_cam is None:
            return  # floor projection needs the intrinsics from camera_info

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            range_max = float(self.get_parameter('range_max').value)
            num_beams = int(self.get_parameter('num_beams').value)
            bottom = int(self.get_parameter('ignore_bottom_rows').value)

            # One HSV conversion gives both masks; walls and boxes never overlap.
            wall, box = split_masks(bgr, self.mask_params())

            # The box scan is what the fusion node uses: always published.
            # a box must stand on the floor: tan furniture seen above the black ducting does
            # not (on the car it made the planner slow for boxes all round an empty track)
            contact = int(self.get_parameter('box_contact_rows').value)
            box_result = mask_scan(box, self.color_cam, num_beams, range_max, bottom,
                                   not_below=wall, contact_rows=contact)
            self.box_scan_publisher.publish(self.to_laserscan(box_result, msg.header.stamp))

            # Walls + boxes and the debug image are only for viewing and tuning, so they are
            # built only while someone subscribes; this saves CPU on the Jetson, where a busy
            # processor makes the LiDAR scans arrive late.
            debug = (self.get_parameter('publish_debug').value
                     and self.debug_publisher.get_subscription_count() > 0)
            if not debug and self.scan_publisher.get_subscription_count() == 0:
                return
            everything = cv2.bitwise_or(wall, box)
            result = mask_scan(everything, self.color_cam, num_beams, range_max, bottom)
            self.scan_publisher.publish(self.to_laserscan(result, msg.header.stamp))

            if debug:
                overlay = draw_debug(bgr, result, self.color_cam, range_max)
                debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                debug_msg.header = msg.header
                self.debug_publisher.publish(debug_msg)

        except Exception as exc:
            # Publishing nothing makes the fusion node fall back to LiDAR only.
            self.get_logger().error(f'camera scan failed: {exc}', throttle_duration_sec=2.0)


def main(args=None) -> None:
    """Run the camera scan node."""
    rclpy.init(args=args)
    node = CameraScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
