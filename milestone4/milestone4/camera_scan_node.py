"""ROS 2 node: colour camera to a LaserScan of the track walls and boxes.

Subscribes to the RealSense colour image and camera_info, runs
perception.boundary.colour_scan, and publishes a LaserScan on ``camera_scan``.
The driving node remaps ``scan`` to this topic. cv_bridge converts each image
to BGR. A debug image is published on ``camera_scan/debug`` when subscribed.
"""

from typing import Optional

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from milestone4.perception.boundary import colour_scan
from milestone4.perception.camera_model import CameraModel
from milestone4.perception.debug_draw import draw_debug
from milestone4.perception.masks import MaskParams


class CameraScanNode(Node):
    """Publish a LaserScan computed from the colour camera image."""

    def __init__(self) -> None:
        """Declare parameters and create the subscriptions and publishers."""
        super().__init__('camera_scan_node')

        # RealSense topic defaults.
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('color_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('scan_topic', 'camera_scan')
        self.declare_parameter('scan_frame', 'camera_link')

        # Camera mounting on our car.
        self.declare_parameter('mount_height', 0.187)   # metres above the floor
        self.declare_parameter('pitch_deg', -5.3)       # positive = tilted down

        # Scan shape. range_max is also reported when nothing is seen.
        self.declare_parameter('num_beams', 180)
        self.declare_parameter('range_min', 0.05)
        self.declare_parameter('range_max', 4.0)
        self.declare_parameter('ignore_bottom_rows', 30)  # rows showing the car body

        # HSV thresholds for ducting and cardboard (H 0-179, S and V 0-255).
        defaults = MaskParams()
        self.declare_parameter('dark_v_max', defaults.dark_v_max)
        self.declare_parameter('use_box_color', defaults.use_box_color)
        self.declare_parameter('box_h_min', defaults.box_h_min)
        self.declare_parameter('box_h_max', defaults.box_h_max)
        self.declare_parameter('box_s_min', defaults.box_s_min)
        self.declare_parameter('box_v_min', defaults.box_v_min)
        self.declare_parameter('open_kernel', defaults.open_kernel)

        # Debug overlay is optional; drawing it only when this is true.
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

        # LaserScan for the driving node, and the debug image beside it.
        self.scan_publisher = self.create_publisher(
            LaserScan, self.get_parameter('scan_topic').value, 10)
        self.debug_publisher = self.create_publisher(
            Image, f"{self.get_parameter('scan_topic').value}/debug", 1)

        # Up, and still waiting for the first camera_info.
        self.get_logger().info('camera_scan_node started; waiting for camera_info')

    def color_info_callback(self, msg: CameraInfo) -> None:
        """Store the colour camera model. Rebuilt when parameters change.

        Args:
            msg (CameraInfo): Colour camera intrinsics.
        """
        # Intrinsics plus height and pitch, for projecting pixels onto the floor.
        self.color_cam = CameraModel.from_k(
            msg.k, msg.width, msg.height,
            float(self.get_parameter('mount_height').value),
            float(self.get_parameter('pitch_deg').value))

    def mask_params(self) -> MaskParams:
        """Read the current colour thresholds from the parameters.

        Returns:
            MaskParams: Thresholds for the colour mask.
        """

        # Live HSV thresholds, so a parameter change applies on the next frame.
        p = self.get_parameter
        return MaskParams(
            dark_v_max=int(p('dark_v_max').value),
            use_box_color=bool(p('use_box_color').value),
            box_h_min=int(p('box_h_min').value),
            box_h_max=int(p('box_h_max').value),
            box_s_min=int(p('box_s_min').value),
            box_v_min=int(p('box_v_min').value),
            open_kernel=int(p('open_kernel').value))

    def color_callback(self, msg: Image) -> None:
        """Compute and publish the camera scan for one colour frame.

        Args:
            msg (Image): Colour image from the camera.
        """
        if self.color_cam is None:
            return  # floor projection needs the intrinsics from camera_info

        # ROS image to an OpenCV BGR array, the order colour_scan expects.
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Distance written into a beam when no wall or box is seen.
        range_max = float(self.get_parameter('range_max').value)

        # Mask the walls and boxes, then keep the nearest floor hit per bearing.
        # The bottom rows are the car body, so they are left out of the search.
        result = colour_scan(
            bgr, self.color_cam, self.mask_params(),
            num_beams=int(self.get_parameter('num_beams').value),
            range_max=range_max,
            ignore_bottom_rows=int(self.get_parameter('ignore_bottom_rows').value))

        # Pack those ranges into the LaserScan the driving node subscribes to.
        scan = LaserScan()
        scan.header.stamp = msg.header.stamp          # same time as the image
        scan.header.frame_id = str(self.get_parameter('scan_frame').value)

        # Beam 0 is the right edge of the camera view; angles increase to the left.
        scan.angle_min = float(result.angle_min)
        scan.angle_increment = float(result.angle_increment)
        scan.angle_max = float(result.angle_min       # last beam, from the step above
                               + (result.ranges.size - 1) * result.angle_increment)
        scan.range_min = float(self.get_parameter('range_min').value)
        scan.range_max = range_max                    # also the "nothing seen" value
        scan.ranges = result.ranges.astype(float).tolist()
        self.scan_publisher.publish(scan)

        # Overlay only while someone is watching; drawing it every frame is costly.
        if (self.get_parameter('publish_debug').value
                and self.debug_publisher.get_subscription_count() > 0):
            # Mask, horizon, boundary dots, and a top-down view of the scan.
            overlay = draw_debug(bgr, result, self.color_cam, range_max)
            debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            debug_msg.header = msg.header            # same time as the colour frame
            self.debug_publisher.publish(debug_msg)


def main(args=None) -> None:
    """Run the camera scan node."""

    # Start ROS and spin until Ctrl-C.
    rclpy.init(args=args)
    node = CameraScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Shut the node down even if spin was interrupted.
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()