"""
Lap counter for Milestone 4 (camera-based).

Our lap_counter_node subscribes to the RGB images published on /camera/camera/color/image_raw.
The car starts at the start line: the first frames are averaged into a reference image of the starting view.
After that, every frame is compared with the reference; when the car is back at the start, the two look alike.

The lap counter then:
    a. counts a lap when the view matches the reference for a few frames in a row
    b. publishes the lap count on /lap_count and the similarity on /lap_counter/similarity (for tuning)
    c. after num_laps laps, requests a latched stop: negative speed on /drive_request, repeated every 0.1 s

This is received by safety_node, which stops the car and keeps it stopped.
A debug image is published on /lap_counter/debug when someone is subscribed.
"""

# include needed imports
import cv2
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32


# Lap Counting Algorithm

# [ Step 1 ] image signature
# crop the bottom rows (they show the car itself), shrink to a 64x48 thumbnail and blur
# (blurring tolerates the car being a little off its starting position)
# convert to L*a*b* (L: brightness, a: green-red, b: blue-yellow)
# subtract each channel's mean (cancels overall brightness / colour-cast changes),
# weight the colour channels by COLOR_WEIGHT and scale the vector to unit length

# [ Step 2 ] similarity
# dot product of two signatures (zero-mean cosine similarity):
# 1 for the same view, near 0 for unrelated views
# Note: a plain cosine of raw pixels would not work: all pixel values are positive,
# so almost any two frames score above 0.9.

# [ Step 3 ] count laps
# reference: average the frames of the first REFERENCE_TIME seconds into the starting view
# blind: ignore everything for BLIND_TIME seconds (the car is still near the start)
# search: once the similarity drops below REARM_THRESHOLD (the car has really left the start),
#         count a lap when it stays at or above SIMILARITY_THRESHOLD for CONFIRM_FRAMES frames,
#         then back to blind, so one pass over the start line counts once


class LapCounterNode(Node):
    """
    Lap counter (Milestone 4: recognises the starting view in the colour image).

    Counts laps by comparing each colour frame with the starting view, and requests
    a latched stop from the safety node after num_laps laps.
    """

    def __init__(self) -> None:
        """Initialize parameters, state, subscriber, publishers and the stop timer."""
        super().__init__('lap_counter_node')

        # Launch parameters (set in milestone4_py.py):
        self.declare_parameter('num_laps', 3)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('similarity_threshold', 0.8)


        # Image signature parameters:

        self.THUMB_SIZE = (64, 48)      # thumbnail (width, height) in pixels
        self.IGNORE_BOTTOM_ROWS = 30    # bottom rows dropped first (they show the car itself)
        self.BLUR_KSIZE = 5             # Gaussian blur kernel size (odd)
        self.COLOR_WEIGHT = 2.0         # weight of the a/b colour channels relative to L
                                        # (colour varies less than L; without it the scene's shape dominates)


        # Lap counting parameters:
        # (tune the thresholds from /lap_counter/similarity on one lap)

        self.NUM_LAPS = max(1, int(self.get_parameter('num_laps').value))   # laps before requesting a stop
        self.REFERENCE_TIME = 0.3       # seconds of frames averaged into the reference (short: the car may already move)
        self.BLIND_TIME = 10.0          # seconds after the reference, and after each lap, in which nothing is counted
        self.SIMILARITY_THRESHOLD = float(self.get_parameter('similarity_threshold').value)  # "back at the start"
        self.REARM_THRESHOLD = 0.5      # the similarity must drop below this before the next lap can count
        self.CONFIRM_FRAMES = 3         # frames in a row at or above the threshold to count a lap


        # What the node remembers between frames:

        self.laps = 0                   # laps counted so far
        self.done = False               # True once NUM_LAPS laps are counted
        self.phase = 'reference'        # 'reference', 'blind' or 'search'
        self.reference = None           # signature of the starting view
        self.reference_thumb = None     # averaged L*a*b* thumbnail of the starting view (for the debug image)
        self.ref_sum = None             # running sum of the reference thumbnails
        self.ref_count = 0              # number of frames in the reference
        self.ref_start = None           # time of the first reference frame (s)
        self.blind_until = 0.0          # end of the current blind period (s)
        self.armed = False              # similarity has dropped since the last lap
        self.streak = 0                 # frames in a row at or above the threshold


        # create subscriber and publishers
        self.bridge = CvBridge()

        self.image_subscription = self.create_subscription(
            Image,
            str(self.get_parameter('color_topic').value),
            self.image_callback,
            qos_profile_sensor_data)

        self.lap_publisher = self.create_publisher(Int32, 'lap_count', 10)
        self.similarity_publisher = self.create_publisher(Float32, 'lap_counter/similarity', 10)
        self.stop_publisher = self.create_publisher(
            AckermannDriveStamped,
            'drive_request',
            10)  # latched stop request to the safety node
        self.debug_publisher = self.create_publisher(Image, 'lap_counter/debug', 1)

        # keep repeating the stop request once done, in case one message is lost
        self.create_timer(0.1, self.stop_timer_callback)

        self.get_logger().info(f'lap_counter_node: stopping after {self.NUM_LAPS} laps')


    # [ Step 1 ] image signature
    def make_thumbnail(self, bgr) -> np.ndarray:
        """
        Shrink a colour frame to a small, blurred L*a*b* thumbnail.

        Args:
            bgr: BGR image from the camera

        Returns:
            thumb: float32 L*a*b* thumbnail of shape (height, width, 3)
        """

        # drop the bottom rows (they show the car itself)
        rows = max(1, bgr.shape[0] - self.IGNORE_BOTTOM_ROWS)

        # shrink and blur
        small = cv2.resize(bgr[:rows], self.THUMB_SIZE, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (self.BLUR_KSIZE, self.BLUR_KSIZE), 0)

        # L*a*b* separates brightness (L) from colour (a: green-red, b: blue-yellow)
        return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)


    def image_signature(self, thumb) -> np.ndarray:
        """
        Turn a colour thumbnail into a zero-mean, unit-length vector.

        Args:
            thumb: L*a*b* thumbnail from make_thumbnail

        Returns:
            signature: flattened float32 vector; all zeros for a blank (uniform) image
        """

        v = thumb.astype(np.float32).reshape(-1, 3)
        v = v - v.mean(axis=0)                    # per channel: brightness/colour casts cancel
        v[:, 1:] *= self.COLOR_WEIGHT             # boost the colour channels
        v = v.ravel()

        # scale to unit length (a blank image has no length)
        norm = float(np.linalg.norm(v))
        if norm < 1e-6:
            return np.zeros_like(v)
        return v / norm


    # [ Step 3 ] count laps
    def update_counter(self, thumb, t) -> tuple[bool, float | None]:
        """
        Feed one frame to the lap counter.

        Args:
            thumb: thumbnail of the frame (make_thumbnail)
            t: frame time in seconds

        Returns:
            lap_done: True if a lap was completed on this frame
            sim: similarity to the reference, or None while the reference is being built
        """

        # reference: average the first REFERENCE_TIME seconds of frames
        if self.phase == 'reference':
            if self.ref_start is None:
                self.ref_start = t
                self.ref_sum = np.zeros_like(thumb, dtype=np.float64)
            self.ref_sum += thumb
            self.ref_count += 1
            if t - self.ref_start >= self.REFERENCE_TIME:
                self.reference_thumb = (self.ref_sum / self.ref_count).astype(np.float32)
                self.reference = self.image_signature(self.reference_thumb)
                self.phase = 'blind'
                self.blind_until = t + self.BLIND_TIME
            return False, None

        # [ Step 2 ] similarity to the starting view
        sim = float(np.dot(self.image_signature(thumb), self.reference))

        # blind: nothing counts until BLIND_TIME has passed
        if self.phase == 'blind':
            if t < self.blind_until:
                return False, sim
            self.phase = 'search'
            self.armed = False
            self.streak = 0

        # search: first wait until the car has clearly left the start ...
        if not self.armed:
            self.armed = sim < self.REARM_THRESHOLD
            return False, sim

        # ... then count when the view matches the start for a few frames in a row
        self.streak = self.streak + 1 if sim >= self.SIMILARITY_THRESHOLD else 0
        if self.streak < self.CONFIRM_FRAMES:
            return False, sim

        # lap counted: back to blind, so one pass over the start line counts once
        self.laps += 1
        self.done = self.laps >= self.NUM_LAPS
        self.phase = 'blind'
        self.blind_until = t + self.BLIND_TIME
        return True, sim


    def publish_stop(self) -> None:
        """Send the latched-stop request (negative speed) to the safety node."""

        stop = AckermannDriveStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.drive.speed = -1.0          # negative speed = latched stop flag
        stop.drive.steering_angle = 0.0
        self.stop_publisher.publish(stop)


    def stop_timer_callback(self) -> None:
        """Repeat the stop request after the final lap."""
        if self.done:
            self.publish_stop()


    def publish_debug(self, bgr, sim, msg) -> None:
        """
        Publish the frame with the lap count, phase and similarity drawn on it.

        The reference thumbnail is shown in the top-right corner, and the ignored
        bottom rows are marked with a line.

        Args:
            bgr: BGR image
            sim: similarity to the reference, or None if not built yet
            msg: original Image message, for the header
        """

        out = bgr.copy()
        h, w = out.shape[:2]

        # line above the ignored bottom rows (the car itself)
        cut = h - self.IGNORE_BOTTOM_ROWS
        cv2.line(out, (0, cut), (w - 1, cut), (255, 255, 0), 1)

        # reference thumbnail in the top-right corner
        if self.reference_thumb is not None:
            lab = np.clip(self.reference_thumb, 0, 255).astype(np.uint8)
            ref = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            ref = cv2.resize(ref, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
            out[0:ref.shape[0], w - ref.shape[1]:w] = ref

        # status text: green when the view matches the start, orange otherwise
        sim_text = 'n/a' if sim is None else f'{sim:.2f}'
        text = f'lap {self.laps}/{self.NUM_LAPS}  {self.phase}  sim {sim_text}  armed {self.armed}'
        colour = (0, 255, 0) if sim is not None and sim >= self.SIMILARITY_THRESHOLD else (0, 165, 255)
        cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        debug_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_publisher.publish(debug_msg)


    # Processes incoming colour frames to count laps and request the stop.
    def image_callback(self, msg) -> None:
        """
        a. makes the frame's thumbnail
        b. updates the lap counter
        c. publishes the similarity and, after a lap, the lap count
        d. after the last lap, requests the latched stop

        Args:
            msg: Image message from the colour camera
        Returns:
            None
        """

        # nothing left to count; the timer keeps repeating the stop
        if self.done:
            return

        # a. thumbnail of the frame, and its time in seconds
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        thumb = self.make_thumbnail(bgr)
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

        # b. update the lap counter
        was_reference = self.phase == 'reference'
        lap_done, sim = self.update_counter(thumb, t)
        if was_reference and self.phase != 'reference':
            self.get_logger().info(
                f'Reference image captured ({self.ref_count} frames); '
                f'counting starts in {self.BLIND_TIME:.0f} s')

        # c. publish the similarity and the lap count
        if sim is not None:
            self.similarity_publisher.publish(Float32(data=sim))

        if lap_done:
            self.get_logger().info(f'Lap {self.laps} (similarity {sim:.2f})')
            self.lap_publisher.publish(Int32(data=self.laps))

            # d. all laps done: request the latched stop
            if self.done:
                self.get_logger().warn(f'All {self.NUM_LAPS} laps completed: requesting stop')
                self.publish_stop()

        if self.debug_publisher.get_subscription_count() > 0:
            self.publish_debug(bgr, sim, msg)


def main(args=None) -> None:
    rclpy.init(args=args)                   # initialize ROS 2
    lap_counter_node = LapCounterNode()     # creates LapCounterNode instance
    try:
        rclpy.spin(lap_counter_node)        # spin the node running + callback
    except KeyboardInterrupt:
        pass
    lap_counter_node.destroy_node()         # clears up the node
    if rclpy.ok():
        rclpy.shutdown()                    # shut down ROS2 (skip if Ctrl+C already did)


if __name__ == '__main__':
    main()