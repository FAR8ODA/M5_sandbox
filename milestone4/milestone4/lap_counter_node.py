"""
Lap counter for Milestone 4 (camera + LiDAR + odometry).

The car starts at the start line. The first frames from the colour camera and the first
LiDAR scans are averaged into a reference "view" of the start. After that, every new
frame and scan is compared with the reference; when the car is back at the start, they
look alike.

Milestone 3 used the camera only and was temperamental (lighting, small heading
changes). Milestone 4 fuses three sources:
    camera:   zero-mean cosine similarity of a small L*a*b* thumbnail (as in Milestone 3)
    LiDAR:    zero-mean cosine similarity of the 270 degree range profile; it does not
              depend on lighting and sees the whole room shape around the start
    odometry: the distance driven since the last lap; a lap only counts after
              min_lap_distance, and after lap 1 after lap_distance_fraction of lap 1
              (straights look alike to the LiDAR; the gate stops look-alike matches)

The lap counter then:
    a. combines the camera and LiDAR similarities into one score (weights w_cam, w_lidar);
       if one source is missing or stale, the other is used alone
    b. confirms a match when the score stays above the threshold for a few ticks in a row
       and the distance gate is open, then counts the lap at the score's peak
    c. publishes /lap_count, the scores on /lap_counter/similarity (and _cam / _lidar)
    d. after num_laps laps, requests a latched stop: negative speed on /drive_request,
       repeated every 0.1 s

This is received by safety_node, which stops the car and keeps it stopped.
A debug image is published on /lap_counter/debug when someone is subscribed.
"""

# include needed imports
from __future__ import annotations

from ackermann_msgs.msg import AckermannDriveStamped
import cv2
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32, Int32


# Lap Counting Algorithm

# [ Step 1 ] signatures
# camera: crop the car body, shrink to 64x48, blur, convert to L*a*b*, subtract each
#         channel's mean, weight the colour channels and scale to unit length (Milestone 3)
# LiDAR:  clip the ranges at LIDAR_CLIP, average them into LIDAR_BINS bins,
#         subtract the mean and scale to unit length

# [ Step 2 ] similarity
# dot product of two signatures (zero-mean cosine similarity): 1 for the same view
# score = (w_cam * sim_cam + w_lidar * sim_lidar) / (w_cam + w_lidar)

# [ Step 3 ] count laps
# reference: average the first REFERENCE_TIME seconds into the starting view
# blind: ignore everything for BLIND_TIME seconds
# search: once the score drops below REARM_THRESHOLD (the car has left the start),
#         confirm a match when it stays at or above SIMILARITY_THRESHOLD for CONFIRM_TICKS
#         ticks AND the car has driven far enough since the last lap, then count the lap
#         at the score's peak (where the view matches best); then back to blind


def unit_signature(v: np.ndarray) -> np.ndarray:
    """
    Subtract the mean and scale a vector to unit length.

    Args:
        v: flattened float vector

    Returns:
        signature: zero-mean unit vector, all zeros for a constant vector
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v - v.mean()
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return np.zeros_like(v)
    return v / norm


class LapCounterNode(Node):
    """
    Lap counter that recognises the start from the camera and the LiDAR.

    Requests a latched stop from the safety node after num_laps laps.
    """

    def __init__(self) -> None:
        """Initialize parameters, state, subscribers, publishers and timers."""
        super().__init__('lap_counter_node')

        # Launch parameters (set in milestone4_py.py):
        self.declare_parameter('num_laps', 3)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('use_camera', True)
        self.declare_parameter('similarity_threshold', 0.8)
        self.declare_parameter('rearm_threshold', 0.5)
        self.declare_parameter('camera_weight', 0.5)
        self.declare_parameter('lidar_weight', 0.5)
        self.declare_parameter('min_lap_distance', 5.0)
        self.declare_parameter('lap_distance_fraction', 0.85)
        self.declare_parameter('blind_time', 5.0)
        self.declare_parameter('lidar_shift_deg', 10.0)

        p = self.get_parameter

        # Image signature parameters:

        self.THUMB_SIZE = (64, 48)      # thumbnail (width, height) in pixels
        self.IGNORE_BOTTOM_ROWS = 30    # bottom rows dropped first (they show the car itself)
        self.BLUR_KSIZE = 5             # Gaussian blur kernel size (odd)
        self.COLOR_WEIGHT = 2.0         # weight of the a/b colour channels relative to L

        # LiDAR signature parameters:

        self.LIDAR_CLIP = 5.0           # ranges are clipped here; far walls add little (m)
        self.LIDAR_BINS = 108           # 1080 beams averaged into 108 bins of 2.5 degrees
        self.LIDAR_SHIFT = max(0, int(round(float(p('lidar_shift_deg').value) / 2.5)))  # bins

        # Lap counting parameters:
        # (tune the thresholds from /lap_counter/similarity on one lap)

        self.NUM_LAPS = max(1, int(p('num_laps').value))          # laps before requesting a stop
        self.USE_CAMERA = bool(p('use_camera').value)             # False: LiDAR signature only
        self.SIMILARITY_THRESHOLD = float(p('similarity_threshold').value)  # "back at the start"
        self.REARM_THRESHOLD = float(p('rearm_threshold').value)  # must drop below this first
        self.CAMERA_WEIGHT = max(0.0, float(p('camera_weight').value))
        self.LIDAR_WEIGHT = max(0.0, float(p('lidar_weight').value))
        self.MIN_LAP_DISTANCE = float(p('min_lap_distance').value)  # shortest possible lap (m)
        self.LAP_DISTANCE_FRACTION = float(p('lap_distance_fraction').value)  # x lap 1 distance
        self.BLIND_TIME = float(p('blind_time').value)  # after the start and each lap (s)
        self.REFERENCE_TIME = 0.3       # seconds averaged into the reference
        self.CONFIRM_TICKS = 3          # ticks in a row at or above the threshold to confirm
        self.PEAK_DROP = 0.03           # score this far below its peak: the start was passed
        self.PEAK_TIMEOUT = 1.0         # count anyway if the score stays at its peak this long (s)
        self.TICK_PERIOD = 0.05         # how often the score is evaluated (s)
        self.INPUT_TIMEOUT = 0.3        # a camera frame or scan older than this is not used (s)
        self.LATE_REFERENCE_TIME = 3.0  # a late camera reference is allowed this long (s) ...
        self.LATE_REFERENCE_MOVE = 0.3  # ... while the car is still this close to the start (m)
        self.LATE_REFERENCE_FRAMES = 5  # frames averaged into a late camera reference

        # What the node remembers:

        self.laps = 0                   # laps counted so far
        self.done = False               # True once NUM_LAPS laps are counted
        self.phase = 'reference'        # 'reference', 'blind' or 'search'
        self.ref_start = None           # time of the first reference sample (s)
        self.ref_distance = 0.0         # odometry distance when the reference started (m)
        self.camera_warned = False      # warned that the camera missed the reference
        self.cam_sum = None             # running sum of the reference thumbnails
        self.cam_count = 0
        self.lidar_sum = None           # running sum of the reference LiDAR profiles
        self.lidar_count = 0
        self.cam_reference = None       # signature of the starting view (camera)
        self.reference_thumb = None     # averaged thumbnail of the starting view (debug image)
        self.lidar_reference = None     # signature of the starting view (LiDAR)
        self.thumb = None               # latest thumbnail
        self.thumb_time = None          # when it arrived (s)
        self.profile = None             # latest LiDAR profile
        self.profile_time = None        # when it arrived (s)
        self.blind_until = 0.0          # end of the current blind period (s)
        self.armed = False              # score has dropped since the last lap
        self.streak = 0                 # ticks in a row at or above the threshold
        self.peak = None                # best score since the match was confirmed
        self.peak_since = 0.0           # when the match was confirmed (s)
        self.scores = (None, None, None)  # (combined, camera, lidar) of the last tick

        # Odometry distance gate:

        self.distance = 0.0             # distance driven since the node started (m)
        self.lap_start_distance = 0.0   # distance at the start of this lap (m)
        self.first_lap_distance = None  # length of lap 1 (m)
        self.last_lap_length = 0.0      # length of the last counted lap (m)
        self.last_odom_time = None      # time of the previous odometry message (s)
        self.odom_warned = False        # warned that there is no odometry

        # create subscribers and publishers
        self.bridge = CvBridge()
        if self.USE_CAMERA:
            self.create_subscription(Image, str(p('color_topic').value), self.image_callback,
                                     qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)
        self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.lap_publisher = self.create_publisher(Int32, 'lap_count', 10)
        self.similarity_publisher = self.create_publisher(Float32, 'lap_counter/similarity', 10)
        self.cam_similarity_publisher = self.create_publisher(
            Float32, 'lap_counter/similarity_cam', 10)
        self.lidar_similarity_publisher = self.create_publisher(
            Float32, 'lap_counter/similarity_lidar', 10)
        self.stop_publisher = self.create_publisher(
            AckermannDriveStamped,
            'drive_request',
            10)  # latched stop request to the safety node
        self.debug_publisher = self.create_publisher(Image, 'lap_counter/debug', 1)

        # evaluate the score at a fixed rate, and keep repeating the stop once done
        self.create_timer(self.TICK_PERIOD, self.tick)
        self.create_timer(0.1, self.stop_timer_callback)

        self.get_logger().info(
            f'lap_counter_node: stopping after {self.NUM_LAPS} laps '
            f'(camera {"on" if self.USE_CAMERA else "off"}, LiDAR on, '
            f'min lap distance {self.MIN_LAP_DISTANCE} m)')

    def now_seconds(self) -> float:
        """Return the node clock time in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    # [ Step 1 ] signatures

    def make_thumbnail(self, bgr: np.ndarray) -> np.ndarray:
        """
        Shrink a colour frame to a small, blurred L*a*b* thumbnail (Milestone 3).

        Args:
            bgr: BGR image from the camera

        Returns:
            thumb: float32 L*a*b* thumbnail of shape (height, width, 3)
        """
        rows = max(1, bgr.shape[0] - self.IGNORE_BOTTOM_ROWS)   # drop the car body
        small = cv2.resize(bgr[:rows], self.THUMB_SIZE, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (self.BLUR_KSIZE, self.BLUR_KSIZE), 0)
        return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    def image_signature(self, thumb: np.ndarray) -> np.ndarray:
        """
        Turn a thumbnail into a zero-mean (per channel), unit-length vector (Milestone 3).

        Args:
            thumb: L*a*b* thumbnail from make_thumbnail

        Returns:
            signature: flattened vector; all zeros for a blank (uniform) image
        """
        v = thumb.astype(np.float64).reshape(-1, 3)
        v = v - v.mean(axis=0)                    # per channel: brightness/colour casts cancel
        v[:, 1:] *= self.COLOR_WEIGHT             # boost the colour channels
        v = v.ravel()
        norm = float(np.linalg.norm(v))
        return np.zeros_like(v) if norm < 1e-6 else v / norm

    def lidar_profile(self, scan: LaserScan) -> np.ndarray:
        """
        Turn a LiDAR scan into a short range profile.

        Args:
            scan: LaserScan from the LiDAR

        Returns:
            profile: LIDAR_BINS averaged ranges, clipped at LIDAR_CLIP (m)
        """
        r = np.asarray(scan.ranges, dtype=np.float64)
        with np.errstate(invalid='ignore'):
            r = np.where(np.isfinite(r) & (r >= scan.range_min), r, self.LIDAR_CLIP)
        r = np.minimum(r, self.LIDAR_CLIP)
        bins = min(self.LIDAR_BINS, r.size)
        return np.array([chunk.mean() for chunk in np.array_split(r, bins)])

    def lidar_similarity(self, profile: np.ndarray) -> float:
        """
        Compare a LiDAR profile with the reference, allowing a small heading difference.

        The car rarely comes back to the start at exactly its starting heading, and a few
        degrees of rotation shift the whole profile. The best match over shifts of up to
        LIDAR_SHIFT bins (lidar_shift_deg) is used. Look-alike places that this also lets
        through are stopped by the odometry distance gate.

        Args:
            profile: from lidar_profile

        Returns:
            similarity: best zero-mean cosine similarity, -1 to 1
        """
        sig = self.lidar_signature(profile)
        if sig.size != self.lidar_reference.size:
            return 0.0
        return max(float(np.dot(np.roll(sig, k), self.lidar_reference))
                   for k in range(-self.LIDAR_SHIFT, self.LIDAR_SHIFT + 1))

    def lidar_signature(self, profile: np.ndarray) -> np.ndarray:
        """
        Turn a LiDAR profile into a zero-mean, unit-length vector.

        Args:
            profile: from lidar_profile

        Returns:
            signature: unit vector (zeros if the profile is flat)
        """
        return unit_signature(profile)

    # Inputs

    def image_callback(self, msg: Image) -> None:
        """
        Make the thumbnail of a colour frame (and add it to the reference at the start).

        Args:
            msg: Image message from the colour camera
        """
        if self.done:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.thumb = self.make_thumbnail(bgr)
            self.thumb_time = self.now_seconds()
            if self.cam_reference is None and self.camera_reference_open(self.thumb_time):
                self.cam_sum = self.thumb.astype(np.float64) + (
                    self.cam_sum if self.cam_sum is not None else 0.0)
                self.cam_count += 1
            if self.debug_publisher.get_subscription_count() > 0:
                self.publish_debug(bgr, msg)
        except Exception as exc:
            self.get_logger().error(f'lap counter image failed: {exc}', throttle_duration_sec=2.0)

    def lidar_callback(self, msg: LaserScan) -> None:
        """
        Make the profile of a LiDAR scan (and add it to the reference at the start).

        Args:
            msg: LaserScan from the LiDAR
        """
        if self.done or len(msg.ranges) == 0:
            return
        try:
            profile = self.lidar_profile(msg)
            if self.profile is not None and self.profile.size != profile.size:
                return                           # a different scan geometry; ignore it
            self.profile = profile
            self.profile_time = self.now_seconds()
            if self.phase == 'reference':
                self.lidar_sum = profile + (self.lidar_sum if self.lidar_sum is not None
                                            else 0.0)
                self.lidar_count += 1
        except Exception as exc:
            self.get_logger().error(f'lap counter scan failed: {exc}', throttle_duration_sec=2.0)

    def odom_callback(self, msg: Odometry) -> None:
        """
        Add the distance driven since the previous odometry message.

        Args:
            msg: Odometry message
        """
        t = self.now_seconds()
        if self.last_odom_time is not None:
            dt = min(max(t - self.last_odom_time, 0.0), 0.2)   # ignore long gaps
            self.distance += abs(msg.twist.twist.linear.x) * dt
        self.last_odom_time = t

    # [ Step 2 ] similarity

    def similarity(self, now: float) -> tuple[float | None, float | None, float | None]:
        """
        Compare the latest frame and scan with the reference.

        Args:
            now: current time (s)

        Returns:
            (score, sim_cam, sim_lidar): combined score and each source's similarity;
            None for a source that is missing or stale (score None if both are)
        """
        sim_cam = sim_lidar = None
        if (self.cam_reference is not None and self.thumb is not None
                and now - self.thumb_time <= self.INPUT_TIMEOUT):
            sim_cam = float(np.dot(self.image_signature(self.thumb), self.cam_reference))
        if (self.lidar_reference is not None and self.profile is not None
                and now - self.profile_time <= self.INPUT_TIMEOUT):
            sim_lidar = self.lidar_similarity(self.profile)

        parts = [(w, s) for w, s in ((self.CAMERA_WEIGHT, sim_cam), (self.LIDAR_WEIGHT, sim_lidar))
                 if s is not None and w > 0.0]
        if not parts:
            return None, sim_cam, sim_lidar
        total = sum(w for w, _ in parts)
        return sum(w * s for w, s in parts) / total, sim_cam, sim_lidar

    # [ Step 3 ] count laps

    def lap_gate(self) -> float:
        """Return the distance the car must drive before the next lap can count (m)."""
        if self.first_lap_distance is None:
            return self.MIN_LAP_DISTANCE
        return max(self.MIN_LAP_DISTANCE, self.LAP_DISTANCE_FRACTION * self.first_lap_distance)

    def distance_ok(self) -> bool:
        """
        Check the odometry distance gate.

        Returns:
            bool: True if the car has driven far enough since the last lap,
            or if there is no odometry at all (the gate is then skipped).
        """
        if self.last_odom_time is None:
            if not self.odom_warned:
                self.get_logger().warn('no odometry: lap distance gate disabled')
                self.odom_warned = True
            return True
        return self.distance - self.lap_start_distance >= self.lap_gate()

    def camera_reference_open(self, now: float) -> bool:
        """
        Check whether camera frames may still be added to the reference.

        The RealSense can start a little after the LiDAR. Its frames still count towards the
        reference for a short time, as long as the car has not yet left the start.

        Args:
            now: current time (s)

        Returns:
            bool: True during the reference phase, or shortly after it near the start.
        """
        if self.phase == 'reference':
            return True
        if self.ref_start is None:
            return False
        return (now - self.ref_start <= self.LATE_REFERENCE_TIME
                and self.distance - self.ref_distance <= self.LATE_REFERENCE_MOVE)

    def late_camera_reference(self, now: float) -> None:
        """
        Build the camera reference from late frames, or warn once that it was missed.

        Args:
            now: current time (s)
        """
        if self.cam_count >= self.LATE_REFERENCE_FRAMES:
            self.reference_thumb = (self.cam_sum / self.cam_count).astype(np.float32)
            self.cam_reference = self.image_signature(self.reference_thumb)
            self.get_logger().info(f'camera reference captured late ({self.cam_count} frames)')
        elif not self.camera_reference_open(now) and not self.camera_warned:
            self.get_logger().warn('camera not ready at the start: laps are counted on the '
                                   'LiDAR only')
            self.camera_warned = True

    def finish_reference(self, now: float) -> None:
        """
        Build the reference signatures from the frames and scans averaged so far.

        Args:
            now: current time (s)
        """
        if self.cam_count > 0:
            self.reference_thumb = (self.cam_sum / self.cam_count).astype(np.float32)
            self.cam_reference = self.image_signature(self.reference_thumb)
        if self.lidar_count > 0:
            self.lidar_reference = self.lidar_signature(self.lidar_sum / self.lidar_count)
        self.phase = 'blind'
        self.blind_until = now + self.BLIND_TIME
        self.get_logger().info(
            f'reference captured ({self.cam_count} frames, {self.lidar_count} scans); '
            f'counting starts in {self.BLIND_TIME:.0f} s')

    def update_counter(self, score: float, now: float) -> bool:
        """
        Feed one score to the lap counter.

        Args:
            score: combined similarity to the reference
            now: current time (s)

        Returns:
            lap_done: True if a lap was completed on this tick
        """
        # blind: nothing counts until BLIND_TIME has passed
        if self.phase == 'blind':
            if now < self.blind_until:
                return False
            self.phase = 'search'
            self.armed = False
            self.streak = 0

        # search: first wait until the car has clearly left the start ...
        if not self.armed:
            self.armed = score < self.REARM_THRESHOLD
            return False

        # ... then wait until the view matches the start for a few ticks in a row
        if self.peak is None:
            self.streak = self.streak + 1 if score >= self.SIMILARITY_THRESHOLD else 0
            if self.streak < self.CONFIRM_TICKS:
                return False

            # a match before the car has driven far enough is a look-alike spot, not the start
            if not self.distance_ok():
                self.get_logger().info(
                    f'start-like view after {self.distance - self.lap_start_distance:.1f} m '
                    f'(< {self.lap_gate():.1f} m): not a lap', throttle_duration_sec=2.0)
                self.armed = False
                self.streak = 0
                return False

            # confirmed: now follow the score up to its peak
            self.peak, self.peak_since = score, now
            return False

        # count at the peak (the best match is where the car really is at the start line);
        # a straight looks alike all along, so a plain threshold would count too early
        if score > self.peak:
            self.peak = score
            return False
        still_at_top = score >= max(self.peak - self.PEAK_DROP, self.SIMILARITY_THRESHOLD)
        if still_at_top and now - self.peak_since < self.PEAK_TIMEOUT:
            return False

        # lap counted: back to blind, so one pass over the start counts once
        self.peak = None
        self.laps += 1
        self.last_lap_length = self.distance - self.lap_start_distance
        if self.first_lap_distance is None and self.last_odom_time is not None:
            self.first_lap_distance = self.last_lap_length
        self.lap_start_distance = self.distance
        self.done = self.laps >= self.NUM_LAPS
        self.phase = 'blind'
        self.blind_until = now + self.BLIND_TIME
        return True

    def tick(self) -> None:
        """
        Evaluate the score and update the lap count.

        a. builds the reference once REFERENCE_TIME has passed
        b. combines the camera and LiDAR similarities
        c. updates the counter and publishes the scores and the lap count
        d. after the last lap, requests the latched stop
        """
        if self.done:
            return
        try:
            self.evaluate(self.now_seconds())
        except Exception as exc:
            self.get_logger().error(f'lap counter failed: {exc}', throttle_duration_sec=2.0)

    def evaluate(self, now: float) -> None:
        """
        Run one tick of the lap counter (see tick).

        Args:
            now: current time (s)
        """
        # a. reference phase: wait for the first samples, then average REFERENCE_TIME of them
        if self.phase == 'reference':
            if self.cam_count == 0 and self.lidar_count == 0:
                return
            if self.ref_start is None:
                self.ref_start = now
                self.ref_distance = self.distance
            if now - self.ref_start >= self.REFERENCE_TIME:
                self.finish_reference(now)
            return
        if self.USE_CAMERA and self.cam_reference is None:
            self.late_camera_reference(now)

        # b. similarity to the starting view
        score, sim_cam, sim_lidar = self.similarity(now)
        self.scores = (score, sim_cam, sim_lidar)
        if score is None:
            return
        self.similarity_publisher.publish(Float32(data=float(score)))
        if sim_cam is not None:
            self.cam_similarity_publisher.publish(Float32(data=sim_cam))
        if sim_lidar is not None:
            self.lidar_similarity_publisher.publish(Float32(data=sim_lidar))

        # c. update the counter
        if self.update_counter(score, now):
            self.get_logger().info(
                f'Lap {self.laps} (score {score:.2f}, {self.last_lap_length:.1f} m driven)')
            self.lap_publisher.publish(Int32(data=self.laps))

            # d. all laps done: request the latched stop
            if self.done:
                self.get_logger().warn(f'All {self.NUM_LAPS} laps completed: requesting stop')
                self.publish_stop()

    # Outputs

    def publish_stop(self) -> None:
        """Send the latched-stop request (negative speed) to the safety node."""
        stop = AckermannDriveStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.drive.speed = -1.0          # negative speed = latched stop flag
        stop.drive.steering_angle = 0.0
        self.stop_publisher.publish(stop)

    def stop_timer_callback(self) -> None:
        """Repeat the stop request after the final lap, in case one message is lost."""
        if self.done:
            self.publish_stop()

    def publish_debug(self, bgr: np.ndarray, msg: Image) -> None:
        """
        Publish the frame with the lap count, phase and scores drawn on it.

        Args:
            bgr: BGR image
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

        # status text: green when the score matches the start, orange otherwise
        score, sim_cam, sim_lidar = self.scores

        def fmt(x: float | None) -> str:
            return 'n/a' if x is None else f'{x:.2f}'

        line1 = f'lap {self.laps}/{self.NUM_LAPS}  {self.phase}  armed {self.armed}'
        line2 = (f'score {fmt(score)}  cam {fmt(sim_cam)}  lidar {fmt(sim_lidar)}  '
                 f'dist {self.distance - self.lap_start_distance:.1f}/{self.lap_gate():.1f} m')
        match = score is not None and score >= self.SIMILARITY_THRESHOLD
        colour = (0, 255, 0) if match else (0, 165, 255)
        cv2.putText(out, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        cv2.putText(out, line2, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        debug_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_publisher.publish(debug_msg)


def main(args=None) -> None:
    """Run the lap counter node."""
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
