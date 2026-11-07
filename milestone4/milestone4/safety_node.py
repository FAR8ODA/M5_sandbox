"""
TTC-based safety node for the F1Tenth vehicle.

The wall-following node requests steering only. This node owns vehicle
speed and is the only node that publishes commands to /drive.

Safety features:

    LiDAR AEB (Milestone 4): TTC is computed from the LiDAR scan (/scan) by
    default. The track walls are matte black ducting, which absorbs the depth
    camera's infrared light, so the Milestone 3 depth-based AEB could not see
    them; the LiDAR does, and covers 270 degrees instead of the camera's ~87.
    The launch argument aeb_scan_topic:=depth_scan restores depth-based AEB.
    Odom Sign Fix: If the car’s odometry reads negative while moving forward,
    TTC math breaks, so added abs() in TTC calculation.
    Watchdog: If no steering command or scan arrives for 0.25 s, the car stops gracefully.
    Lap Counter Stop: A message with negative speed on drive_request stops the car

"""

# include needed imports
from __future__ import annotations  # allows "X | None" hints on Python 3.8 (Foxy)

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class SafetyNode(Node):
    """Control speed, apply Automatic Emergency Braking, and watch for stale inputs."""

    # AEB stops the car at or below this time-to-collision (seconds).
    TTC_THRESHOLD = 0.65

    # Speed the car drives at whenever AEB is not engaged (m/s).
    DRIVE_SPEED = 0.7

    # Brake if steering requests or scans are older than this (seconds).
    STALE_TIMEOUT = 0.25

    # How often the watchdog checks the inputs (seconds).
    WATCHDOG_PERIOD = 0.05

    def __init__(self) -> None:
        """Initialize parameters, ROS communication, safety state and the watchdog."""
        super().__init__('safety_node')

        # The shared launch file sets these; the class constants are the defaults.
        self.declare_parameter('ttc_threshold', self.TTC_THRESHOLD)
        self.declare_parameter('drive_speed', self.DRIVE_SPEED)
        self.declare_parameter('stale_timeout', self.STALE_TIMEOUT)
        self.ttc_threshold = float(self.get_parameter('ttc_threshold').value)
        self.drive_speed = float(self.get_parameter('drive_speed').value)
        self.stale_timeout = float(self.get_parameter('stale_timeout').value)

        # Wall stop: latched stop when the scan shows one continuous object wider than
        # wall_width within wall_distance ahead (a cardboard wall across the lane).
        # Designed for the depth scan, which cannot see the black track walls. On the
        # LiDAR scan the outer wall of every corner matches, so keep it off with LiDAR AEB.
        self.declare_parameter('wall_stop', False)
        self.declare_parameter('wall_distance', 0.7)    # m ahead (the floor in the depth band is farther)
        self.declare_parameter('wall_width', 0.6)       # m, wider than one box (0.5 m), narrower than the lane
        self.wall_frames = 0                            # depth scans in a row with a wall ahead

        self.current_speed = 0.0                        # forward speed of the vehicle
        self.latest_scan: LaserScan | None = None       # latest scan message
        self.latest_steering_angle = 0.0                # latest steering angle command
        self.safety_mode = 'normal'                     # normal, aeb, or stale

        # Arrival times (node clock) of the latest inputs, for the watchdog.
        self.last_request_time: Time | None = None
        self.last_scan_time: Time | None = None

        # create the required publisher, subscriber ...

        # create a publisher that send drive commands to drive topic
        # note: only safety_node publishes to /drive.
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            'drive',
            10
        )

        # create a subscriber to detect obstacles and calculate TTC.
        # Remapped by the launch file to /scan (LiDAR, default) or depth_scan.
        # Sensor-data QoS (best effort) receives from both reliable and
        # best-effort publishers, so it works with any LiDAR driver setting.
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # create a subscriber to get current forward speed
        # used when calculating TTC value (remapped to /odom on the car)
        self.odom_subscription = self.create_subscription(
            Odometry,
            'ego_racecar/odom',
            self.odom_callback,
            10
        )

        # create a subscriber for requests from the steering controller
        # note: ensure the controller does not override safety
        self.drive_request_subscription = self.create_subscription(
            AckermannDriveStamped,
            'drive_request',
            self.drive_request_callback,
            10
        )

        # watchdog: brakes when inputs go stale, even if no callback fires
        self.watchdog_timer = self.create_timer(
            self.WATCHDOG_PERIOD,
            self.watchdog_callback
        )

    def odom_callback(self, msg: Odometry) -> None:
        """
        Update the measured forward speed.

        Args:
            msg (Odometry): Incoming odometry message.
        """

        # Update current speed from odometry message.
        self.current_speed = abs(msg.twist.twist.linear.x)

        # log for AEB events
        if self.safety_mode == 'aeb':

            # throttle once per second max to avoid flooding the console
            self.get_logger().warn(
                f'AEB braking: current speed={self.current_speed:.2f} m/s',
                throttle_duration_sec=1
            )

    def get_range(self, scan_msg: LaserScan, angle_deg: float) -> float:
        """
        Get the scan range at a given angle.

        Args:
            scan_msg (LaserScan): The scan.
            angle_deg (float): Angle to read, in degrees.

        Returns:
            float: Range in meters,
            or NaN if invalid/out-of-range
        """

        # convert angle from degrees to radians
        angle_rad = math.radians(angle_deg)

        # convert angle to the corresponding array index
        index = int(
            round(
                (angle_rad - scan_msg.angle_min)    # offset from first beam
                / scan_msg.angle_increment          # divide by angle increment/step between each beam
            )
        )

        # check if the index is within the valid range of the scan
        if index < 0 or index >= len(scan_msg.ranges):
            return np.nan

        # get the distance at the corresponding index
        distance = scan_msg.ranges[index]

        # discard invalid (inf/NaN) or out-of-range values
        if (
            not np.isfinite(distance)
            or distance < scan_msg.range_min
            or distance > scan_msg.range_max
        ):
            return np.nan

        return distance

    def compute_min_ttc(self, scan_msg: LaserScan, speed: float) -> float:
        """
        Compute the minimum time-to-collision (TTC) across all valid beams.

        Args:
            scan_msg (LaserScan): The scan.
            speed (float): Current forward speed of the vehicle, in m/s.

        Returns:
            float: Minimum TTC in seconds, or inf if no collision is approaching.
        """

        # convert ranges to a numpy array
        ranges = np.array(
            scan_msg.ranges,
            dtype=np.float64
        )

        # compute the angle of each beam in radians
        angles = (
            scan_msg.angle_min                                       # angle of first beam
            + np.arange(len(ranges)) * scan_msg.angle_increment      # increment for each beam index
        )

        # range_rate:
        # project current forward speed onto each beam direction
        # used to estimate how quickly the measured obstacle distance is changing.
        range_rate = (
            -speed * np.cos(angles)        # Sign Convention:
        )                                  # A negative range rate means the obstacle is getting closer.

        # default beams to inf (no collision)
        ttc = np.full(
            len(ranges),
            np.inf
        )

        # keep beams only if they are valid and approaching
        # (beams at range_max mean "nothing seen" on the camera scan, so they are excluded)
        valid = (
            np.isfinite(ranges)
            & (ranges >= scan_msg.range_min)
            & (ranges < scan_msg.range_max)
            & (range_rate < 0.0)
        )

        # TTC (Time to Collision): Distance/positive closing speed.
        ttc[valid] = (
            ranges[valid]
            / (-range_rate[valid])        # Sign Convention:
        )                                 # Negate so closing speed is positive before TTC calculation

        # return the minimum TTC across all beams
        # since it represents an approaching collision
        return float(
            np.min(ttc)
        )

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Store scan data and immediately brake if motion is unsafe.

        Args:
            msg (LaserScan): Incoming scan.
        """

        # store the latest scan and when it arrived
        self.latest_scan = msg
        self.last_scan_time = self.get_clock().now()

        # compute the minimum TTC based on the latest scan
        min_ttc = self.compute_min_ttc(
            msg,
            self.current_speed
        )

        # if TTC is below the threshold, engage AEB
        if min_ttc <= self.ttc_threshold:
            self.engage_aeb(
                min_ttc
            )

        # wall stop: a wide object close ahead for 2 depth scans in a row -> latched stop
        if self.get_parameter('wall_stop').value and self.safety_mode != 'aeb':
            self.wall_frames = self.wall_frames + 1 if self.wall_ahead(msg) else 0
            if self.wall_frames >= 2:
                self.get_logger().warn('Wall ahead (scan): latched stop')
                self.safety_mode = 'aeb'
                self.emergency_brake()

    def wall_ahead(self, scan_msg: LaserScan) -> bool:
        """
        Check for one continuous object wider than wall_width within wall_distance ahead.

        Args:
            scan_msg (LaserScan): Depth scan.

        Returns:
            bool: True if such an object is in the scan.
        """

        distance = float(self.get_parameter('wall_distance').value)
        width_needed = float(self.get_parameter('wall_width').value)

        # convert the beams into (x ahead, y left) points
        ranges = np.array(scan_msg.ranges, dtype=np.float64)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # beams that hit something closer than wall_distance ahead
        with np.errstate(invalid='ignore'):
            near = (np.isfinite(ranges) & (ranges >= scan_msg.range_min)
                    & (ranges < scan_msg.range_max) & (x > 0.0) & (x < distance))

        # widest run of neighbouring beams whose points are within 0.15 m of each other
        widest, start = 0.0, None
        for i in range(len(ranges) + 1):
            joined = (i < len(ranges) and near[i] and start is not None
                      and math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) <= 0.15)
            if joined:
                continue
            if start is not None:
                widest = max(widest, math.hypot(x[i - 1] - x[start], y[i - 1] - y[start]))
            start = i if i < len(ranges) and near[i] else None

        return widest >= width_needed

    def drive_request_callback(self, msg: AckermannDriveStamped) -> None:
        """
        Drive at drive_speed with the requested steering, or stop if
        a collision is imminent. Once stopped by AEB, stay stopped.

        Args:
            msg (AckermannDriveStamped): Steering request. A negative speed
                flag requests a latched stop; zero leaves speed to this node;
                a positive speed caps the speed (never above drive_speed).
        """

        # store the latest steering angle and when it arrived
        self.latest_steering_angle = (
            msg.drive.steering_angle
        )
        self.last_request_time = self.get_clock().now()

        # A negative speed flag requests a permanent stop (e.g. lap counter done).
        # Normal steering-only requests leave speed at its default of zero.
        if msg.drive.speed < 0.0 and self.safety_mode != 'aeb':
            self.safety_mode = 'aeb'
            self.get_logger().warn('Navigation requested a latched stop')

        # AEB is latched: once engaged, stay stopped until the node restarts
        if self.safety_mode == 'aeb':
            self.emergency_brake()
            return

        # if the scan is missing or stale, the watchdog handles braking
        if not self.inputs_fresh():
            return

        # inputs are fresh again after a stale period
        if self.safety_mode == 'stale':
            self.safety_mode = 'normal'
            self.get_logger().info('Inputs fresh again: resuming')

        # a positive requested speed caps the speed (e.g. gap-follow asking to crawl
        # into a corner); 0 leaves it at drive_speed. Never faster than drive_speed.
        speed = min(self.drive_speed, msg.drive.speed) if msg.drive.speed > 0.0 else self.drive_speed

        # compute the minimum TTC based on the latest scan and the speed to be commanded
        min_ttc = self.compute_min_ttc(
            self.latest_scan,
            speed
        )

        # if TTC is at or below the threshold, engage brake
        if min_ttc <= self.ttc_threshold:
            self.engage_aeb(
                min_ttc
            )
            return

        # publish drive commands only after safety checks
        self.publish_drive_command(
            speed,
            self.latest_steering_angle   # forward request from the steering controller
        )

    def age_seconds(self, stamp: Time | None) -> float:
        """
        Return how long ago an input arrived.

        Args:
            stamp (Time | None): Arrival time of the input, or None if never.

        Returns:
            float: Age in seconds, or inf if the input never arrived.
        """
        if stamp is None:
            return math.inf
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def inputs_fresh(self) -> bool:
        """
        Check that both the steering requests and the scan are recent.

        Returns:
            bool: True if both arrived within stale_timeout.
        """
        return (
            self.age_seconds(self.last_request_time) <= self.stale_timeout
            and self.age_seconds(self.last_scan_time) <= self.stale_timeout
        )

    def watchdog_callback(self) -> None:
        """
        Command zero speed whenever an input is stale.

        """
        if self.inputs_fresh():
            return

        # log once when entering the stale state (AEB keeps its own state)
        if self.safety_mode == 'normal':
            self.safety_mode = 'stale'
            self.get_logger().warn(
                f'Inputs stale (steering {self.age_seconds(self.last_request_time):.2f}s, '
                f'scan {self.age_seconds(self.last_scan_time):.2f}s): braking'
            )

        self.emergency_brake()

    def engage_aeb(self, min_ttc: float) -> None:
        """
        Engage Automatic Emergency Braking.

        Args:
            min_ttc (float): Minimum time-to-collision that triggered AEB,
            in seconds.
        """

        # log AEB engaged
        if self.safety_mode != 'aeb':
            self.get_logger().warn(
                f'AEB engaged: '
                f'TTC={min_ttc:.2f}s, '
                f'current speed={self.current_speed:.2f} m/s'
            )

        # update safety mode to AEB
        self.safety_mode = 'aeb'

        # and engage emergency brake
        self.emergency_brake()

    def publish_drive_command(self, speed: float, steering_angle: float) -> None:
        """
        Publish the final command to /drive.

        Args:
            speed (float): Target speed, in m/s.
            steering_angle (float): Target steering angle, in radians.
        """

        # create a new drive message
        drive_msg = AckermannDriveStamped()

        # set the header timestamp
        drive_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        # set the speed and steering angle
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle

        # publish the drive command
        self.drive_publisher.publish(
            drive_msg
        )

    def emergency_brake(self) -> None:
        """
        Command zero speed while preserving steering.
        """

        # publish zero speed command
        # and latest steering angle
        self.publish_drive_command(
            0.0,
            self.latest_steering_angle
        )


def main(args=None) -> None:
    """Run the safety node."""
    rclpy.init(args=args)          # initialize ROS 2
    safety_node = SafetyNode()     # create instance of SafetyNode
    try:
        rclpy.spin(safety_node)    # spin to handle callbacks
    except KeyboardInterrupt:
        pass
    safety_node.destroy_node()     # destroy safety node
    if rclpy.ok():
        rclpy.shutdown()           # shutdown ROS 2 (skip if Ctrl+C already did)


if __name__ == '__main__':
    main()