# include needed imports
from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


# Gap Following Algorithm 

# [ FTG Tweak 1 ] at every timestep avoid the nearest obstacle
# a: find the nearest LIDAR point and put a safety bubble of radius rb around it
# b: set all points inside the bubble to 0, all nonzero points are free space
# c: find the maximum length sequence of consecutive nonzeros, the max-gap
# d: find the best point in the max-gap

# [ FTG Tweak 4 ] minimize the wiggling problem
# Cap LiDAR ranges to MAX_RANGE to limit influence of distant readings.
# Then follow the center of the max-gap, instead of the deepest direcion point.


class GapFollowNode(Node):
    """
    Reactive gap-following controller using LIDAR range data.
    Steers toward the center of the largest free gap away from the nearest obstacle.
    """

    def __init__(self) -> None:
        """Initialize parameters, state, subscriber, publisher."""
        super().__init__('gap_follow_node')

        # Gap Follow Parameters: 

        self.FREE_DISTANCE = 1.0        # Gaps must be further than this distance (meters)
        self.FOV = math.radians(60.0)   # ± forward looking field of view
        self.BUBBLE_RADIUS = 0.2        # Bubble safety radius, slightly over radius of car ~ (0.3m)
        self.MAX_RANGE = 3.0            # Cap LiDAR ranges above this threshold to this distance


        # Steering parameters: 
        # Note: Does not ensure collision avoidance
        # Eg. If gap is at 60°, applying gain reduces to 30° 
        # and max steering clamps to 22.9°. 
        # Neither param ensure collision avoidance
        # AEB handled seperately in safety node.

        self.MAX_STEERING = 0.4         # Set Max steering limit (±22.9°) 
        self.STEERING_GAIN = 0.5        # Scales gap angle by reducing how much controller reacts 
        self.SMOOTHING = 0.3            # Avoid turning sharply by applying previous and new steering cmds
        self.previous_steering = 0.0    # previous steering cmd, initialized to straight ahead


        # create subscriber and publisher
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            'drive_request',
            10)  # steering request only, safety node sets the speed


        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)

    
    def preprocess_scan(self, scan) -> tuple[np.ndarray, np.ndarray]:
        """
        Keep scans within FOV, clean up invalid readings and clip the ranges.

        Args:
            scan: LaserScan message from the LiDAR

        Returns:
            ranges: cleaned ranges in meters        (0.0 means blocked or invalid)
            angles: angle in radians of each range  (positive angle indicates left)
        """

        # convert scans into numpy array (meters)
        ranges = np.asarray(scan.ranges, dtype=float)
        # calculate the angle corresponding to each scan index (radians)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        # keep scans within FOV
        front = np.abs(angles) <= self.FOV
        ranges = ranges[front]
        angles = angles[front]

        # handle open space and invalid measurements 
        with np.errstate(invalid='ignore'):

            # cap LiDAR max range readings to MAX_RANGE 
            ranges = np.where(ranges > scan.range_max, self.MAX_RANGE, ranges)

            # set NaN and readings below range_min as bad values
            bad = ~np.isfinite(ranges) | (ranges < scan.range_min)

        # set bad values to 0.0
        ranges[bad] = 0.0

        # Cap LiDAR ranges above MAX_RANGE to 3.0 m
        ranges = np.minimum(ranges, self.MAX_RANGE)

        return ranges, angles


    def create_safety_bubble(self, ranges, angle_increment) -> np.ndarray:
        """
        Set every point inside a safety bubble around the nearest obstacle to 0.0

        Args:
            ranges: valid ranges from preprocess_scan
            angle_increment: angle between two beams in radians

        Returns:
            ranges: ranges within the bubble set to 0.0
        """

        # find the indices of nearest valid points
        valid = np.where(ranges > 0.0)[0]
        if len(valid) == 0:
            return ranges

        # find the index of the nearest valid obstacle
        closest = valid[np.argmin(ranges[valid])]

        # find the bubble's angular half-width 
        ratio = min(1.0, self.BUBBLE_RADIUS / ranges[closest])
        bubble_angle = math.asin(ratio)

        # convert angular half width to number of LiDAR beams
        half_width = int(math.ceil(bubble_angle / angle_increment))

        # set the range of indices inside the safety bubble to 0
        start = max(0, closest - half_width)
        end = min(len(ranges), closest + half_width + 1)
        ranges[start:end] = 0.0

        return ranges


    def find_max_gap(self, ranges) -> tuple[int, int] | tuple[None, None]:
        """
        Find the longest sequence of consecutive nonzero ranges.

        Args:
            ranges: ranges after creating safety bubble.

        Returns:
            (start, end): indices of the max-gap
            (None, None): if there is no gap
        """

        best_start = None
        best_end = None
        start = None

        for i, r in enumerate(ranges):

            # a gap begins at the first nonzero point
            if r > 0.0 and start is None:
                start = i

            # a gap ends at a zero or at the end of the scan
            if start is not None and (r <= 0.0 or i == len(ranges) - 1):
                end = i if r > 0.0 else i - 1

                # iterate and find the longest consecutive gap
                if best_start is None or end - start > best_end - best_start:
                    best_start = start
                    best_end = end

                # reset to detect next gap
                start = None

        return best_start, best_end


    def find_best_point(self, gap_start, gap_end) -> int: 
        """
        Select the center point of the max-gap.

        Args:
            gap_start: first index of the max-gap
            gap_end: last index of the max-gap

        Returns:
            index of the best point in the ranges array
        """

        return (gap_start + gap_end) // 2


 
    def calculate_steering_angle(self, best_angle) -> float: 
        """
        Converts the angle of the best point into a smoothed + clamped steering command.

        Args:
            best_angle: angle of the best point in radians  (positive is left)

        Returns:
            steering_angle: Clamped steering angle in radians
        """

        # apply steering gain to scale the best gap angle
        # this adjust how strongly the controller responds
        target = self.STEERING_GAIN * best_angle

        # calculate a steering angle using previous steering command and smoothing
        # this prevents sudden changes in steering
        steering_angle = (self.SMOOTHING * target + (1.0 - self.SMOOTHING) * self.previous_steering)
        self.previous_steering = steering_angle

        # clamp the steering angle
        # MAX_STEERING = 0.4 rad (approximately ±22.9°)
        # based on the F1tenth sim config
        steering_angle = max(-self.MAX_STEERING, min(self.MAX_STEERING, steering_angle))

        return steering_angle


    # Processes incoming LiDAR scan data to calculate and publish gap-following commands.
    def scan_callback(self, msg) -> None:
        """
        a. preprocess scan
        b. puts a safety bubble around the nearest obstacle
        c. finds the max-gap and its center
        d. publishes the steering angle to drive_request

        Error Handling: 
        publishes nothing if there is no gap or if any error raises, AEB is applied by safety node.

        Args:
            msg: LaserScan message containing LiDAR range data and scan attributes.
        Returns:
            None
        """

        # preprocess the LiDAR ranges and corresponding angles
        try:
            ranges, angles = self.preprocess_scan(msg)
            if len(ranges) == 0:
                return

            # set points in the safety bubble to be 0.0
            ranges = self.create_safety_bubble(ranges, msg.angle_increment)

            # set points closer than FREE_DISTANCE to be 0.0
            ranges[ranges < self.FREE_DISTANCE] = 0.0

            # find the longest consecutive gap 
            gap_start, gap_end = self.find_max_gap(ranges)
            if gap_start is None:
                self.get_logger().warn('no gap found, not publishing steering', throttle_duration_sec=1.0)
                return

            # find the center index of the max-gap
            best_index = self.find_best_point(gap_start, gap_end)

            # convert the best index's angle into steering command
            steering_angle = self.calculate_steering_angle(float(angles[best_index]))

            # create and publish the steering command
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.steering_angle = steering_angle
            self.publisher_.publish(drive_msg)

        # error logging
        except Exception as exc:
            self.get_logger().error(f'gap follow failed: {exc}', throttle_duration_sec=1.0)


def main(args=None) -> None:
    rclpy.init(args=args)                   # initialize ROS 2
    gap_follow_node = GapFollowNode()       # creates GapFollowNode instance
    rclpy.spin(gap_follow_node)             # spin the node running + callback
    gap_follow_node.destroy_node()          # clears up the node
    rclpy.shutdown()                        # shut down ROS2


if __name__ == '__main__':
    main()