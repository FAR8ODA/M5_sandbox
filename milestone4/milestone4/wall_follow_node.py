# name and student number: Nyi Nyi Oo, 91180125

# include needed imports
import math
import time

import rclpy
from rclpy.node import Node

import numpy as np  # optional
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class WallFollowNode(Node):
    """
    Reactive wall-following controller using LIDAR range data.
    Keeps the car a set distance from the wall and roughly parallel to it.
    """

    def __init__(self):
        """Initialize the wall-following node: PID gains, subscriber, publisher."""
        super().__init__('wall_follow_node')

  
        # Initialize PID controller parameters and state variables

        # PID gains 
        self.Kp = 0.5
        self.Ki = 0.00
        self.Kd = 0.003

        # Desired distance from the wall
        self.desired_distance = 0.7
        self.filtered_alpha = 0.0
        # AC ~ look ahead distance (slide 43)
        self.ac = 0.3 # in meters

        # Initialize PID integrator and derivative terms
        self.integral = 0.0
        self.previous_error = 0.0

        # Initialize previous time for PID derivative calculation
        self.previous_time = self.get_clock().now()

        # Steering limit to 0.4 rad ~ 22.9°
        self.MAX_STEERING = 0.4    

        # ADD ERROR HANDLING FOR MILESTONE 2 DEMO ----------------------------------------------

        # stop publishing if the wall has been lost for longer than this (s)
        self.MAX_HOLD_TIME = 2.5
        self.CORNER_STEERING = 0.1
        self.CORNER_DELAY = 0.6

        self.last_valid_time = self.get_clock().now()

        #  -------------------------------------------------------------------------------------


        # create subscriber and publisher
        
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            'drive_request',
            10) # avoid conflict with safety node

        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)


    # add methods as needed below here
    # ...


    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle
        """

        # find the index corresponding to the given angle
        index = round((angle - range_data.angle_min) / range_data.angle_increment)

        # check that the index is within the scan range
        if index < 0 or index >= len(range_data.ranges):
            return None

        # get the range measurement at the calculated index
        r = range_data.ranges[index]

        # take care of NaNs and infs ranges before returning range
        if not (math.isfinite(r) and range_data.range_min <= r <= range_data.range_max):
            return None

        return r



    # Calculate the wall following error (slides 41-43)

    # slide 41
    # use LiDar to hold a fixed distance from the wall
    # Slide angles:    [   0,  180], 0: (right) wall
    # Sim Scan angles: [-135, +135], 0: straight ahead
    # Following Right wall: sim angle = slide angle - 90
    # Following Left wall: b: +90, a: 90 - theta.

    # slide 42
    # pick 2 rays: b at slide 0 deg (perpendicular to the wall)
    # a at theta deg (toward the front)
    # alpha = atan2(a*cos(theta) - b, a*sin(theta)) 
    # AB = b * cos(alpha) 

    # slide 43: 
    # look ahead: CD = AB + AC*sin(alpha)
    # AC is the look-ahead distance 
    # CD is the distance we predict to the wall after driving AC meters.
    # error = desired_distance - CD

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop).

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: dist - CD
            error: desired distance minus the predicted distance after the look-ahead
        """
    
        # angle between 2 rays (a, b) ~ 45 degrees
        theta = math.radians(45.0)

        # assign 2 rays (a,b) towards the left wall
        b = self.get_range(range_data, math.radians(90.0))
        a = self.get_range(range_data, math.radians(90.0) - theta)

        # check if range measurements are valid
        if a is None or b is None or (b > 0 and abs(a - b) / b > 1.5):
            return None

        # calculate α: orientation of the car with respect to the wall
        alpha = math.atan2(a * math.cos(theta) - b, a * math.sin(theta))

        # a car following a wall is never this far from parallel; treat it as no wall
        if abs(alpha) > math.radians(40.0):
            return None


        # calculate AB: the current perpendicular distance
        ab = b * math.cos(alpha)

        # calculate CD: the predicted distance after AC meters
        cd = ab + self.ac * math.sin(alpha)
        
        # calculate error = desired trajectory - CD
        error = dist - cd
        self.get_logger().info(f'alpha={math.degrees(alpha):.2f} error={error:.3f}',throttle_duration_sec=1.0)
        return error




    # PID Output (slides 33 - 37)
    # PID: Theta = Kp*e(t) + Ki*E(t) + Kd*de(t)/dt

    # Slide 33
    # Porpotional Control: Θd= C Kp(-y –Lsin(Θ)) = C Kpe(t)
    # C scales distances to angles (vehicle dimensions, tire diameter,
    # cornering stiffness, Ackermann parameters). We fold C into Kp.

    # Slide 35
    # Derivative Control: Θ = Kpe(t) + Kdde(t)/dt
    # de/dt is approximated as (error - previous_error) / dt.

    # Slide 37
    # Integral Control: Θ = Kpe(t) + KI E(t) + Kdde(t)/dt
    # E(t) is the integral of the error, approximated as a running sum of error*dt.
    # E(t) must be bounded

    # Sign Convention:
    # our error (dist - CD) is the negative of the slide's e(t) -(y + L*sin(Theta));
    # this function returns the PID output in our error's sign
    # and calculate_steering_angle() applies the minus sign to match the sign convention.

    def calculate_pid_output(self, error):
        """
        Calculates the PID output using the current wall-following error.

        Args:
            error: current wall-following error in meters

        Returns:
            pid_output: combined PID control signal
        """

        # calculate difference between current time and previous time of the last PID calculation
        current_time = self.get_clock().now()
        dt = (current_time - self.previous_time).nanoseconds / 1e9
        
        # prevent division by zero or negative time intervals
        if dt <= 0.0:          
            dt = 1e-3  

        # integral term I = Ki * sum(ek) * delta(t)
        # helps the controller eliminate steady-state error
        self.integral += error * dt                         # accumulate the integral term, error over time
        self.integral = max(-1.0, min(1.0, self.integral))  # bound the integral between [-1,1] 

        # derivative term D = Kd * (ek - ek-1) / delta(t)
        # helps the controller respond to how quickly the error is changing
        derivative = (error - self.previous_error) / dt     # measure the derivative term, change in error over time

        # calculate PID terms 
        P = self.Kp * error          # responds to current error
        I = self.Ki * self.integral  # corrects accumulated error 
        D = self.Kd * derivative     # responds to the rate of change of error

        # combine the PID terms into one control seteering signal 
        pid_output = P + I + D   

        # update previous error and time for the next PID calculation
        self.previous_error = error
        self.previous_time = current_time

        return pid_output



    # Sign Convention: positive result steers left, negative steers right 
    # Steering limit ~ ±0.4189 rad ~ 24°
    # Matches s_min/s_max in f1tenth_gym's vehicle params
    def calculate_steering_angle(self, pid_output):
        """
        Converts PID output to steering angle and limits the maximum allowed steering angle.

        Args:
            pid_output: Combined PID control signal.

        Returns:
            steering_angle: Clamped steering angle in radians.
        """

        # negate pid_output to match the sign convention 
        # (positive steers left, negative steers right)
        steering_angle = -pid_output

        # Limit the steering angle 
        steering_angle = max(-self.MAX_STEERING,min(self.MAX_STEERING, steering_angle))

        return steering_angle


    # Processes incoming LiDAR scan data to calculate and publish wall-following commands.
    def scan_callback(self, msg):
        """
        a. calculates error 
        b. calculates pid_output 
        c. clamps steering angle 
        d.  publishes the steering angle to wall_follow/drive

        Error Handling: Publishes nothing if the wall has been lost longer than MAX_HOLD_TIME
        or if error raises, so safety_node stops the car.

        Args:
            msg: LaserScan message containing LiDAR range data and scan attributes.
        Returns:
            None
        """
        # ADD ERROR HANDLING FOR MILESTONE 2 DEMO ----------------------------------------------
        try:
            error = self.get_error(msg, self.desired_distance)
            now = self.get_clock().now()

            if error is None:
                lost_for = (now - self.last_valid_time).nanoseconds / 1e9
                if lost_for > self.MAX_HOLD_TIME:
                    self.get_logger().warn('wall lost, not publishing steering', throttle_duration_sec=1.0)
                    return
                # wall just vanished: it is a left corner, so turn left to find it again
                if lost_for < self.CORNER_DELAY:
                    steering_angle = 0.0
                else:
                    steering_angle = self.CORNER_STEERING
            else:
                self.last_valid_time = now
                pid_output = self.calculate_pid_output(error)
                steering_angle = self.calculate_steering_angle(pid_output)

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering_angle
            self.publisher_.publish(drive_msg)
        except Exception as exc:
            self.get_logger().error(
                f'wall follow failed: {exc}', throttle_duration_sec=1.0)
        # -------------------------------------------------------------------------------------            



def main(args=None):
    rclpy.init(args=args)                   # initialize ROS 2
    wall_follow_node = WallFollowNode()     # creates WallFollowNode instance 
    rclpy.spin(wall_follow_node)            # spin the node running + callback
    wall_follow_node.destroy_node()         # clears up the node
    rclpy.shutdown()                        # shut down ROS2


if __name__ == '__main__':
    main()