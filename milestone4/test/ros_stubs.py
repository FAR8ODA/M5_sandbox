"""
Lightweight stand-ins for rclpy and the ROS message types, for the unit tests.

Importing this module puts fake ``rclpy``, ``sensor_msgs``, ``nav_msgs``, ``std_msgs``,
``ackermann_msgs`` and ``cv_bridge`` modules in ``sys.modules``, so the node classes can
be created and their callbacks called directly, with a clock the test controls
(``CLOCK.seconds``). No ROS installation or running graph is needed, so the tests run
on a laptop as well as under ``colcon test``.

If OpenCV is not installed, an empty ``cv2`` module is added too, so the node modules
import; tests that really need OpenCV check ``HAVE_CV2`` and skip.
"""

import importlib.util
from pathlib import Path
import sys
import types

# the package source (milestone4/milestone4) must be importable
# as ``milestone4``
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

HAVE_CV2 = importlib.util.find_spec('cv2') is not None


# Time -------------------------------------------------------------------------------------------

class Stamp:
    """builtin_interfaces/Time."""

    def __init__(self, sec: int = 0, nanosec: int = 0) -> None:
        self.sec = sec
        self.nanosec = nanosec


class Duration:
    """rclpy.duration.Duration (nanoseconds only)."""

    def __init__(self, nanoseconds: int) -> None:
        self.nanoseconds = nanoseconds


class Time:
    """rclpy.time.Time (nanoseconds only)."""

    def __init__(self, seconds: float = 0.0, nanoseconds: int = 0) -> None:
        self.nanoseconds = int(round(seconds * 1e9)) + int(nanoseconds)

    def __sub__(self, other: 'Time') -> Duration:
        return Duration(self.nanoseconds - other.nanoseconds)

    def to_msg(self) -> Stamp:
        sec = self.nanoseconds // 1_000_000_000
        return Stamp(sec=sec, nanosec=self.nanoseconds - sec * 1_000_000_000)

    @classmethod
    def from_msg(cls, msg: Stamp) -> 'Time':
        return cls(nanoseconds=msg.sec * 1_000_000_000 + msg.nanosec)


class FakeClock:
    """Clock shared by every fake node; tests set ``CLOCK.seconds``."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def now(self) -> Time:
        return Time(seconds=self.seconds)


CLOCK = FakeClock()


def stamp(seconds: float) -> Stamp:
    """Return a header stamp for a time in seconds."""
    return Time(seconds=seconds).to_msg()


# Node -------------------------------------------------------------------------------------------

class Parameter:
    """rclpy Parameter (value only)."""

    def __init__(self, value) -> None:
        self.value = value


class Logger:
    """
    Collects log messages so tests can check them.

    Like rclpy, each source line that logs is one "call site": it must always use the same
    severity and the same filters (such as throttle_duration_sec), or rclpy raises
    ValueError inside the callback and the node stops spinning. The stand-in enforces that.
    """

    def __init__(self) -> None:
        self.records = []
        self.call_sites = {}

    def _log(self, level: str, message: str, kwargs: dict) -> None:
        caller = sys._getframe(2)
        site = (caller.f_code.co_filename, caller.f_lineno, caller.f_lasti)
        filters = frozenset(k for k in kwargs if k in ('throttle_duration_sec', 'once',
                                                       'skip_first'))
        if site not in self.call_sites:
            self.call_sites[site] = (level, filters)
        elif self.call_sites[site][0] != level:
            raise ValueError('Logger severity cannot be changed between calls.')
        elif self.call_sites[site][1] != filters:
            raise ValueError('Requested logging filters cannot be changed between calls.')
        self.records.append((level, message))

    def debug(self, message: str, **kwargs) -> None:
        self._log('debug', message, kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log('info', message, kwargs)

    def warn(self, message: str, **kwargs) -> None:
        self._log('warn', message, kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log('warn', message, kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log('error', message, kwargs)


class Publisher:
    """Records every published message."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages = []
        self.subscribers = 0                # tests set this to pretend someone is listening

    def publish(self, msg) -> None:
        self.messages.append(msg)

    def get_subscription_count(self) -> int:
        return self.subscribers


# parameter values a test wants a node to start with: {'node_name': {'param': value}}
PARAM_OVERRIDES = {}


class Node:
    """rclpy.node.Node with only what our nodes use."""

    def __init__(self, node_name: str, **kwargs) -> None:
        self._name = node_name
        self._params = {}
        self._logger = Logger()
        self.publishers_by_topic = {}
        self.callbacks_by_topic = {}
        self.timers = []

    def declare_parameter(self, name: str, value=None, *args, **kwargs) -> Parameter:
        value = PARAM_OVERRIDES.get(self._name, {}).get(name, value)
        self._params[name] = value
        return Parameter(value)

    def get_parameter(self, name: str) -> Parameter:
        return Parameter(self._params[name])

    def create_publisher(self, msg_type, topic: str, qos) -> Publisher:
        pub = Publisher(topic)
        self.publishers_by_topic[topic] = pub
        return pub

    def create_subscription(self, msg_type, topic: str, callback, qos) -> object:
        self.callbacks_by_topic[topic] = callback
        return object()

    def create_timer(self, period: float, callback) -> object:
        self.timers.append((period, callback))
        return object()

    def get_clock(self) -> FakeClock:
        return CLOCK

    def get_logger(self) -> Logger:
        return self._logger

    def destroy_node(self) -> None:
        pass


# Messages ---------------------------------------------------------------------------------------

class Header:
    """std_msgs/Header."""

    def __init__(self) -> None:
        self.stamp = Stamp()
        self.frame_id = ''


class LaserScan:
    """sensor_msgs/LaserScan."""

    def __init__(self) -> None:
        self.header = Header()
        self.angle_min = 0.0
        self.angle_max = 0.0
        self.angle_increment = 0.0
        self.time_increment = 0.0
        self.scan_time = 0.0
        self.range_min = 0.0
        self.range_max = 0.0
        self.ranges = []
        self.intensities = []


class Image:
    """sensor_msgs/Image (header only)."""

    def __init__(self) -> None:
        self.header = Header()


class CameraInfo:
    """sensor_msgs/CameraInfo (fields we use)."""

    def __init__(self) -> None:
        self.header = Header()
        self.k = [0.0] * 9
        self.width = 0
        self.height = 0


class _Vector3:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _Twist:
    def __init__(self) -> None:
        self.linear = _Vector3()
        self.angular = _Vector3()


class _TwistWithCovariance:
    def __init__(self) -> None:
        self.twist = _Twist()


class Odometry:
    """nav_msgs/Odometry (twist only)."""

    def __init__(self) -> None:
        self.header = Header()
        self.twist = _TwistWithCovariance()


class AckermannDrive:
    """ackermann_msgs/AckermannDrive."""

    def __init__(self) -> None:
        self.steering_angle = 0.0
        self.steering_angle_velocity = 0.0
        self.speed = 0.0
        self.acceleration = 0.0
        self.jerk = 0.0


class AckermannDriveStamped:
    """ackermann_msgs/AckermannDriveStamped."""

    def __init__(self) -> None:
        self.header = Header()
        self.drive = AckermannDrive()


class _Data:
    """std_msgs messages with a single data field."""

    def __init__(self, data=None) -> None:
        self.data = data


class String(_Data):
    """std_msgs/String."""


class Float32(_Data):
    """std_msgs/Float32."""


class Int32(_Data):
    """std_msgs/Int32."""


class CvBridge:
    """cv_bridge.CvBridge; tests call the node logic directly instead."""

    def imgmsg_to_cv2(self, msg, desired_encoding: str = 'passthrough'):
        raise NotImplementedError('tests pass NumPy images directly')

    def cv2_to_imgmsg(self, img, encoding: str = 'passthrough'):
        raise NotImplementedError('tests pass NumPy images directly')


# Install ----------------------------------------------------------------------------------------

def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def install() -> None:
    """Put the fake modules in sys.modules (safe to call more than once)."""
    rclpy = _module('rclpy', init=lambda args=None: None, spin=lambda node: None,
                    shutdown=lambda: None, ok=lambda: False)
    rclpy.node = _module('rclpy.node', Node=Node)
    rclpy.time = _module('rclpy.time', Time=Time)
    rclpy.qos = _module('rclpy.qos', qos_profile_sensor_data=10)

    sensor_msgs = _module('sensor_msgs')
    sensor_msgs.msg = _module('sensor_msgs.msg', LaserScan=LaserScan, Image=Image,
                              CameraInfo=CameraInfo)
    nav_msgs = _module('nav_msgs')
    nav_msgs.msg = _module('nav_msgs.msg', Odometry=Odometry)
    std_msgs = _module('std_msgs')
    std_msgs.msg = _module('std_msgs.msg', String=String, Float32=Float32, Int32=Int32)
    ackermann_msgs = _module('ackermann_msgs')
    ackermann_msgs.msg = _module('ackermann_msgs.msg', AckermannDrive=AckermannDrive,
                                 AckermannDriveStamped=AckermannDriveStamped)
    _module('cv_bridge', CvBridge=CvBridge)
    if not HAVE_CV2:
        _module('cv2')


install()


# Helpers for the tests --------------------------------------------------------------------------

LIDAR_ANGLE_MIN = -2.35619449   # Hokuyo 10LX: 270 degrees ...
LIDAR_INCREMENT = 0.00436332    # ... in 0.25 degree steps
LIDAR_BEAMS = 1081


def lidar_angles():
    """Return the bearing of every simulated LiDAR beam (rad)."""
    import numpy as np
    return LIDAR_ANGLE_MIN + np.arange(LIDAR_BEAMS) * LIDAR_INCREMENT


def make_scan(ranges, angle_min: float = LIDAR_ANGLE_MIN,
              angle_increment: float = LIDAR_INCREMENT,
              range_min: float = 0.06, range_max: float = 10.0, t: float = 0.0) -> LaserScan:
    """Build a LaserScan message from a list of ranges."""
    scan = LaserScan()
    scan.header.stamp = stamp(t)
    scan.angle_min = angle_min
    scan.angle_increment = angle_increment
    scan.angle_max = angle_min + (len(ranges) - 1) * angle_increment
    scan.range_min = range_min
    scan.range_max = range_max
    scan.ranges = [float(r) for r in ranges]
    return scan


def make_request(steering: float, speed: float) -> AckermannDriveStamped:
    """Build a drive request."""
    msg = AckermannDriveStamped()
    msg.drive.steering_angle = steering
    msg.drive.speed = speed
    return msg


def make_odom(speed: float, yaw_rate: float = 0.0) -> Odometry:
    """Build an odometry message with a forward speed and a yaw rate."""
    msg = Odometry()
    msg.twist.twist.linear.x = speed
    msg.twist.twist.angular.z = yaw_rate
    return msg


def world_scan(world, x: float, y: float, yaw: float):
    """
    Return simulated LiDAR ranges of world points (Nx2, m) seen from a car pose.

    Each beam keeps the nearest point that falls on it, so dense points (a few mm apart)
    behave like solid walls.
    """
    import math
    import numpy as np
    d = np.asarray(world, dtype=float) - [x, y]
    c, s = math.cos(-yaw), math.sin(-yaw)
    px = c * d[:, 0] - s * d[:, 1]
    py = s * d[:, 0] + c * d[:, 1]
    r = np.hypot(px, py)
    beams = np.round((np.arctan2(py, px) - LIDAR_ANGLE_MIN) / LIDAR_INCREMENT).astype(int)
    inside = (beams >= 0) & (beams < LIDAR_BEAMS)
    out = np.full(LIDAR_BEAMS, np.inf)
    np.minimum.at(out, beams[inside], r[inside])
    return out


def wall_points(x0: float, y0: float, x1: float, y1: float, step: float = 0.005):
    """Return dense points (Nx2) along a straight wall from (x0, y0) to (x1, y1)."""
    import numpy as np
    n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
    t = np.linspace(0.0, 1.0, n)
    return np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], axis=1)


def corridor_ranges(half_width: float = 0.6, far: float = 10.0):
    """
    Return LiDAR ranges for a straight corridor with walls at y = +/- half_width.

    Beams that never hit a wall (straight down the corridor) read ``far``.
    """
    import numpy as np
    a = lidar_angles()
    with np.errstate(divide='ignore'):
        r = np.where(np.abs(np.sin(a)) > 1e-6, half_width / np.abs(np.sin(a)), np.inf)
    return np.minimum(r, far)


def add_box(ranges, x: float, y_min: float, y_max: float):
    """
    Put a flat box face at distance x ahead, from y_min to y_max (left positive).

    Each beam that crosses the face is shortened to it.
    """
    import numpy as np
    a = lidar_angles()
    out = np.array(ranges, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_face = x / np.cos(a)
        y_face = r_face * np.sin(a)
    hit = (np.cos(a) > 0.0) & (y_face >= y_min) & (y_face <= y_max) & (r_face < out)
    out[hit] = r_face[hit]
    return out
