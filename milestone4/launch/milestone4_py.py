"""
Launch file for Milestone 4: LiDAR-primary driving with camera fusion and AEB.

Parameters come from config/milestone4.yaml (or params_file:=...); tuning arguments
given on the command line override it for that run, e.g.
    ros2 launch milestone4 milestone4_py.py max_speed:=1.5 use_camera:=false
"""

from typing import Callable, Dict, List, Tuple

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml


def as_bool(value: str) -> bool:
    """
    Read a launch argument as a bool.

    Args:
        value (str): 'true' or 'false' (any case).

    Returns:
        bool: The value.

    Raises:
        ValueError: If the value is not true or false.
    """
    # case-insensitive, surrounding spaces ignored
    text = value.strip().lower()
    if text not in ('true', 'false'):
        raise ValueError(f'expected true or false, got {value!r}')
    return text == 'true'


# Tuning arguments: (argument, description, type). They default to '' (not given), so the
# YAML file decides; only arguments given on the command line override it.
TUNING_ARGS: List[Tuple[str, str, Callable]] = [
    # Driving
    ('max_speed', 'Highest speed on a clear straight, also the safety cap (m/s).', float),
    ('pass_speed', 'Speed cap while passing a camera-confirmed box (m/s).', float),
    ('lookahead', 'Longest pure-pursuit distance (m).', float),
    ('gain', 'Pure-pursuit steering gain.', float),
    ('a_lat', 'Sideways grip for the cornering speed (m/s^2).', float),
    ('margin', 'Clearance kept at obstacle and barrier edges (m).', float),
    ('smoothing', 'Weight of the previous steering command (0-1).', float),
    ('deep_fraction', 'Beams this close to the deepest count as the gap (0-1).', float),
    ('lidar_x', 'LiDAR distance ahead of the rear axle, for pure pursuit (m).', float),
    # Safety (the braking model is shared by gap_follow_node and safety_node)
    ('ttc_margin', 'Extra time added to the variable TTC threshold (s).', float),
    ('a_brake', 'Measured braking deceleration (m/s^2).', float),
    ('stop_margin', 'Gap AEB leaves in front of the LiDAR (m).', float),
    ('track_obstacles', 'true: AEB also uses obstacle speeds (moving ones).', as_bool),
    # Lap counter
    ('num_laps', 'Laps before the latched stop.', int),
    ('similarity_threshold', 'Lap score meaning "back at the start".', float),
    ('rearm_threshold', 'Lap score must drop below this before the next lap.', float),
    ('min_lap_distance', 'Shortest possible lap, about 3/4 of a lap (m).', float),
    # Camera mounting
    ('cam_x', 'Camera position ahead of the LiDAR (m).', float),
    ('cam_yaw_deg', 'Camera heading relative to the LiDAR (degrees).', float),
    ('cam_z', 'Camera height above the LiDAR scan plane (m).', float),
]

# Which node parameter each tuning argument sets: {node: {parameter: argument}}.
# max_speed sets both the planner's top speed and the safety node's cap, so they never differ.
OVERRIDES: Dict[str, Dict[str, str]] = {
    'gap_follow_node': {
        'max_speed': 'max_speed', 'pass_speed': 'pass_speed', 'lookahead': 'lookahead',
        'gain': 'gain', 'a_lat': 'a_lat', 'margin': 'margin', 'smoothing': 'smoothing',
        'deep_fraction': 'deep_fraction', 'lidar_x': 'lidar_x', 'a_brake': 'a_brake',
        'ttc_margin': 'ttc_margin', 'stop_margin': 'stop_margin',
    },
    'safety_node': {
        'drive_speed': 'max_speed', 'ttc_margin': 'ttc_margin', 'a_brake': 'a_brake',
        'stop_margin': 'stop_margin', 'track_obstacles': 'track_obstacles',
    },
    'lap_counter_node': {
        'num_laps': 'num_laps', 'similarity_threshold': 'similarity_threshold',
        'rearm_threshold': 'rearm_threshold', 'min_lap_distance': 'min_lap_distance',
    },
    'fusion_node': {
        'cam_x': 'cam_x', 'cam_yaw_deg': 'cam_yaw_deg', 'cam_z': 'cam_z',
    },
}


def load_parameter_file(path: str) -> Dict[str, dict]:
    """
    Read a ROS 2 parameter file.

    Args:
        path (str): Parameter file in the ros__parameters layout.

    Returns:
        dict: {section: parameters}, where the sections are '/**' (shared by every node)
        and the node names.
    """
    with open(path, encoding='utf-8') as stream:
        document = yaml.safe_load(stream) or {}
    # strip the leading '/' from node sections; empty sections become empty dicts
    return {(name if name == '/**' else name.lstrip('/')):
            dict((section or {}).get('ros__parameters') or {})
            for name, section in document.items()}


def overrides_for(node: str, context: LaunchContext) -> dict:
    """
    Collect the parameters a node gets from launch arguments given on the command line.

    Args:
        node (str): Node name, a key of OVERRIDES.
        context (LaunchContext): Launch context holding the argument values.

    Returns:
        dict: {parameter: typed value} for every given argument; empty if none were given.
    """
    # only arguments given on the command line (non-empty) override, cast to their type
    casts = {name: cast for name, _, cast in TUNING_ARGS}
    params = {}
    for param, arg in OVERRIDES.get(node, {}).items():
        value = LaunchConfiguration(arg).perform(context).strip()
        if value:
            params[param] = casts[arg](value)
    return params


def launch_nodes(context: LaunchContext) -> list:
    """
    Create the nodes once the launch arguments are known.

    Each node gets one set of parameters, merged here in this order (later wins): the
    file's shared /** section, the node's own section, the sensor switches, then the
    command-line overrides. Passing the file and the overrides separately does not work:
    launch writes the overrides under /** (our nodes have no namespace), and ROS applies a
    node's own section of the file after /**, so max_speed:=1.4 was replaced by the file's
    drive_speed: 1.0 in safety_node (and likewise every argument whose value sits in a
    node's own section).

    Args:
        context (LaunchContext): Launch context holding the argument values.

    Returns:
        list: The RealSense include and the Milestone 4 nodes.
    """
    # resolve the launch arguments to plain values
    sections = load_parameter_file(LaunchConfiguration('params_file').perform(context))
    use_camera = as_bool(LaunchConfiguration('use_camera').perform(context))
    use_depth = as_bool(LaunchConfiguration('use_depth').perform(context))
    start_camera = as_bool(LaunchConfiguration('start_camera').perform(context))
    odom = [('ego_racecar/odom', LaunchConfiguration('odom_topic').perform(context))]

    def params(node: str, extra: dict | None = None) -> list:
        """
        Merge the file's shared and own sections, the sensor switches and the overrides.

        Args:
            node (str): Node name (its section in the parameter file).
            extra (dict | None): Sensor switches set by this launch file.

        Returns:
            list: One parameter dict, as Node(parameters=...) expects.
        """
        return [{**sections.get('/**', {}), **sections.get(node, {}), **(extra or {}),
                 **overrides_for(node, context)}]

    actions = []

    # Sensors ------------------------------------------------------------------------------------

    # RealSense driver (Vision intro, slide 6), only if start_camera:=true.
    if use_camera and start_camera:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare('realsense2_camera'),
                                      'launch', 'rs_launch.py'])),
            launch_arguments={'enable_color': 'true', 'enable_depth': 'true'}.items(),
        ))

    if use_camera:
        # Colour image -> /camera_box_mask (cardboard pixels, for fusion_node). It also makes
        # /camera_scan (walls + boxes) and /camera_box_scan for rviz, while they are viewed.
        # Track walls are matte black, which absorbs infrared, so walls are found by colour.
        actions.append(Node(
            package='milestone4', executable='camera_scan_node', name='camera_scan_node',
            output='screen', parameters=params('camera_scan_node'),
        ))

    if use_camera and use_depth:
        # Depth image -> /depth_scan (floor rejected): boxes and low obstacles. Off by default,
        # which also saves the Jetson's CPU.
        actions.append(Node(
            package='milestone4', executable='depth_scan_node', name='depth_scan_node',
            output='screen', parameters=params('depth_scan_node'),
        ))

    # Fusion -------------------------------------------------------------------------------------

    # /scan + /camera_box_mask + /depth_scan -> /fused_scan and /obstacle_scan: each LiDAR
    # return is projected into the box mask, so the LiDAR gives range and the camera class.
    actions.append(Node(
        package='milestone4', executable='fusion_node', name='fusion_node', output='screen',
        parameters=params('fusion_node', {'use_camera': use_camera, 'use_depth': use_depth}),
        remappings=odom,
    ))

    # Planning and safety ------------------------------------------------------------------------

    # Disparity extender on /fused_scan (raw /scan if fused stops) -> /drive_request.
    # It shares the braking model with safety_node (the /** section of the YAML file).
    actions.append(Node(
        package='milestone4', executable='gap_follow_node', name='gap_follow_node',
        output='screen', parameters=params('gap_follow_node'), remappings=odom,
    ))

    # Path-aware AEB with a variable TTC + watchdog: the only node publishing /drive.
    actions.append(Node(
        package='milestone4', executable='safety_node', name='safety_node', output='screen',
        parameters=params('safety_node'), remappings=odom,
    ))

    # Camera + LiDAR + odometry lap counter -> latched stop after num_laps.
    actions.append(Node(
        package='milestone4', executable='lap_counter_node', name='lap_counter_node',
        output='screen', parameters=params('lap_counter_node', {'use_camera': use_camera}),
        remappings=odom,
    ))

    return actions


def generate_launch_description() -> LaunchDescription:
    """
    Declare the launch arguments and defer node creation to launch_nodes().

    Returns:
        LaunchDescription: Launch arguments and the Milestone 4 nodes.
    """
    args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([FindPackageShare('milestone4'), 'config',
                                                'milestone4.yaml']),
            description='Parameter file for all nodes.'),
        # Sensor switches: these choose which nodes start, so they keep real defaults here.
        DeclareLaunchArgument('use_camera', default_value='true',
                              description='false: LiDAR only (camera nodes not started).'),
        DeclareLaunchArgument('use_depth', default_value='false',
                              description='true: also fuse the depth camera (check the floor '
                                          'first).'),
        DeclareLaunchArgument('start_camera', default_value='false',
                              description='true: also start the RealSense driver.'),
        DeclareLaunchArgument('odom_topic', default_value='/odom',
                              description='Odometry topic (/ego_racecar/odom in the sim).'),
    ]
    # tuning arguments default to '' so that the parameter file decides
    args += [DeclareLaunchArgument(name, default_value='',
                                   description=f'{text} Default: params_file.')
             for name, text, _ in TUNING_ARGS]

    return LaunchDescription(args + [OpaqueFunction(function=launch_nodes)])
