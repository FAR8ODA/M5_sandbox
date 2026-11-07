"""
Launch file for Milestone 4: LiDAR-primary driving with camera fusion and AEB.

On the car (after starting the F1Tenth stack and, unless start_camera:=true, the camera):
    ros2 launch milestone4 milestone4_py.py
    ros2 launch milestone4 milestone4_py.py max_speed:=1.5 a_brake:=2.5
    ros2 launch milestone4 milestone4_py.py use_camera:=false        # LiDAR only (or sim)

In the simulator:
    ros2 launch milestone4 milestone4_py.py use_camera:=false odom_topic:=/ego_racecar/odom

Only safety_node publishes to /drive. Every other node sends requests on /drive_request.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def as_float(name: str) -> PythonExpression:
    """
    Read a launch argument as a float, so max_speed:=2 still gives a float parameter.

    Args:
        name (str): Launch argument name.

    Returns:
        PythonExpression: Substitution that evaluates to the argument as a float.
    """
    return PythonExpression(['float(', LaunchConfiguration(name), ')'])


def generate_launch_description() -> LaunchDescription:
    """
    Milestone 4 Launch File with Args.

    Returns:
        LaunchDescription: Launch arguments and the Milestone 4 nodes.
    """
    use_camera = LaunchConfiguration('use_camera')
    odom_topic = LaunchConfiguration('odom_topic')

    args = [
        # Driving
        DeclareLaunchArgument('max_speed', default_value='1.0',
                              description='Highest speed on a clear straight (m/s).'),
        DeclareLaunchArgument('pass_speed', default_value='0.55',
                              description='Speed cap while passing a camera-confirmed box (m/s).'),
        DeclareLaunchArgument('lookahead', default_value='2.0',
                              description='Longest pure-pursuit distance (m).'),
        DeclareLaunchArgument('gain', default_value='1.0',
                              description='Pure-pursuit steering gain.'),
        DeclareLaunchArgument('a_lat', default_value='2.5',
                              description='Sideways grip for the cornering speed (m/s^2).'),
        DeclareLaunchArgument('margin', default_value='0.20',
                              description='Clearance kept at obstacle and barrier edges (m).'),
        DeclareLaunchArgument('smoothing', default_value='0.6',
                              description='Weight of the previous steering command (0-1).'),
        DeclareLaunchArgument('deep_fraction', default_value='0.95',
                              description='Beams this close to the deepest count as the gap; '
                                          'lower aims further from inner walls but can aim at '
                                          'a near obstacle when the view ahead is short (0-1).'),
        DeclareLaunchArgument('lidar_x', default_value='0.27',
                              description='LiDAR distance ahead of the rear axle, for pure '
                                          'pursuit (m); 0 treats the LiDAR as the rear axle.'),
        # Safety
        DeclareLaunchArgument('ttc_margin', default_value='0.1',
                              description='Extra time added to the variable TTC threshold (s).'),
        DeclareLaunchArgument('a_brake', default_value='2.0',
                              description='Measured braking deceleration (m/s^2).'),
        DeclareLaunchArgument('stop_margin', default_value='0.25',
                              description='Gap AEB leaves in front of the LiDAR (m).'),
        DeclareLaunchArgument('track_obstacles', default_value='false',
                              description='true: AEB also uses obstacle speeds (moving ones).'),
        # Lap counter
        DeclareLaunchArgument('num_laps', default_value='3',
                              description='Laps before the latched stop.'),
        DeclareLaunchArgument('similarity_threshold', default_value='0.8',
                              description='Lap score meaning "back at the start".'),
        DeclareLaunchArgument('rearm_threshold', default_value='0.5',
                              description='Lap score must drop below this before the next lap.'),
        DeclareLaunchArgument('min_lap_distance', default_value='15.0',
                              description='Shortest possible lap (m); about 3/4 of a lap '
                                          '(15.0 for a 20 m lap).'),
        # Sensors
        DeclareLaunchArgument('use_camera', default_value='true',
                              description='false: LiDAR only (camera nodes not started).'),
        DeclareLaunchArgument('use_depth', default_value='false',
                              description='true: also fuse the depth camera (check the floor '
                                          'first).'),
        DeclareLaunchArgument('start_camera', default_value='false',
                              description='true: also start the RealSense driver.'),
        DeclareLaunchArgument('odom_topic', default_value='/odom',
                              description='Odometry topic (/ego_racecar/odom in the sim).'),
        DeclareLaunchArgument('cam_x', default_value='0.0',
                              description='Camera position ahead of the LiDAR (m).'),
        DeclareLaunchArgument('cam_yaw_deg', default_value='0.0',
                              description='Camera heading relative to the LiDAR (degrees).'),
    ]

    # Sensors ------------------------------------------------------------------------------------

    # RealSense driver (Vision intro, slide 6), only if start_camera:=true.
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('realsense2_camera'),
                                  'launch', 'rs_launch.py'])),
        launch_arguments={'enable_color': 'true', 'enable_depth': 'true'}.items(),
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('start_camera'), "' == 'true' and '", use_camera,
            "' == 'true'"])),
    )

    # Colour image -> /camera_scan (walls + boxes) and /camera_box_scan (boxes only).
    # Track walls are matte black, which absorbs infrared, so walls are found by colour.
    camera_scan_node = Node(
        package='milestone4', executable='camera_scan_node', name='camera_scan_node',
        output='screen',
        parameters=[{'mount_height': 0.187, 'pitch_deg': -5.3, 'ignore_bottom_rows': 30}],
        condition=IfCondition(use_camera),
    )

    # Depth image -> /depth_scan (floor rejected): boxes and low obstacles. Only started with
    # use_depth:=true (and the camera on), which also saves the Jetson's CPU.
    depth_scan_node = Node(
        package='milestone4', executable='depth_scan_node', name='depth_scan_node',
        output='screen',
        parameters=[{'mount_height': 0.187, 'pitch_deg': -5.3}],
        condition=IfCondition(PythonExpression([
            "'", use_camera, "' == 'true' and '", LaunchConfiguration('use_depth'),
            "' == 'true'"])),
    )

    # Fusion -------------------------------------------------------------------------------------

    # /scan + /camera_box_scan + /depth_scan -> /fused_scan and /obstacle_scan.
    fusion_node = Node(
        package='milestone4', executable='fusion_node', name='fusion_node',
        output='screen',
        parameters=[{'use_camera': use_camera,
                     'use_depth': LaunchConfiguration('use_depth'),
                     'cam_x': as_float('cam_x'),
                     'cam_yaw_deg': as_float('cam_yaw_deg')}],
        remappings=[('ego_racecar/odom', odom_topic)],
    )

    # Planning and safety ------------------------------------------------------------------------

    # Disparity extender on /fused_scan (raw /scan if fused stops) -> /drive_request.
    # It shares the braking model (a_brake, ttc_margin, stop_margin) with safety_node.
    gap_follow_node = Node(
        package='milestone4', executable='gap_follow_node', name='gap_follow_node',
        output='screen',
        parameters=[{'max_speed': as_float('max_speed'),
                     'pass_speed': as_float('pass_speed'),
                     'lookahead': as_float('lookahead'),
                     'gain': as_float('gain'),
                     'a_lat': as_float('a_lat'),
                     'margin': as_float('margin'),
                     'smoothing': as_float('smoothing'),
                     'deep_fraction': as_float('deep_fraction'),
                     'lidar_x': as_float('lidar_x'),
                     'a_brake': as_float('a_brake'),
                     'ttc_margin': as_float('ttc_margin'),
                     'stop_margin': as_float('stop_margin')}],
        remappings=[('ego_racecar/odom', odom_topic)],
    )

    # Path-aware AEB with a variable TTC + watchdog: the only node publishing /drive.
    safety_node = Node(
        package='milestone4', executable='safety_node', name='safety_node',
        output='screen',
        parameters=[{'drive_speed': as_float('max_speed'),
                     'ttc_margin': as_float('ttc_margin'),
                     'a_brake': as_float('a_brake'),
                     'stop_margin': as_float('stop_margin'),
                     'track_obstacles': LaunchConfiguration('track_obstacles')}],
        remappings=[('ego_racecar/odom', odom_topic)],
    )

    # Camera + LiDAR + odometry lap counter -> latched stop after num_laps.
    lap_counter_node = Node(
        package='milestone4', executable='lap_counter_node', name='lap_counter_node',
        output='screen',
        parameters=[{'num_laps': LaunchConfiguration('num_laps'),
                     'use_camera': use_camera,
                     'similarity_threshold': as_float('similarity_threshold'),
                     'rearm_threshold': as_float('rearm_threshold'),
                     'min_lap_distance': as_float('min_lap_distance')}],
        remappings=[('ego_racecar/odom', odom_topic)],
    )

    return LaunchDescription(args + [realsense, camera_scan_node, depth_scan_node, fusion_node,
                                     gap_follow_node, safety_node, lap_counter_node])
