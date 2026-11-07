"""
Launch file for Milestone 4: camera-only reactive driving with AEB.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """
    Milestone 4 Launch File with Args.

    Returns:
        LaunchDescription: Launch arguments and the Milestone 4 nodes.
    """
    driver = LaunchConfiguration('driver')
    drive_speed = LaunchConfiguration('drive_speed')
    args = [
        DeclareLaunchArgument('num_laps', default_value='3'),
        DeclareLaunchArgument('driver', default_value='gap'),
        DeclareLaunchArgument('drive_speed', default_value='0.7'),
        DeclareLaunchArgument('ttc_threshold', default_value='0.65'),
        # Scan used for AEB: the LiDAR by default (sees the black walls the depth
        # camera misses); 'depth_scan' brings back the Milestone 3 depth-based AEB.
        DeclareLaunchArgument('aeb_scan_topic', default_value='/scan'),
        DeclareLaunchArgument('lookahead', default_value='1.0'),
        DeclareLaunchArgument('gain', default_value='1.5'),
        DeclareLaunchArgument('similarity_threshold', default_value='0.8'),
        DeclareLaunchArgument('rearm_threshold', default_value='0.5'),
    ]

    # Milestone 2 Nodes ----------------------------------------------------------------------------

    # Milestone 2 wall-follow on the camera scan.
    wall_follow_node = Node(
        package='milestone4', executable='wall_follow_node', name='wall_follow_node',
        output='screen',
        remappings=[('scan', 'camera_scan')],
        condition=IfCondition(PythonExpression(["'", driver, "' == 'wall'"])),
    )

    # Milestone 2 gap-follow on the camera scan.
    gap_follow_node = Node(
        package='milestone4', executable='gap_follow_node', name='gap_follow_node',
        output='screen',
        remappings=[('scan', 'camera_scan')],
        condition=IfCondition(PythonExpression(["'", driver, "' == 'gap'"])),
    )


    # CV Nodes -------------------------------------------------------------------------------------

    # Colour image -> LaserScan 
    # Track walls are matte black which abosrbs infrared light.
    # So walls are found by colour, using the RGB image.
    camera_scan_node = Node(
        package='milestone4', executable='camera_scan_node', name='camera_scan_node',
        output='screen',
        parameters=[{'mount_height': 0.187, 'pitch_deg': -5.3, 'ignore_bottom_rows': 30}],
    )

    # Depth image -> LaserScan (AEB with aeb_scan_topic:=depth_scan; LiDAR by default)
    depth_scan_node = Node(
        package='milestone4', executable='depth_scan_node', name='depth_scan_node',
        output='screen',
    )


    # Milestone 4 Nodes ----------------------------------------------------------------------------

    # Milestone 4 Safety (AEB + watchdog)
    # forwards drive_request to /drive at a capped speed.
    # AEB runs on the LiDAR scan (/scan) by default, see aeb_scan_topic.
    safety_node = Node(
        package='milestone4', executable='safety_node', name='safety_node',
        output='screen',
        parameters=[{'drive_speed': drive_speed,
                     'ttc_threshold': LaunchConfiguration('ttc_threshold')}],
        remappings=[('scan', LaunchConfiguration('aeb_scan_topic')),
                    ('ego_racecar/odom', '/odom')],
    )

    # Milestone 4: Lane-Follow on the camera scan.
    lane_follow_node = Node(
        package='milestone4', executable='lane_follow_node', name='lane_follow_node',
        output='screen',
        remappings=[('scan', 'camera_scan')],
        parameters=[{'max_speed': drive_speed, 'lookahead': LaunchConfiguration('lookahead'),
                     'gain': LaunchConfiguration('gain')}],
        condition=IfCondition(PythonExpression(["'", driver, "' == 'lane'"])),
    )

    # Milestone 4: starting view (image similarity) -> lap count
    # after num_laps laps, a latched stop on drive_request.
    lap_counter_node = Node(
        package='milestone4', executable='lap_counter_node', name='lap_counter_node',
        output='screen',
        parameters=[{'num_laps': LaunchConfiguration('num_laps'),
                     'similarity_threshold': LaunchConfiguration('similarity_threshold'),
                     'rearm_threshold': LaunchConfiguration('rearm_threshold')}],
    )

    return LaunchDescription(args + [camera_scan_node, lane_follow_node, 
    wall_follow_node, gap_follow_node, depth_scan_node, lap_counter_node, safety_node])
