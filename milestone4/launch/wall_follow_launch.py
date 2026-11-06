from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        Node(
            package='milestone4',
            executable='safety_node',
            name='safety_node',
            output='screen',
        ),
        Node(
            package='milestone4',
            executable='wall_follow_node',
            name='wall_follow_node',
            output='screen',
        )
    ])