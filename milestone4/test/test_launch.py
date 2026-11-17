"""Check what each node gets from launch/milestone4_py.py, with launch stubbed out."""

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.dirname(HERE)
LAUNCH_FILE = os.path.join(PACKAGE, 'launch', 'milestone4_py.py')
PARAMS_FILE = os.path.join(PACKAGE, 'config', 'milestone4.yaml')


class Recorded:
    """Stands in for every launch action: keeps its arguments."""

    def __init__(self, *args, **kwargs) -> None:
        """Keep the positional and keyword arguments for inspection."""
        self.args, self.kwargs = args, kwargs


class LaunchConfiguration:
    """launch.substitutions.LaunchConfiguration that reads the fake context."""

    def __init__(self, name: str) -> None:
        """Store the launch argument name."""
        self.name = name

    def perform(self, context) -> str:
        """Return the argument's value from the fake context."""
        return context.launch_configurations[self.name]


def load_launch_module() -> types.ModuleType:
    """Import the launch file with launch and launch_ros replaced by stand-ins."""
    # fake modules exposing only the names the launch file imports
    stubs = {}
    for name, attrs in {
        'launch': {'LaunchContext': object, 'LaunchDescription': Recorded},
        'launch.actions': {'DeclareLaunchArgument': Recorded, 'IncludeLaunchDescription': Recorded,
                           'OpaqueFunction': Recorded},
        'launch.launch_description_sources': {'PythonLaunchDescriptionSource': Recorded},
        'launch.substitutions': {'LaunchConfiguration': LaunchConfiguration,
                                 'PathJoinSubstitution': Recorded},
        'launch_ros': {},
        'launch_ros.actions': {'Node': Recorded},
        'launch_ros.substitutions': {'FindPackageShare': Recorded},
    }.items():
        stubs[name] = types.ModuleType(name)
        stubs[name].__dict__.update(attrs)
    # import the launch file by path while the stubs are installed
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location('milestone4_launch', LAUNCH_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


LAUNCH = load_launch_module()


class TestLaunchParameters(unittest.TestCase):
    """Launch arguments must reach the nodes, whatever section of the file they are in."""

    def nodes(self, **arguments) -> dict:
        """Run launch_nodes with these command-line arguments; return {node: parameters}."""
        # every launch argument at its default, then the given overrides
        config = {'params_file': PARAMS_FILE, 'use_camera': 'true', 'use_depth': 'false',
                  'start_camera': 'false', 'odom_topic': '/odom'}
        config.update({name: '' for name, _, _ in LAUNCH.TUNING_ARGS})
        config.update(arguments)
        context = types.SimpleNamespace(launch_configurations=config)
        found = {}
        for action in LAUNCH.launch_nodes(context):
            if 'executable' in action.kwargs:
                parameters = action.kwargs['parameters']
                # one merged set: a file passed next to it would win over the overrides
                self.assertEqual(len(parameters), 1, action.kwargs['name'])
                self.assertIsInstance(parameters[0], dict)
                found[action.kwargs['name']] = parameters[0]
        return found

    def test_max_speed_sets_the_safety_cap_too(self) -> None:
        """Check max_speed sets the planner's speed and the safety cap."""
        nodes = self.nodes(max_speed='1.4')
        self.assertEqual(nodes['gap_follow_node']['max_speed'], 1.4)
        self.assertEqual(nodes['safety_node']['max_speed'], 1.4)

    def test_file_values_without_arguments(self) -> None:
        """Check the YAML values (own and shared sections) apply with no arguments."""
        nodes = self.nodes()
        self.assertEqual(nodes['gap_follow_node']['max_speed'], 1.5)
        self.assertEqual(nodes['safety_node']['max_speed'], 1.5)
        self.assertEqual(nodes['gap_follow_node']['pass_speed'], 1.2)   # own section
        for node in ('gap_follow_node', 'safety_node'):
            self.assertEqual(nodes[node]['a_brake'], 2.0)      # the shared /** section
            self.assertEqual(nodes[node]['wheelbase'], 0.33)

    def test_arguments_in_any_section_win(self) -> None:
        """Check command-line arguments override shared and own YAML sections."""
        nodes = self.nodes(a_brake='2.5', num_laps='2', pass_speed='0.6', cam_x='0.05',
                           track_obstacles='true')
        self.assertEqual(nodes['gap_follow_node']['a_brake'], 2.5)   # shared section
        self.assertEqual(nodes['safety_node']['a_brake'], 2.5)
        self.assertEqual(nodes['lap_counter_node']['num_laps'], 2)   # own sections
        self.assertEqual(nodes['gap_follow_node']['pass_speed'], 0.6)
        self.assertEqual(nodes['fusion_node']['cam_x'], 0.05)
        self.assertIs(nodes['safety_node']['track_obstacles'], True)

    def test_sensor_switches(self) -> None:
        """Check use_camera:=false drops the camera node and reaches the others."""
        nodes = self.nodes(use_camera='false')
        self.assertNotIn('camera_scan_node', nodes)
        self.assertIs(nodes['fusion_node']['use_camera'], False)
        self.assertIs(nodes['lap_counter_node']['use_camera'], False)


if __name__ == '__main__':
    unittest.main()
