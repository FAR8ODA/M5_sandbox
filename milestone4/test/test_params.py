"""
Check config/milestone4.yaml against the parameters the nodes declare (no ROS).

Catches misspelt names (silently ignored by ROS) and wrongly typed values.
"""

import ast
import os
import unittest

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.dirname(HERE)
PARAMS_FILE = os.path.join(PACKAGE, 'config', 'milestone4.yaml')
NODES = ['camera_scan_node', 'depth_scan_node', 'fusion_node', 'gap_follow_node',
         'safety_node', 'lap_counter_node']


def declared_parameters(node: str) -> dict:
    """
    Read the declare_parameter calls of a node's source file.

    Args:
        node (str): Node module name.

    Returns:
        dict: {name: default} where the default is a literal, or None where it is an
        expression (a class constant or a dataclass default).
    """
    with open(os.path.join(PACKAGE, 'milestone4', node + '.py'), encoding='utf-8') as f:
        tree = ast.parse(f.read())
    # every self.declare_parameter(name, default) call with a literal name
    found = {}
    for call in ast.walk(tree):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'declare_parameter' and len(call.args) >= 2):
            name = ast.literal_eval(call.args[0])
            try:
                found[name] = ast.literal_eval(call.args[1])
            except ValueError:
                found[name] = None
    return found


class TestParamsFile(unittest.TestCase):
    """The YAML file only sets declared parameters, with the declared types."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the YAML file and every node's declared parameters."""
        with open(PARAMS_FILE, encoding='utf-8') as f:
            cls.config = yaml.safe_load(f)
        cls.declared = {node: declared_parameters(node) for node in NODES}

    def section(self, key: str) -> dict:
        """Return the ros__parameters of one section of the YAML file."""
        self.assertIn(key, self.config, f'{key} missing from the params file')
        return self.config[key]['ros__parameters']

    def check_type(self, where: str, name: str, value, default) -> None:
        """Fail if a YAML value would be rejected for the declared default's type."""
        if default is None:
            return
        if isinstance(default, bool) or isinstance(value, bool):
            self.assertIsInstance(value, bool, f'{where}: {name} must be true or false')
            self.assertIsInstance(default, bool, f'{where}: {name} must be a number')
        elif isinstance(default, float):
            self.assertIsInstance(value, float,
                                  f'{where}: {name} needs a decimal point ({value})')
        else:
            self.assertIs(type(value), type(default), f'{where}: {name} has the wrong type')

    def test_only_known_sections(self) -> None:
        """Every section is a node of this package or the shared /** section."""
        for key in self.config:
            self.assertIn(key, NODES + ['/**'], f'unknown section {key}')

    def test_node_sections_are_declared(self) -> None:
        """Every per-node value is a parameter that node declares, with its type."""
        for node in NODES:
            for name, value in self.section(node).items():
                self.assertIn(name, self.declared[node], f'{node} does not declare {name}')
                self.check_type(node, name, value, self.declared[node][name])

    def test_shared_values_are_used(self) -> None:
        """Every shared value is declared by at least one node, with its type."""
        for name, value in self.section('/**').items():
            users = [node for node in NODES if name in self.declared[node]]
            self.assertTrue(users, f'no node declares the shared parameter {name}')
            for node in users:
                self.check_type(node, name, value, self.declared[node][name])

    def test_shared_values_not_repeated(self) -> None:
        """A shared value is never set again per node, where it could drift apart."""
        shared = set(self.section('/**'))
        for node in NODES:
            repeated = shared & set(self.section(node))
            self.assertFalse(repeated, f'{node} repeats shared values {sorted(repeated)}')

    def test_every_parameter_is_in_the_file(self) -> None:
        """Every declared parameter is tuned from the file (sensor switches excepted)."""
        launch_set = {'use_camera', 'use_depth'}      # set from the launch arguments
        shared = set(self.section('/**'))
        for node in NODES:
            missing = (set(self.declared[node]) - set(self.section(node)) - shared
                       - launch_set)
            self.assertFalse(missing, f'{node} parameters missing: {sorted(missing)}')

    def test_speed_caps_match(self) -> None:
        """The planner's top speed and the safety node's cap are the same."""
        self.assertEqual(self.section('gap_follow_node')['max_speed'],
                         self.section('safety_node')['drive_speed'])


if __name__ == '__main__':
    unittest.main()
