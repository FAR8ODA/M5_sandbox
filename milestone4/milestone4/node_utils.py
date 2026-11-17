"""Helpers shared by the ROS nodes: clock arithmetic, header stamps and the entry point."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time


def stamp_seconds(stamp) -> float:
    """
    Convert a message header stamp to seconds.

    Args:
        stamp: builtin_interfaces/Time with sec and nanosec fields.

    Returns:
        float: The stamp in seconds.
    """
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def now_seconds(node: Node) -> float:
    """
    Read a node's clock in seconds.

    Args:
        node (Node): Node whose clock is read (the sim clock when use_sim_time is set).

    Returns:
        float: Current time (s).
    """
    return node.get_clock().now().nanoseconds * 1e-9


def seconds_since(node: Node, stamp: Time | None) -> float:
    """
    Work out how long ago something happened, by a node's clock.

    Args:
        node (Node): Node whose clock is read.
        stamp (Time | None): When it happened, or None if it never did.

    Returns:
        float: Age in seconds, or inf if stamp is None, so that "never" is always stale.
    """
    if stamp is None:
        return math.inf
    return (node.get_clock().now() - stamp).nanoseconds * 1e-9


def run_node(node_class: type) -> None:
    """
    Start ROS, spin one node until Ctrl+C, then shut down cleanly.

    This is the console entry point of every node except safety_node, which also has to
    send a final stop command on the way out.

    Args:
        node_class (type): The Node subclass to create and spin.
    """
    rclpy.init()
    node = node_class()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass                    # Ctrl+C is the normal way to stop
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()        # skip if Ctrl+C already shut ROS down
