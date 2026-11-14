"""
Text for the debug topics (no ROS code).

gap_follow_node and safety_node each publish one line of JSON per decision on
gap_follow/debug and safety/debug (std_msgs/String), so a run can be recorded with
ros2 bag and read afterwards without any custom message type.
"""

from __future__ import annotations

import json
import math


def encode(**fields) -> str:
    """
    Return the fields as one line of JSON.

    Numbers are rounded to 3 decimals; inf and NaN (e.g. "nothing on the path") become
    null, since JSON has no way to write them.

    Args:
        **fields: Name and value of each field (number, bool, str or None).

    Returns:
        str: The JSON text, e.g. '{"t":12.35,"free":null,"limit":"max"}'.
    """
    def clean(value):
        if value is None or isinstance(value, (bool, str, int)):
            return value
        value = float(value)
        return round(value, 3) if math.isfinite(value) else None

    return json.dumps({name: clean(value) for name, value in fields.items()},
                      separators=(',', ':'))
