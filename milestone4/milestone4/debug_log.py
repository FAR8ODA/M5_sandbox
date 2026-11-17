"""
JSON encoding for the gap_follow/debug and safety/debug String topics (no ROS code).

One line per decision, so a run can be recorded with ros2 bag without custom messages.
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
        """Return value as a JSON-safe scalar: rounded float, or None for inf/NaN."""
        if value is None or isinstance(value, (bool, str, int)):
            return value
        value = float(value)
        return round(value, 3) if math.isfinite(value) else None

    return json.dumps({name: clean(value) for name, value in fields.items()},
                      separators=(',', ':'))
