"""
Debug overlay for the colour scan, viewed live with rqt_image_view (no ROS code).

draw_debug() tints masked pixels red, draws the horizon (yellow), the boundary hits
(green) and a top-down inset of the scan.
"""

import math

import cv2
from milestone4.boundary import ScanResult
from milestone4.camera_model import CameraModel
import numpy as np


def draw_debug(bgr: np.ndarray, result: ScanResult, cam: CameraModel,
               range_max: float) -> np.ndarray:
    """Draw the mask, detected boundary and a top-down scan view on a frame.

    Args:
        bgr (np.ndarray): Original BGR colour image.
        result (ScanResult): Scan output for this frame.
        cam (CameraModel): Colour camera model (for the horizon line).
        range_max (float): Maximum range, used to scale the top-down view.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    out = bgr.copy()

    # Tint detected boundary/obstacle pixels red.
    red = np.zeros_like(out)
    red[:, :, 2] = 255
    m = result.mask > 0
    out[m] = cv2.addWeighted(out, 0.5, red, 0.5, 0)[m]

    # Horizon line (yellow): nothing above it is used by the colour scan.
    hz = int(round(cam.horizon_row()))
    if 0 <= hz < out.shape[0]:
        cv2.line(out, (0, hz), (out.shape[1] - 1, hz), (0, 255, 255), 1)

    # First hit per column (green): where the boundary meets the floor.
    for u in range(0, result.boundary_rows.size, 4):
        v = int(result.boundary_rows[u])
        if v >= 0:
            cv2.circle(out, (u, v), 1, (0, 255, 0), -1)

    # Top-down inset of the scan, car at the bottom centre, forward = up.
    size = 160
    inset = np.full((size, size, 3), 30, np.uint8)
    scale = (size - 10) / range_max
    origin = (size // 2, size - 5)
    for i, r in enumerate(result.ranges):
        if r >= range_max:
            continue
        a = result.angle_min + i * result.angle_increment
        x = int(origin[0] - r * math.sin(a) * scale)
        y = int(origin[1] - r * math.cos(a) * scale)
        cv2.circle(inset, (x, y), 1, (0, 255, 0), -1)
    cv2.circle(inset, origin, 3, (255, 255, 255), -1)
    out[5:5 + size, out.shape[1] - size - 5:out.shape[1] - 5] = inset

    # Closest range straight ahead (+/- 10 degrees), useful when tuning.
    centre = np.abs(result.angle_min
                    + np.arange(result.ranges.size) * result.angle_increment) <= math.radians(10)
    ahead = float(np.min(result.ranges[centre])) if centre.any() else range_max
    cv2.putText(out, f'ahead: {ahead:.2f} m', (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    return out
