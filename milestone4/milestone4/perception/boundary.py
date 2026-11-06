"""
Colour image -> LaserScan-style ranges.

For each image column, find the lowest obstacle pixel below the horizon. This shows where 
the nearest object meets the floor. The pixel's row determines the distance, while its column 
determines the bearing. Grouping these measurements by bearing produces LiDAR-like ranges.
"""

from dataclasses import dataclass
import numpy as np
from milestone4.perception.camera_model import CameraModel, ground_points
from milestone4.perception.masks import MaskParams, obstacle_mask


@dataclass
class ScanResult:
    """Output of the colour scan for one frame.

    Attributes:
        ranges (np.ndarray): Range per bearing, in metres (float32); range_max
            where nothing was detected.
        angle_min (float): Bearing of ranges[0], in radians (right edge).
        angle_increment (float): Spacing between bearings, in radians.
        boundary_rows (np.ndarray): Per image column, the row of the first
            boundary/obstacle pixel from the bottom, or -1 if none.
        mask (np.ndarray): Binary boundary/obstacle mask (uint8, 0 or 255).
    """

    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    boundary_rows: np.ndarray
    mask: np.ndarray

def colour_scan(bgr: np.ndarray, cam: CameraModel, mask_params: MaskParams,
                num_beams: int, range_max: float,
                ignore_bottom_rows: int = 0) -> ScanResult:
    """
    Turns one camera image into a lidar-style scan. Finds the closest obstacle in each direction the
    camera can see; returns its distance so lane follow node can use the camera info like lidar.

    Args:
        bgr (np.ndarray): Colour image from camera (BGR).
        cam (CameraModel): cam settings.
        mask_params (MaskParams): Colour thresholds for walls and boxes.
        num_beams (int): Num of directions the scan has.
        range_max (float): Distance reported when nothing is detected (m).
        ignore_bottom_rows (int): Rows at the bottom to skip (the car body).

    Returns:
        ScanResult: Ranges, scan geometry and the mask/boundary for debugging.
    """
    angle_min, angle_max = cam.bearing_limits()
    num_beams = np.maximum(2, int(num_beams)) # need at least 2 beams
    angle_increment = (angle_max - angle_min) / (num_beams - 1) # gap between beams
    ranges = np.full(num_beams, range_max, dtype=np.float32) # start staus 

    # finds the closest obstacle pixel in each image column
    mask = obstacle_mask(bgr, mask_params)
    hit_rows = _first_hit_rows(mask, cam.horizon_row(), ignore_bottom_rows)

    # turn pixels from each col into a point on the floor (meters)
    cols = np.arange(len(hit_rows))
    forward, lateral, valid = ground_points(cols, hit_rows, cam)

    # filter columns so only ones with usable points are kept
    kept_cols = []
    for col in cols:
        if hit_rows[col] < 0:
            continue # no obstacle found in this col
        if not valid[col]:
            continue # pixel never reaches the floor
        if np.hypot(forward[col], lateral[col]) > range_max:
            continue #too far away
        kept_cols.append(col)
        # put each point in its beam, keeping nearest
    _bin_points(forward[kept_cols],lateral[kept_cols], ranges, angle_min, angle_increment)

    return ScanResult(ranges=ranges, angle_min=angle_min, angle_increment=angle_increment, boundary_rows=hit_rows, mask=mask)

def _first_hit_rows(mask: np.ndarray, horizon_row: float, ignore_bottom_rows: int = 0) -> np.ndarray:
    """
    For each column, find the lowest obstacle pixel below the horizon.

    Lower in the image means closer to the car, so the lowest obstacle pixel
    in a column is the closest obstacle in that direction. Rows above the
    horizon and rows showing the car itself are skipped.

    Args:
        mask (np.ndarray): B&W image; non-zero = obstacle
        horizon_row (float): Row of the horizon; only rows below this row are considered.
        ignore_bottom_rows (int): Rows at the bottom to skip (the car body).

    Returns:
        np.ndarray: One row num per col; -1 if nothing found
    """
    height, width = mask.shape
    row_top = np.maximum(0, int(np.floor(horizon_row)) + 1) # first row below the horizon
    row_bot = height - max(0, ignore_bottom_rows) # first row of the car body
    hits = np.full(width, -1, dtype=np.int32) # -1 = no obstacle found in direction.
    if row_top >= row_bot:
        return hits
    for col in range(width):
        obstacle_rows = np.nonzero(mask[row_top:row_bot, col])[0]
        if len(obstacle_rows) > 0:
            hits[col] = row_top + obstacle_rows[-1]
    return hits



def _bin_points(forward: np.ndarray, lateral: np.ndarray, ranges: np.ndarray, angle_min: float, angle_increment: float) -> None:
    """
    Fill in the ranges with the nearest point in each direction. Each beam is 
    one direction the camera can see. Several points can fall in the same beam; 
    only the closest one is kept.

    Args:
        forward (np.ndarray): Forward distances, in metres.
        lateral (np.ndarray): Lateral offsets (positive = right), in metres.
        ranges (np.ndarray): Ranges array to update in place.
        angle_min (float): Bearing of ranges[0], in radians.
        angle_increment (float): Spacing between bearings, in radians.
    """
    for i in range(len(forward)):
        dist = np.hypot(forward[i], lateral[i])  # distance to the point
        bearing = np.arctan2(-lateral[i], forward[i])  # direction, left positive
        beam = int(np.round((bearing - angle_min) / angle_increment))  # nearest beam
        if 0 <= beam < len(ranges):  # skip points just outside the view
            ranges[beam] = min(ranges[beam], dist)  # keep the closest per beam