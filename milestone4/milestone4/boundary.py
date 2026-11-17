"""
Colour mask -> LaserScan-style ranges (no ROS code).

The lowest masked pixel in each image column is where the nearest object meets the
floor: its row gives the distance and its column the bearing. Use mask_scan() for any
mask, or colour_scan() to mask and scan a BGR image in one call.
"""

from __future__ import annotations

from dataclasses import dataclass

from milestone4.camera_model import CameraModel, ground_points
from milestone4.masks import MaskParams, obstacle_mask
import numpy as np


@dataclass
class ScanResult:
    """Output of the colour scan for one frame.

    Attributes:
        ranges (np.ndarray): Range per bearing, in metres (float32); range_max
            where nothing was detected.
        angle_min (float): Bearing of ranges[0], in radians (right edge).
        angle_increment (float): Spacing between bearings, in radians.
        boundary_rows (np.ndarray): Per image column, the row of the first
            masked pixel from the bottom, or -1 if none.
        mask (np.ndarray): Binary mask the scan was built from (uint8, 0 or 255).
    """

    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    boundary_rows: np.ndarray
    mask: np.ndarray


def mask_scan(mask: np.ndarray, cam: CameraModel, num_beams: int, range_max: float,
              ignore_bottom_rows: int = 0, not_below: np.ndarray | None = None,
              contact_rows: int = 0) -> ScanResult:
    """
    Turn one binary mask into a LiDAR-style scan.

    Finds the closest masked pixel in each direction the camera can see and returns its
    floor distance, so the other nodes can treat the camera like a LiDAR.

    The floor distance is only right if the object really stands on the floor there. Tan
    desks and cardboard behind the track's black ducting have the colour of a box, but
    their lowest visible pixel sits on top of the ducting, not on the floor. With
    not_below, a column is dropped when a not_below pixel (the ducting) is within
    contact_rows rows under its lowest masked pixel.

    Args:
        mask (np.ndarray): HxW uint8 mask, non-zero where a wall or box is.
        cam (CameraModel): Camera settings.
        num_beams (int): Number of directions in the scan.
        range_max (float): Distance reported when nothing is detected (m).
        ignore_bottom_rows (int): Rows at the bottom to skip (the car body).
        not_below (np.ndarray | None): HxW mask of what an object must not stand on.
        contact_rows (int): Rows under the lowest pixel checked against not_below.

    Returns:
        ScanResult: Ranges, scan geometry and the mask/boundary for debugging.
    """
    angle_min, angle_max = cam.bearing_limits()
    num_beams = max(2, int(num_beams))                            # need at least 2 beams
    angle_increment = (angle_max - angle_min) / (num_beams - 1)   # gap between beams
    ranges = np.full(num_beams, range_max, dtype=np.float32)      # start as "nothing seen"

    # finds the closest masked pixel in each image column
    hit_rows = _first_hit_rows(mask, cam.horizon_row(), ignore_bottom_rows)
    # drop columns whose object sits on a not_below pixel (e.g. furniture on the ducting)
    if not_below is not None and contact_rows > 0:
        hit_rows[_standing_on(hit_rows, not_below, contact_rows, ignore_bottom_rows)] = -1

    # turn the pixel found in each column into a point on the floor (metres)
    cols = np.arange(hit_rows.size)
    forward, lateral, valid = ground_points(cols, hit_rows, cam)

    # keep columns with a hit whose ray reaches the floor within range_max
    keep = (hit_rows >= 0) & valid & (np.hypot(forward, lateral) <= range_max)

    # put each point in its beam, keeping the nearest
    _bin_points(forward[keep], lateral[keep], ranges, angle_min, angle_increment)

    return ScanResult(ranges=ranges, angle_min=angle_min, angle_increment=angle_increment,
                      boundary_rows=hit_rows, mask=mask)


def colour_scan(bgr: np.ndarray, cam: CameraModel, mask_params: MaskParams,
                num_beams: int, range_max: float,
                ignore_bottom_rows: int = 0) -> ScanResult:
    """
    Turn one camera image into a LiDAR-style scan of walls and boxes (Milestone 3 behaviour).

    Args:
        bgr (np.ndarray): Colour image from the camera (BGR).
        cam (CameraModel): Camera settings.
        mask_params (MaskParams): Colour thresholds for walls and boxes.
        num_beams (int): Number of directions in the scan.
        range_max (float): Distance reported when nothing is detected (m).
        ignore_bottom_rows (int): Rows at the bottom to skip (the car body).

    Returns:
        ScanResult: Ranges, scan geometry and the mask/boundary for debugging.
    """
    return mask_scan(obstacle_mask(bgr, mask_params), cam, num_beams, range_max,
                     ignore_bottom_rows)


def _standing_on(hit_rows: np.ndarray, below: np.ndarray, contact_rows: int,
                 ignore_bottom_rows: int = 0) -> np.ndarray:
    """
    Find the columns whose lowest masked pixel stands on a `below` pixel.

    Args:
        hit_rows (np.ndarray): Lowest masked row of each column (-1 if none).
        below (np.ndarray): HxW mask; non-zero where an object must not stand.
        contact_rows (int): Rows under each hit that are checked.
        ignore_bottom_rows (int): Rows at the bottom showing the car body (not checked).

    Returns:
        np.ndarray: True for columns with a `below` pixel within contact_rows under the hit.
    """
    height = below.shape[0]
    last = height - max(0, ignore_bottom_rows) - 1          # lowest row that is not the car
    # one row of contact_rows candidate rows under each column's hit
    rows = hit_rows[:, None] + np.arange(1, contact_rows + 1)[None, :]
    # only check columns that have a hit, and rows above the car body
    inside = (hit_rows[:, None] >= 0) & (rows <= last)
    cols = np.broadcast_to(np.arange(hit_rows.size)[:, None], rows.shape)
    values = below[np.clip(rows, 0, height - 1), cols] > 0   # clip keeps indices in the image
    return np.any(values & inside, axis=1)


def _first_hit_rows(mask: np.ndarray, horizon_row: float,
                    ignore_bottom_rows: int = 0) -> np.ndarray:
    """
    For each column, find the lowest masked pixel below the horizon.

    Lower in the image means closer to the car, so the lowest masked pixel in a column
    is the closest object in that direction. Rows above the horizon and rows showing the
    car itself are skipped. Flipping the band upside down makes np.argmax return the
    lowest hit of every column at once.

    Args:
        mask (np.ndarray): B&W image; non-zero = wall or box.
        horizon_row (float): Row of the horizon; only rows below this row are considered.
        ignore_bottom_rows (int): Rows at the bottom to skip (the car body).

    Returns:
        np.ndarray: One row number per column; -1 if nothing found.
    """
    height, width = mask.shape
    row_top = max(0, int(np.floor(horizon_row)) + 1)    # first row below the horizon
    row_bot = height - max(0, ignore_bottom_rows)       # first row of the car body
    hits = np.full(width, -1, dtype=np.int32)           # -1 = nothing found in this direction
    if row_top >= row_bot:
        return hits

    band = mask[row_top:row_bot] > 0                    # usable rows only
    found = band.any(axis=0)                            # columns with at least one hit
    lowest = band.shape[0] - 1 - np.argmax(band[::-1], axis=0)   # last True row of each column
    hits[found] = row_top + lowest[found]
    return hits


def _bin_points(forward: np.ndarray, lateral: np.ndarray, ranges: np.ndarray,
                angle_min: float, angle_increment: float) -> None:
    """
    Fill in the ranges with the nearest point in each direction.

    Each beam is one direction the camera can see. Several points can fall in the same
    beam; only the closest one is kept (np.minimum.at handles repeated beams).

    Args:
        forward (np.ndarray): Forward distances, in metres.
        lateral (np.ndarray): Lateral offsets (positive = right), in metres.
        ranges (np.ndarray): Ranges array to update in place.
        angle_min (float): Bearing of ranges[0], in radians.
        angle_increment (float): Spacing between bearings, in radians.
    """
    if forward.size == 0:
        return
    dist = np.hypot(forward, lateral)                                     # distance to the point
    bearing = np.arctan2(-lateral, forward)                               # left positive
    beam = np.round((bearing - angle_min) / angle_increment).astype(int)  # nearest beam
    inside = (beam >= 0) & (beam < ranges.size)                           # inside the view
    np.minimum.at(ranges, beam[inside], dist[inside].astype(ranges.dtype))
