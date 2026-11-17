"""
Depth image -> obstacle distances per bearing (no ROS code).

depth_ranges() splits a horizontal band of the image into one strip per beam and
reports a low percentile of each strip. Pixels outside [min_height, max_height]
above the floor are rejected, so the floor never appears as an obstacle.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


@dataclass
class DepthBandParams:
    """Settings for how the depth image is processed.

    Attributes:
        band_top (int): First row used, relative to the principal point (px, negative = up).
        band_bottom (int): Last row used, relative to the principal point (px).
        num_beams (int): Number of beams (vertical strips) in the scan.
        percentile (float): Percentile of each strip's distances used as its range.
        min_valid_frac (float): Fraction of a strip that must be valid, otherwise "nothing seen".
        range_min (float): Minimum valid depth (m).
        range_max (float): Maximum valid depth, also the value for "nothing seen" (m).
        floor_reject (bool): Drop pixels that are floor (or higher than an obstacle).
        mount_height (float): Camera height above the floor (m).
        pitch_deg (float): Camera pitch (degrees, positive = tilted down).
        min_height (float): Points lower than this above the floor are floor (m).
        max_height (float): Points higher than this are ignored (m).
    """

    band_top: int = -40
    band_bottom: int = 90
    num_beams: int = 90
    percentile: float = 10.0
    min_valid_frac: float = 0.05
    range_min: float = 0.15
    range_max: float = 2.5
    floor_reject: bool = True
    mount_height: float = 0.187
    pitch_deg: float = -5.3
    min_height: float = 0.05
    max_height: float = 0.40


def depth_ranges(depth_m: np.ndarray, fx: float, cx: float, cy: float, prm: DepthBandParams,
                 fy: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get obstacle distances from a depth image.

    Args:
        depth_m (np.ndarray): HxW depth in metres (0 or NaN = no reading).
        fx (float): Focal length in x (px).
        cx (float): Principal point x (px).
        cy (float): Principal point y (px).
        prm (DepthBandParams): Band, percentile and floor rejection settings.
        fy (float, optional): Focal length in y (px). Needed for floor rejection;
            without it the floor is not rejected (Milestone 3 behaviour).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (distances, angles), starting from the right side
        of the image and going left. Distances are range_max where nothing was seen.

    Raises:
        ValueError: If the image is not a non-empty 2D array or fx is not positive.
    """
    # error checking
    if depth_m.ndim != 2 or 0 in depth_m.shape:
        raise ValueError('Expected a non-empty 2D depth image')
    if not math.isfinite(fx) or fx <= 0:
        raise ValueError('Camera focal length must be positive')

    height, width = depth_m.shape
    beam_count = max(2, int(prm.num_beams))

    # the image rows used to check for obstacles
    row_start = int(np.clip(cy + prm.band_top, 0, height - 1))
    row_end = int(np.clip(cy + prm.band_bottom, row_start + 1, height))
    band = depth_m[row_start:row_end].astype(np.float64)

    # forward distance of each pixel and which pixels can be obstacles
    forward, keep = _forward_and_keep(band, row_start, cy, fy, prm)

    # divide the image width into equally spaced strips
    column_edges = np.linspace(0, width, beam_count + 1).astype(int)
    distances = np.full(beam_count, prm.range_max, dtype=np.float32)
    angles = np.zeros(beam_count, dtype=np.float64)

    # pixels with a valid reading inside the range limits and at obstacle height
    with np.errstate(invalid='ignore'):
        usable = np.isfinite(band) & (band > prm.range_min) & (band < prm.range_max) & keep

    for beam in range(beam_count):
        # start at the right side so angles increase leftward
        left = width - column_edges[beam + 1]
        right = width - column_edges[beam]

        # direction of this beam
        middle = (left + right - 1) / 2.0
        angle = math.atan((cx - middle) / fx)
        angles[beam] = angle

        # skip strips with too few valid pixels (left at range_max)
        section = usable[:, left:right]
        if section.size == 0 or section.mean() < prm.min_valid_frac:
            continue

        # low percentile of the forward distances, to resist noise
        forward_depth = float(np.percentile(forward[:, left:right][section], prm.percentile))

        # forward distance -> distance along the beam
        distances[beam] = min(forward_depth / math.cos(angle), prm.range_max)

    return distances, angles


def _forward_and_keep(band: np.ndarray, row_start: int, cy: float, fy: Optional[float],
                      prm: DepthBandParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute each pixel's forward distance and whether it is at obstacle height.

    A pixel in row v looks along the ray (xn, yn, 1) with yn = (v - cy) / fy. Tilting the
    camera down by p, a point at depth z is z * (yn cos p + sin p) below the camera and
    z * (cos p - yn sin p) ahead of it. Its height above the floor is the mounting height
    minus the drop.

    Args:
        band (np.ndarray): Depth band in metres.
        row_start (int): Image row of band[0].
        cy (float): Principal point y (px).
        fy (float, optional): Focal length in y (px); None disables floor rejection.
        prm (DepthBandParams): Mounting and height limits.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (forward, keep), both the shape of band.
    """
    # floor rejection off: depth is the forward distance and every pixel is kept
    if not prm.floor_reject or fy is None or fy <= 0:
        return band, np.ones(band.shape, dtype=bool)

    # normalised row coordinate of each band row, broadcast across columns
    rows = np.arange(row_start, row_start + band.shape[0], dtype=np.float64)
    yn = ((rows - cy) / fy)[:, None]
    p = math.radians(prm.pitch_deg)
    c, s = math.cos(p), math.sin(p)

    with np.errstate(invalid='ignore'):
        drop = band * (yn * c + s)
        forward = band * (c - yn * s)
        height_above_floor = prm.mount_height - drop
        keep = (height_above_floor >= prm.min_height) & (height_above_floor <= prm.max_height)
    return forward, keep
