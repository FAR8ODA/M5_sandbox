
"""
Takes a depth camera image and converts it into a set of distances
and angles that can be used for obstacle detection

only a horizontal section of the image is used, which is split
into smaller sections to get each distance measurement.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DepthBandParams:
    """Settings for how we process the depth image."""

    band_top: int = -40
    band_bottom: int = 50
    num_beams: int = 90
    percentile: float = 10.0
    min_valid_frac: float = 0.05
    range_min: float = 0.15
    range_max: float = 3.0


def depth_ranges(depth_m: np.ndarray,fx: float, cx: float, cy: float, prm: DepthBandParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets distance measurements from a depth image

    It uses the camera calibration values to then calculate the angle
    for each section of the image

    Returns: the distances and angles, starting from the right
    side of the image and going leftward.
    """

    #eror checking 
    if depth_m.ndim != 2 or 0 in depth_m.shape:
        raise ValueError("Expected a non-empty 2D depth image")

    
    if not math.isfinite(fx) or fx <= 0:
        raise ValueError("Camera focal length must be positive")

    height, width = depth_m.shape
    beam_count = max(2, int(prm.num_beams))

    #this will find the image rows used to check for obstacles
    row_start = int(np.clip(cy + prm.band_top, 0, height - 1))
    row_end = int(np.clip(cy + prm.band_bottom, row_start + 1, height))

    #makes the band that depth will precieve (baiscally cropping it)
    depth_band = depth_m[row_start:row_end]

    # Divide the image width into equally spaced sections.
    column_edges = np.linspace(0, width, beam_count + 1).astype(int)

    distances = np.full(beam_count, prm.range_max, dtype=np.float32)
    
    angles = np.zeros(beam_count, dtype=np.float64)

    for beam in range(beam_count):

        #this is so the start is at the right side so angles increase leftward
        left = width - column_edges[beam + 1]
        right = width - column_edges[beam]

        # Determine the direction of this beam
        middle = (left + right - 1) / 2.0
        angle = math.atan((cx - middle) / fx)
        angles[beam] = angle
        #collects depth measurements for this section
        section = depth_band[:, left:right]

        if section.size == 0:
            continue

        # Only keep finite distances inside the selected range.
        valid_mask = (
            np.isfinite(section)
            & (section > prm.range_min)
            & (section < prm.range_max)
        )

        valid_depths = section[valid_mask]

        if valid_depths.size == 0:
            continue

        valid_fraction = valid_depths.size / section.size

        if valid_fraction < prm.min_valid_frac:
            continue

        # Use a low percentile this is to help with noise in the images
        forward_depth = float(
            np.percentile(valid_depths, prm.percentile)
        )

        #Converts forward depth to horizontal ray distance.
        ray_distance = forward_depth / math.cos(angle)

        distances[beam] = min(ray_distance, prm.range_max)

    return distances, angles