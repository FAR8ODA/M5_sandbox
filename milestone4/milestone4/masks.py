"""
Colour segmentation: which pixels are track wall and which are cardboard box.

The track boundary is black ducting and the obstacles are cardboard boxes; both
stand out against the light grey floor (Vision intro, slide 26: bright floor,
dark walls).

The image is converted to HSV (slide 18: cv2.cvtColor with COLOR_BGR2HSV)
because HSV separates the kind of colour (hue), its intensity (saturation) and
its brightness (value) (slide 13). That makes both rules one-line thresholds:
black ducting is low value whatever its hue, and cardboard is a brown/orange
hue with clear saturation, while the grey floor has almost none. Each rule gives
a binary image, like cv2.threshold (slide 20).

Milestone 4 change: the two rules are kept as two separate masks. The wall mask
and the box mask are combined with a bitwise OR (slide 19) for the driving scan,
and the box mask alone becomes /camera_box_scan, which the fusion node uses to
tell the LiDAR which of its points are boxes. A pixel is never in both masks.

Anything above the horizon (chairs, people, the room) is never used later, so
it does not matter if it matches too.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class MaskParams:
    """HSV thresholds that separate walls and boxes from the floor.

    OpenCV HSV ranges: H 0-179, S 0-255, V 0-255. Every bound is inclusive.

    Attributes:
        dark_v_max (int): Pixels with V at or below this are "dark" (ducting).
        use_box_color (bool): Also detect cardboard by colour.
        box_h_min (int): Minimum hue for cardboard.
        box_h_max (int): Maximum hue for cardboard (must be >= box_h_min; the
            range does not wrap around from 179 to 0).
        box_s_min (int): Minimum saturation for cardboard (floor is near 0).
        box_v_min (int): Minimum value for cardboard.
        open_kernel (int): Size of the morphological opening that removes
            speckle; 0 or 1 disables it.
    """

    # black ducting rule.
    dark_v_max: int = 70
    # cardboard rule.
    use_box_color: bool = True
    box_h_min: int = 5
    box_h_max: int = 30
    box_s_min: int = 60
    box_v_min: int = 60
    # noise clean-up.
    open_kernel: int = 5


def _clean(mask: np.ndarray, open_kernel: int) -> np.ndarray:
    """Turn a True/False mask into a 0/255 mask and remove speckle.

    The speckle (floor texture, glare edges) is removed with a morphological
    opening: erosion then dilation. It deletes blobs the square kernel cannot
    fit inside and keeps the shape of larger ones, so the kernel must stay
    smaller than the ducting's width in the image. Not in the slides; see the
    OpenCV morphological operations tutorial:
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    Args:
        mask (np.ndarray): HxW boolean mask.
        open_kernel (int): Opening kernel size; 0 or 1 disables it.

    Returns:
        np.ndarray: HxW uint8 mask, 255 for a hit and 0 otherwise.
    """
    mask_u8 = mask.astype(np.uint8) * 255
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8


def split_masks(bgr: np.ndarray, params: MaskParams) -> Tuple[np.ndarray, np.ndarray]:
    """Segment wall pixels and box pixels as two separate masks.

    The HSV conversion is done once and shared by both rules.

    Args:
        bgr (np.ndarray): HxWx3 uint8 BGR image.
        params (MaskParams): Thresholds to apply.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (wall, box) HxW uint8 masks, 255 where
        the pixel matches and 0 elsewhere. No pixel is set in both.
    """
    # opencv images are bgr, not rgb (slide 11), so convert from bgr to hsv.
    # opencv stores hue as 0-179 (degrees halved so it fits in a uint8).
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # black ducting: low brightness (value) whatever the hue or saturation.
    wall = v <= params.dark_v_max

    # cardboard: brown/orange hue with clear saturation. the grey floor has
    # almost no saturation, so the s test keeps it out. pixels already counted
    # as wall are left out, so a shadowed box edge is never reported twice.
    if params.use_box_color:
        box = ((h >= params.box_h_min) & (h <= params.box_h_max)
               & (s >= params.box_s_min) & (v >= params.box_v_min) & ~wall)
    else:
        box = np.zeros_like(wall)

    return _clean(wall, params.open_kernel), _clean(box, params.open_kernel)


def wall_mask(bgr: np.ndarray, params: MaskParams) -> np.ndarray:
    """Return the mask of black ducting pixels.

    Args:
        bgr (np.ndarray): HxWx3 uint8 BGR image.
        params (MaskParams): Thresholds to apply.

    Returns:
        np.ndarray: HxW uint8 mask, 255 on walls.
    """
    return split_masks(bgr, params)[0]


def box_mask(bgr: np.ndarray, params: MaskParams) -> np.ndarray:
    """Return the mask of cardboard box pixels.

    Args:
        bgr (np.ndarray): HxWx3 uint8 BGR image.
        params (MaskParams): Thresholds to apply.

    Returns:
        np.ndarray: HxW uint8 mask, 255 on boxes.
    """
    return split_masks(bgr, params)[1]


def obstacle_mask(bgr: np.ndarray, params: MaskParams) -> np.ndarray:
    """Return walls and boxes together, as in Milestone 3.

    Args:
        bgr (np.ndarray): HxWx3 uint8 BGR image.
        params (MaskParams): Thresholds to apply.

    Returns:
        np.ndarray: HxW uint8 mask, 255 on walls or boxes.
    """
    wall, box = split_masks(bgr, params)
    return cv2.bitwise_or(wall, box)
