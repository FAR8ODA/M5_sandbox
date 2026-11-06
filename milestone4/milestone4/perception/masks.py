"""
Colour segmentation: which pixels are track boundary or obstacle.

The track boundary is black ducting and the obstacles are cardboard boxes; both
stand out against the light grey floor (slide 26: bright floor, dark walls).

The image is converted to HSV (slide 18: cv2.cvtColor with COLOR_BGR2HSV)
because HSV separates the kind of colour (hue), its intensity (saturation) and
its brightness (value) (slide 13). That makes both rules one-line thresholds:
black ducting is low value whatever its hue, and cardboard is a brown/orange
hue with clear saturation, while the grey floor has almost none. Each rule gives
a binary image, like cv2.threshold (slide 20), and the two are combined with a
bitwise OR into one mask (slide 19: masks and bitwise operations).

Anything above the horizon (chairs, people, the room) is never used later, so
it does not matter if it matches too.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MaskParams:
    """HSV thresholds that separate boundary/obstacles from the floor.

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


def obstacle_mask(bgr: np.ndarray, params: MaskParams) -> np.ndarray:
    """Segment boundary and obstacle pixels in a colour image.

    Args:
        bgr (np.ndarray): HxWx3 uint8 BGR image.
        params (MaskParams): Thresholds to apply.

    Returns:
        np.ndarray: HxW uint8 mask, 255 where a boundary/obstacle is detected
        and 0 elsewhere.
    """
    # opencv images are bgr, not rgb (slide 11), so convert from bgr to hsv
    # (slides 13, 18). opencv stores hue as 0-179 (degrees halved so it fits in
    # a uint8); saturation and value are 0-255.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # pull out the three channels. these are views into hsv, not copies.
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # black ducting: low brightness (value) whatever the hue or saturation. this
    # is a binary threshold (slide 20); the result is a true/false array.
    mask = v <= params.dark_v_max

    # cardboard: brown/orange hue with clear saturation. the grey floor has
    # almost no saturation, so the s test keeps it out, and the v test stops
    # very dark pixels (whose hue is unreliable) from counting as cardboard.
    # the hue test is a plain min..max range, so it does not wrap past 179.
    # the box pixels are merged into the ducting mask with a bitwise or (slide 19).
    if params.use_box_color:
        box = ((h >= params.box_h_min) & (h <= params.box_h_max)
               & (s >= params.box_s_min) & (v >= params.box_v_min))
        mask |= box

    # turn true/false into the usual opencv mask: 255 for a hit, 0 otherwise.
    mask_u8 = mask.astype(np.uint8) * 255

    # remove isolated noise pixels (floor texture, glare edges) with a
    # morphological opening: erosion then dilation. this deletes blobs that the
    # square kernel cannot fit inside and keeps the shape of larger ones. any
    # line thinner than the kernel is removed too, so the kernel must stay
    # smaller than the ducting's width in the image.
    # not in the slides; see the opencv morphological operations tutorial:
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    if params.open_kernel > 1:
        kernel = np.ones((params.open_kernel, params.open_kernel), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8