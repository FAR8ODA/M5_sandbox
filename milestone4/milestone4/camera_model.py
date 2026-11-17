"""
Camera geometry: pinhole intrinsics plus mounting height and pitch (no ROS code).

ground_points() maps pixels to floor points (flat-floor assumption); image_points()
maps known 3D points back to pixels. Frames: camera x right, y down, z forward;
vehicle forward and lateral (positive = right); bearing left-positive; pitch
positive when tilted down.
"""

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np

# smallest downward ray component that still counts as hitting the floor. rays
# flatter than this are treated as never landing, which also avoids dividing by
# a near-zero value (the floor point would be absurdly far away anyway).
_MIN_DOWN = 1e-6


@dataclass
class CameraModel:
    """Pinhole intrinsics plus the camera's mounting height and pitch.

    Attributes:
        fx (float): Focal length in x, in pixels.
        fy (float): Focal length in y, in pixels.
        cx (float): Principal point x, in pixels.
        cy (float): Principal point y, in pixels.
        width (int): Image width, in pixels.
        height_px (int): Image height, in pixels (named so it is not mixed up
            with mount_height).
        mount_height (float): Height of the camera above the floor, in metres.
        pitch_rad (float): Downward tilt of the camera, in radians.
    """

    # intrinsics, read from camera_info.k (all in pixels).
    fx: float
    fy: float
    cx: float
    cy: float
    # image size, read from camera_info (in pixels).
    width: int
    height_px: int
    # how the camera is mounted on the car, set by node parameters.
    mount_height: float
    pitch_rad: float

    @classmethod
    def from_k(cls, k: Sequence[float], width: int, height_px: int,
               mount_height: float, pitch_deg: float) -> 'CameraModel':
        """Build a camera model from a row-major 3x3 intrinsic matrix K.

        Args:
            k (Sequence[float]): The 9 values of K, as in CameraInfo.k.
            width (int): Image width, in pixels.
            height_px (int): Image height, in pixels.
            mount_height (float): Camera height above the floor, in metres.
            pitch_deg (float): Downward tilt of the camera, in degrees.

        Returns:
            CameraModel: The populated camera model.
        """
        # k is the 3x3 matrix [fx 0 cx; 0 fy cy; 0 0 1] flattened row by row,
        # so fx = k[0], cx = k[2], fy = k[4] and cy = k[5]. everything is cast
        # to plain python numbers so numpy/ros types do not leak into the model.
        return cls(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]),
                   width=int(width), height_px=int(height_px),
                   mount_height=float(mount_height),
                   pitch_rad=math.radians(pitch_deg))

    def horizon_row(self) -> float:
        """Return the image row of the horizon (rays parallel to the floor).

        Rows below this value look down at the floor; rows at or above it look
        level or upward and can never hit the floor.

        Returns:
            float: Horizon row, in pixels (can be outside the image).
        """
        # a ray has no downward component when yn * cos(p) + sin(p) = 0 (see
        # ground_points), where yn = (row - cy) / fy is the row's offset from
        # the principal point in focal lengths. solving gives yn = -tan(p), so
        # row = cy - fy * tan(p). tilting the camera down (p > 0) moves the
        # horizon up the image (to a smaller row).
        return self.cy - self.fy * math.tan(self.pitch_rad)

    def bearing_limits(self) -> Tuple[float, float]:
        """Return the bearings of the rightmost and leftmost image columns.

        Returns:
            Tuple[float, float]: (angle_min, angle_max) in radians, where
            angle_min is the right edge (negative) and angle_max the left edge.
        """
        # a column u looks along xn = (u - cx) / fx, and its bearing is
        # atan2(-xn, 1) = -atan(xn) (minus because left is positive).
        # rightmost column (u = width - 1): xn > 0, so the bearing is negative.
        # leftmost column (u = 0): xn = -cx / fx, so the bearing is +atan(cx / fx).
        # note: these are the bearings along the optical axis, ignoring pitch.
        # with the camera tilted down, floor points in the edge columns can land
        # slightly outside this range, and boundary.bin_points drops them.
        angle_min = -math.atan((self.width - 1 - self.cx) / self.fx)
        angle_max = math.atan(self.cx / self.fx)
        return angle_min, angle_max


def ground_points(us: np.ndarray, vs: np.ndarray,
                  cam: CameraModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect pixel rays with the floor plane.

    Args:
        us (np.ndarray): Pixel columns.
        vs (np.ndarray): Pixel rows, the same shape as us.
        cam (CameraModel): Camera intrinsics, height and pitch.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (forward, lateral, valid),
        where forward/lateral are in metres (lateral positive = right) and
        valid is False for rays that never reach the floor. Forward and
        lateral are 0 wherever valid is False, so filter with valid first.
    """
    # normalised image coordinates: the pixel's offset from the principal point
    # divided by the focal length (pinhole model, opencv camera model docs).
    # the ray through the pixel is then (xn, yn, 1) in the camera optical frame.
    xn = (np.asarray(us, dtype=np.float64) - cam.cx) / cam.fx
    yn = (np.asarray(vs, dtype=np.float64) - cam.cy) / cam.fy
    c, s = math.cos(cam.pitch_rad), math.sin(cam.pitch_rad)

    # rotate the ray (xn, yn, 1) from the camera frame into the vehicle frame.
    # tilting the camera down by p is a rotation about its x axis: the optical
    # axis (z) now points (forward c, down s) and the image-down axis (y) points
    # (forward -s, down c). the x axis is unchanged, so lateral stays xn.
    down = yn * c + s
    forward_dir = c - yn * s

    # walking t along the ray drops t * down metres, so it meets the floor at
    # t = mount_height / down. rays that do not point down never get there.
    # the inner np.where puts 1.0 in place of down for those rays so the
    # division is always safe, and the outer np.where sets their t to 0.
    valid = down > _MIN_DOWN
    t = np.where(valid, cam.mount_height / np.where(valid, down, 1.0), 0.0)

    # scale the ray's forward and lateral parts by t to get the floor point.
    return forward_dir * t, xn * t, valid


def image_points(forward: np.ndarray, lateral: np.ndarray, down: float,
                 cam: CameraModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points at a known height into the image (the inverse of ground_points).

    ground_points needs the floor to turn a pixel into a distance. Here the distance is
    already known (the LiDAR measured it), so no floor assumption is needed: the point's
    pixel follows directly from the pinhole model.

    Args:
        forward (np.ndarray): Distance ahead of the camera, in metres.
        lateral (np.ndarray): Distance to the side, in metres (positive = right).
        down (float): How far the points are below the camera, in metres.
        cam (CameraModel): Camera intrinsics and pitch (mount_height is not used).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (u, v, valid): pixel column and row of
        each point (can be outside the image), and False for points behind the camera.
    """
    forward = np.asarray(forward, dtype=np.float64)
    lateral = np.asarray(lateral, dtype=np.float64)
    c, s = math.cos(cam.pitch_rad), math.sin(cam.pitch_rad)

    # rotate from the vehicle frame into the tilted camera frame (ground_points does the
    # opposite): depth along the optical axis, and the offset along the image-down axis.
    depth = forward * c + down * s
    y_cam = down * c - forward * s

    # points closer than 5 cm in depth (or behind the camera) have no sensible pixel.
    valid = depth > 0.05
    safe = np.where(valid, depth, 1.0)
    return cam.cx + cam.fx * lateral / safe, cam.cy + cam.fy * y_cam / safe, valid
