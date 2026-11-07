"""
LiDAR and camera fusion (no ROS code).

The LiDAR is the primary sensor: 270 degrees, 1080 beams, about 4 cm accuracy and 40 Hz,
and it sees the black ducting. The camera adds what the LiDAR cannot know:

    Box labels:  the colour image tells cardboard from ducting. A camera box point that
                 lands on a LiDAR return (same bearing, ranges within assoc_tol) marks
                 that LiDAR beam as a confirmed box. The LiDAR range is kept, because it
                 is the more accurate one.
    Low objects: a depth point clearly nearer than anything the LiDAR sees at that bearing
                 is something the LiDAR plane misses, such as a low object. It is only
                 trusted after persist_frames camera frames in a row, so a single noisy
                 frame can never create a phantom obstacle. It is written into the
                 fused scan only: it is not a box, so it gets no box margin or pass speed.

Colour points only ever label LiDAR returns; they never add an object of their own. The
colour scan cannot measure height, so a flat orange line taped on the floor would look
like the base of a box. The depth scan measures height (floor rejected), so it can.
Colour wall points are never fused: on walls the LiDAR is better in every way.

Frames: camera points are (x ahead, y left) in the camera frame. They are rotated by
cam_yaw and moved by (cam_x, cam_y) into the LiDAR frame. The camera is mounted on top of
the LiDAR, so all three are close to 0 and are tuned by overlaying the scans in rviz.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


@dataclass
class FusionParams:
    """Settings for placing camera points on the LiDAR scan.

    Attributes:
        cam_x (float): Camera position ahead of the LiDAR (m).
        cam_y (float): Camera position left of the LiDAR (m).
        cam_yaw (float): Camera heading relative to the LiDAR (rad, left positive).
        assoc_tol (float): Camera and LiDAR ranges this close are the same object (m).
        assoc_window (int): LiDAR beams searched on each side of a camera point.
        persist_frames (int): Camera frames in a row before a camera-only point is trusted.
    """

    cam_x: float = 0.0
    cam_y: float = 0.0
    cam_yaw: float = 0.0
    assoc_tol: float = 0.25
    assoc_window: int = 4
    persist_frames: int = 3


@dataclass
class CameraPoints:
    """Camera points already placed on LiDAR beams.

    Attributes:
        beams (np.ndarray): LiDAR beam index of each point.
        ranges (np.ndarray): Distance of each point from the LiDAR (m).
        is_box (bool): True for cardboard points from the colour image,
            False for depth points.
    """

    beams: np.ndarray
    ranges: np.ndarray
    is_box: bool


def scan_to_points(ranges, angle_min: float, angle_increment: float,
                   range_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the beams of a camera scan that saw something into 2D points.

    Args:
        ranges: Beam ranges (m); range_max means nothing was seen.
        angle_min (float): Bearing of the first beam (rad, left positive).
        angle_increment (float): Spacing between beams (rad).
        range_max (float): "Nothing seen" value of the scan (m).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (x ahead, y left) of each point (m).
    """
    r = np.asarray(ranges, dtype=np.float64)
    a = angle_min + np.arange(r.size) * angle_increment
    with np.errstate(invalid='ignore'):
        seen = np.isfinite(r) & (r > 0.0) & (r < range_max - 1e-3)
    return r[seen] * np.cos(a[seen]), r[seen] * np.sin(a[seen])


def camera_to_lidar(x: np.ndarray, y: np.ndarray, params: FusionParams,
                    shift_x: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Move camera points into the LiDAR frame.

    Args:
        x (np.ndarray): Points ahead of the camera (m).
        y (np.ndarray): Points left of the camera (m).
        params (FusionParams): Camera mounting relative to the LiDAR.
        shift_x (float): Distance the car drove between the camera frame and the LiDAR
            scan (m). Points are moved this much closer (ego-motion compensation).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (x ahead, y left) in the LiDAR frame (m).
    """
    c, s = math.cos(params.cam_yaw), math.sin(params.cam_yaw)
    xl = c * x - s * y + params.cam_x - shift_x
    yl = s * x + c * y + params.cam_y
    return xl, yl


def points_to_beams(x: np.ndarray, y: np.ndarray, angle_min: float, angle_increment: float,
                    num_beams: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the LiDAR beam each point lies on, and its range from the LiDAR.

    Args:
        x (np.ndarray): Points ahead of the LiDAR (m).
        y (np.ndarray): Points left of the LiDAR (m).
        angle_min (float): Bearing of the LiDAR's first beam (rad).
        angle_increment (float): Spacing between LiDAR beams (rad).
        num_beams (int): Number of LiDAR beams.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (beam index, range) of the points that fall on a beam.
    """
    r = np.hypot(x, y)
    beams = np.round((np.arctan2(y, x) - angle_min) / angle_increment).astype(int)
    inside = (beams >= 0) & (beams < num_beams) & (r > 0.0)
    return beams[inside], r[inside]


def clean_lidar(ranges, range_min: float, range_max: float) -> np.ndarray:
    """
    Copy LiDAR ranges with every invalid reading set to infinity.

    Args:
        ranges: Raw LiDAR ranges (m).
        range_min (float): Shortest valid range (m).
        range_max (float): Longest valid range (m).

    Returns:
        np.ndarray: Ranges, inf where the reading is NaN, inf or outside the valid range.
    """
    r = np.array(ranges, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        bad = ~np.isfinite(r) | (r < range_min) | (r > range_max)
    r[bad] = np.inf
    return r


class ScanFuser:
    """Fuse camera points into LiDAR scans, remembering persistence between scans."""

    def __init__(self, params: FusionParams) -> None:
        """Store the settings; persistence counters start empty.

        Args:
            params (FusionParams): Association and persistence settings.
        """
        self.params = params
        self.persist: np.ndarray | None = None   # camera frames in a row each beam was seen
        self.trusted: np.ndarray | None = None   # beams the last fuse() added from depth

    def fuse(self, lidar_ranges, range_min: float, range_max: float,
             sources: List[CameraPoints],
             new_camera_frame: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse one LiDAR scan with the latest camera points.

        Args:
            lidar_ranges: Raw LiDAR ranges (m).
            range_min (float): LiDAR range_min (m).
            range_max (float): LiDAR range_max (m).
            sources (List[CameraPoints]): Camera points already placed on LiDAR beams.
            new_camera_frame (bool): True if any source is a camera frame not used before.
                Persistence only counts new frames (camera 30 Hz, LiDAR 40 Hz).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (fused, obstacle). fused is the raw LiDAR scan
            with trusted depth-only points written in. obstacle holds the range of every
            beam the colour camera confirmed as a box, inf elsewhere. Depth-only points are
            not boxes: on the car, floor that slipped past the floor rejection made the
            planner slow down for "boxes" all round an empty track.
        """
        raw = np.array(lidar_ranges, dtype=np.float64)
        n = raw.size
        lidar = clean_lidar(raw, range_min, range_max)
        if self.persist is None or self.persist.size != n or not sources:
            self.persist = np.zeros(n, dtype=int)   # new geometry, or the camera went quiet

        fused = raw.copy()
        obstacle = np.full(n, np.inf)
        seen = np.zeros(n, dtype=bool)          # beams with a camera-only point this frame
        camera_only = np.full(n, np.inf)        # nearest camera-only range on each beam

        w = max(0, int(self.params.assoc_window))
        offsets = np.arange(-w, w + 1)
        tol = self.params.assoc_tol

        for src in sources:
            if src.beams.size == 0:
                continue
            # the LiDAR beams around each camera point
            window = np.clip(src.beams[:, None] + offsets[None, :], 0, n - 1)
            lidar_window = lidar[window]
            close = np.abs(lidar_window - src.ranges[:, None]) <= tol

            # associated: the LiDAR sees the same object; a box label marks those beams
            if src.is_box:
                hit = window[close]
                obstacle[hit] = lidar[hit]

            # camera-only: clearly nearer than anything the LiDAR sees there. Only depth
            # points, which measure height, may add an object (see the module docstring)
            if src.is_box:
                continue
            nearer = src.ranges < lidar_window.min(axis=1) - tol
            beams = src.beams[nearer]
            seen[beams] = True
            np.minimum.at(camera_only, beams, src.ranges[nearer])

        # count camera frames in a row; a nearby beam last frame also counts (bearings drift)
        if new_camera_frame:
            self.persist = np.where(seen, self._dilate(self.persist, w) + 1, 0)
        trusted = seen & (self.persist >= self.params.persist_frames)

        fused[trusted] = np.minimum(lidar[trusted], camera_only[trusted])
        self.trusted = trusted
        return fused, obstacle

    @staticmethod
    def _dilate(counts: np.ndarray, w: int) -> np.ndarray:
        """
        Return the largest count within w beams of each beam.

        Args:
            counts (np.ndarray): Per-beam counts.
            w (int): Half-width of the window (beams).

        Returns:
            np.ndarray: Windowed maximum of counts.
        """
        if w <= 0:
            return counts
        padded = np.pad(counts, w)
        return np.max(np.stack([padded[k:k + counts.size] for k in range(2 * w + 1)]), axis=0)
