"""
LiDAR and camera fusion (no ROS code).

The LiDAR is the primary sensor. box_labels() projects each LiDAR return into the
colour box mask to mark which beams hit cardboard (colour never adds points of its
own). ScanFuser.fuse() also writes depth points that are clearly nearer than the
LiDAR, once they persist for persist_frames camera frames. Camera points are
(x ahead, y left) and are moved into the LiDAR frame by cam_x, cam_y and cam_yaw.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

from milestone4.camera_model import CameraModel, image_points
import numpy as np


@dataclass
class FusionParams:
    """Settings for placing camera points on the LiDAR scan.

    Attributes:
        cam_x (float): Camera position ahead of the LiDAR (m).
        cam_y (float): Camera position left of the LiDAR (m).
        cam_yaw (float): Camera heading relative to the LiDAR (rad, left positive).
        cam_z (float): Camera height above the LiDAR's scan plane (m).
        assoc_tol (float): Depth and LiDAR ranges this close are the same object (m).
        assoc_window (int): LiDAR beams searched on each side of a depth point.
        persist_frames (int): Camera frames in a row before a depth-only point is trusted.
        label_strip (float): Half-height of the pixel strip checked around each projected
            LiDAR point, as an angle (rad). It absorbs pitch and cam_z errors.
        label_fraction (float): Share of the strip that must be box pixels (0-1).
        label_max_range (float): LiDAR returns farther than this are not labelled (m).
    """

    cam_x: float = 0.0
    cam_y: float = 0.0
    cam_yaw: float = 0.0
    cam_z: float = 0.06
    assoc_tol: float = 0.25
    assoc_window: int = 4
    persist_frames: int = 3
    label_strip: float = math.radians(2.0)
    label_fraction: float = 0.5
    label_max_range: float = 4.0


@dataclass
class CameraPoints:
    """Depth camera points already placed on LiDAR beams.

    Attributes:
        beams (np.ndarray): LiDAR beam index of each point.
        ranges (np.ndarray): Distance of each point from the LiDAR (m).
    """

    beams: np.ndarray
    ranges: np.ndarray


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
    # keep beams with a real reading (range_max, with a little slack, means nothing seen)
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
    # rotate by cam_yaw, then translate by the camera offset minus the distance driven
    c, s = math.cos(params.cam_yaw), math.sin(params.cam_yaw)
    xl = c * x - s * y + params.cam_x - shift_x
    yl = s * x + c * y + params.cam_y
    return xl, yl


def lidar_to_camera(x: np.ndarray, y: np.ndarray, params: FusionParams,
                    shift_x: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Move LiDAR points into the camera frame (the inverse of camera_to_lidar).

    Args:
        x (np.ndarray): Points ahead of the LiDAR (m).
        y (np.ndarray): Points left of the LiDAR (m).
        params (FusionParams): Camera mounting relative to the LiDAR.
        shift_x (float): Distance the car drove between the camera frame and the LiDAR
            scan (m). Points are moved this much farther away, to where the camera saw them.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (x ahead, y left) in the camera frame (m).
    """
    # undo the translation, then rotate by -cam_yaw
    c, s = math.cos(params.cam_yaw), math.sin(params.cam_yaw)
    dx, dy = x - params.cam_x + shift_x, y - params.cam_y
    return c * dx + s * dy, c * dy - s * dx


def column_counts(mask: np.ndarray) -> np.ndarray:
    """
    Count the box pixels down every column of a mask, once per camera frame.

    With these running counts, the number of box pixels in any strip of rows of a column is
    one subtraction, so labelling a whole LiDAR scan (40 Hz) costs almost nothing.

    Args:
        mask (np.ndarray): HxW mask, non-zero on box pixels.

    Returns:
        np.ndarray: (H + 1) x W counts; counts[r, c] = box pixels in rows 0 .. r-1 of column c.
    """
    mask = np.asarray(mask)
    # row 0 stays zero so counts[bottom + 1] - counts[top] is the strip sum
    counts = np.zeros((mask.shape[0] + 1, mask.shape[1]), dtype=np.int32)
    np.cumsum(mask > 0, axis=0, out=counts[1:])
    return counts


def box_labels(lidar: np.ndarray, angle_min: float, angle_increment: float,
               counts: np.ndarray, cam: CameraModel, params: FusionParams,
               shift_x: float = 0.0) -> np.ndarray:
    """
    Find the LiDAR beams that hit a cardboard box, by looking at their pixels.

    Each LiDAR return is projected into the box mask. Around its pixel, a vertical strip
    of +/- label_strip is checked; the beam is a box if at least label_fraction of the
    strip is box pixels. A box face runs from the floor to its top, well above and below
    the LiDAR's plane, so the strip stays on it even when the car pitches.

    Args:
        lidar (np.ndarray): Cleaned LiDAR ranges (m), inf where invalid.
        angle_min (float): Bearing of the LiDAR's first beam (rad).
        angle_increment (float): Spacing between LiDAR beams (rad).
        counts (np.ndarray): column_counts() of the box mask.
        cam (CameraModel): Intrinsics at the mask's resolution, and the camera pitch.
        params (FusionParams): Camera mounting and label settings.
        shift_x (float): Distance driven between the camera frame and the LiDAR scan (m).

    Returns:
        np.ndarray: True for every beam whose return is on a box.
    """
    # only valid returns within label_max_range can be labelled
    lidar = np.asarray(lidar, dtype=np.float64)
    labels = np.zeros(lidar.size, dtype=bool)
    index = np.nonzero(np.isfinite(lidar) & (lidar <= params.label_max_range))[0]
    if index.size == 0 or counts.size == 0:
        return labels

    # the LiDAR returns as seen from the camera, cam_z below it
    angles = angle_min + index * angle_increment
    x, y = lidar_to_camera(lidar[index] * np.cos(angles), lidar[index] * np.sin(angles),
                           params, shift_x)
    u, v, valid = image_points(x, -y, params.cam_z, cam)       # lateral: right positive

    # the strip of rows around each pixel, cut off at the image edges
    height, width = counts.shape[0] - 1, counts.shape[1]
    half = cam.fy * math.tan(params.label_strip)
    inside = valid & (u >= 0.0) & (u <= width - 1.0) & (v >= 0.0) & (v <= height - 1.0)
    col = np.round(u[inside]).astype(int)
    top = np.clip(np.floor(v[inside] - half), 0, height - 1).astype(int)
    bottom = np.clip(np.ceil(v[inside] + half), 0, height - 1).astype(int)

    # box pixels in each strip, from the running column counts; enough of them = a box
    box_pixels = counts[bottom + 1, col] - counts[top, col]
    labels[index[inside]] = box_pixels >= params.label_fraction * (bottom - top + 1)
    return labels


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
    # nearest beam by bearing; drop points outside the scan or at the origin
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
    """
    Fuse box labels and depth points into LiDAR scans, remembering persistence.

    Create once with FusionParams and call fuse() for every LiDAR scan; the
    per-beam persistence counters carry over between calls.
    """

    def __init__(self, params: FusionParams) -> None:
        """Store the settings; persistence counters start empty.

        Args:
            params (FusionParams): Association and persistence settings.
        """
        self.params = params
        self.persist: np.ndarray | None = None   # camera frames in a row each beam was seen

    def fuse(self, lidar_ranges, range_min: float, range_max: float,
             sources: List[CameraPoints], new_camera_frame: bool = True,
             labels: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse one LiDAR scan with the latest camera data.

        Args:
            lidar_ranges: Raw LiDAR ranges (m).
            range_min (float): LiDAR range_min (m).
            range_max (float): LiDAR range_max (m).
            sources (List[CameraPoints]): Depth points already placed on LiDAR beams.
            new_camera_frame (bool): True if any source is a camera frame not used before.
                Persistence only counts new frames (camera 30 Hz, LiDAR 40 Hz).
            labels (np.ndarray | None): Per beam, True where box_labels() found a box;
                None if there is no fresh colour frame.

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
        if labels is not None and labels.size == n:
            # a label only marks a beam the LiDAR has a return on; the LiDAR range is kept
            hit = labels & np.isfinite(lidar)
            obstacle[hit] = lidar[hit]
        seen = np.zeros(n, dtype=bool)         # beams with a camera-only point this frame
        camera_only = np.full(n, np.inf)        # nearest camera-only range on each beam

        # beam offsets searched around each depth point
        w = max(0, int(self.params.assoc_window))
        offsets = np.arange(-w, w + 1)
        tol = self.params.assoc_tol

        for src in sources:
            if src.beams.size == 0:
                continue
            # the LiDAR beams around each camera point
            window = np.clip(src.beams[:, None] + offsets[None, :], 0, n - 1)
            lidar_window = lidar[window]

            # depth-only: clearly nearer than anything the LiDAR sees there. A depth point
            # on a LiDAR return adds nothing: the LiDAR range is the more accurate one
            nearer = src.ranges < lidar_window.min(axis=1) - tol
            beams = src.beams[nearer]
            seen[beams] = True
            np.minimum.at(camera_only, beams, src.ranges[nearer])

        # count camera frames in a row; a nearby beam last frame also counts (bearings drift)
        if new_camera_frame:
            self.persist = np.where(seen, self._dilate(self.persist, w) + 1, 0)
        trusted = seen & (self.persist >= self.params.persist_frames)

        # write trusted depth points into the fused scan (never farther than the LiDAR)
        fused[trusted] = np.minimum(lidar[trusted], camera_only[trusted])
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
        # zero-pad, stack every shifted copy and take the maximum across the window
        padded = np.pad(counts, w)
        return np.max(np.stack([padded[k:k + counts.size] for k in range(2 * w + 1)]), axis=0)
