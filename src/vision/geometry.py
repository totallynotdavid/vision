"""Oriented-box geometry and rotated IoU share one angle convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon


@dataclass(frozen=True)
class AngleConvention:
    """Positive-angle direction must match the calibrated annotation convention."""

    ccw: bool = True

    def theta_rad(self, angle_deg: float) -> float:
        sign = 1.0 if self.ccw else -1.0
        return sign * np.deg2rad(angle_deg)


DEFAULT_CONVENTION = AngleConvention()


def obb_to_corners(
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    conv: AngleConvention = DEFAULT_CONVENTION,
) -> np.ndarray:
    """Angle 0 places width on the image x-axis and height on the y-axis."""
    theta = conv.theta_rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = w / 2.0, h / 2.0
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy], dtype=np.float64)


def corners_to_obb(
    corners: np.ndarray, conv: AngleConvention = DEFAULT_CONVENTION
) -> tuple[float, float, float, float, float]:
    """Recover the project OBB convention from OpenCV's rectangle convention."""
    import cv2

    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    # OpenCV's angle sign is normalized back into the calibrated convention.
    angle_deg = ang if conv.ccw else -ang
    return float(cx), float(cy), float(w), float(h), float(angle_deg)


def obb_to_polygon(
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    conv: AngleConvention = DEFAULT_CONVENTION,
) -> Polygon:
    return Polygon(obb_to_corners(cx, cy, w, h, angle_deg, conv))


def rotated_iou(box_a, box_b, conv: AngleConvention = DEFAULT_CONVENTION) -> float:
    """Degenerate or non-overlapping boxes have rIoU 0.0.

    Shapely repairs invalid polygons before measuring overlap.
    """
    pa = obb_to_polygon(*box_a, conv=conv)
    pb = obb_to_polygon(*box_b, conv=conv)
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    if pa.area <= 0 or pb.area <= 0:
        return 0.0
    inter = pa.intersection(pb).area
    if inter <= 0:
        return 0.0
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def rotated_iou_matrix(
    boxes_a: np.ndarray, boxes_b: np.ndarray, conv: AngleConvention = DEFAULT_CONVENTION
) -> np.ndarray:
    """Return pairwise rIoU for project OBB arrays."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 5)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 5)
    polys_a = [obb_to_polygon(*b, conv=conv) for b in boxes_a]
    polys_b = [obb_to_polygon(*b, conv=conv) for b in boxes_b]
    polys_a = [p if p.is_valid else p.buffer(0) for p in polys_a]
    polys_b = [p if p.is_valid else p.buffer(0) for p in polys_b]
    out = np.zeros((len(polys_a), len(polys_b)), dtype=np.float64)
    for i, pa in enumerate(polys_a):
        if pa.area <= 0:
            continue
        for j, pb in enumerate(polys_b):
            if pb.area <= 0:
                continue
            inter = pa.intersection(pb).area
            if inter <= 0:
                continue
            union = pa.area + pb.area - inter
            if union > 0:
                out[i, j] = inter / union
    return out


def is_valid_box(cx, cy, w, h, angle_deg) -> bool:
    """Competition boxes must be finite and have positive area."""
    vals = (cx, cy, w, h, angle_deg)
    if not all(np.isfinite(v) for v in vals):
        return False
    return w > 0 and h > 0
