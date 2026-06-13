"""Temporal post-processing appends interpolated detections within each clip."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import AngleConvention, DEFAULT_CONVENTION, rotated_iou_matrix


def _interp_angle(a0: float, a1: float, t: float) -> float:
    """OBB angles are 180-degree periodic."""
    d = ((a1 - a0 + 90) % 180) - 90
    return (a0 + t * d) % 180


def track_clip(
    preds_by_frame: dict[str, np.ndarray],
    frame_order: list[str],
    iou_thr: float = 0.3,
    max_gap: int = 5,
    interp_score_decay: float = 0.7,
    conv: AngleConvention = DEFAULT_CONVENTION,
) -> dict[str, np.ndarray]:
    """Original detections are preserved. Interpolated boxes are appended."""
    tracks: list[dict] = []
    active: list[int] = []

    for fi, fid in enumerate(frame_order):
        dets = np.asarray(preds_by_frame.get(fid, np.zeros((0, 7)))).reshape(-1, 7)
        new_active = []
        matched_dets = set()
        if active and len(dets):
            prev_boxes = np.array([tracks[t]["last"][2:7] for t in active])
            prev_cls = np.array([tracks[t]["cls"] for t in active])
            iou = rotated_iou_matrix(prev_boxes, dets[:, 2:7], conv=conv)
            # Tracking never associates detections across classes.
            for r, tc in enumerate(prev_cls):
                iou[r, dets[:, 1] != tc] = 0.0
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= iou_thr:
                    t = active[r]
                    tracks[t]["frames"][fi] = dets[c]
                    tracks[t]["last"] = dets[c]
                    new_active.append(t)
                    matched_dets.add(c)
        for c in range(len(dets)):
            if c in matched_dets:
                continue
            tracks.append(
                {"cls": int(dets[c, 1]), "frames": {fi: dets[c]}, "last": dets[c]}
            )
            new_active.append(len(tracks) - 1)
        active = new_active

    out = {
        fid: np.asarray(preds_by_frame.get(fid, np.zeros((0, 7)))).reshape(-1, 7).copy()
        for fid in frame_order
    }
    for tr in tracks:
        fis = sorted(tr["frames"].keys())
        for a, b in zip(fis, fis[1:]):
            gap = b - a
            if gap <= 1 or gap > max_gap + 1:
                continue
            box_a, box_b = tr["frames"][a], tr["frames"][b]
            for fi in range(a + 1, b):
                t = (fi - a) / gap
                interp = box_a.copy()
                interp[0] = min(box_a[0], box_b[0]) * interp_score_decay
                interp[2:5] = (1 - t) * box_a[2:5] + t * box_b[2:5]
                interp[5] = (1 - t) * box_a[5] + t * box_b[5]
                interp[6] = _interp_angle(box_a[6], box_b[6], t)
                fid = frame_order[fi]
                out[fid] = np.vstack([out[fid], interp.reshape(1, 7)])
    return out


def track_all(preds: dict, clips: dict[str, list[str]], cfg: dict) -> dict:
    """Tracking is isolated per ordered clip."""
    tcfg = cfg.get("tracking", {})
    out = dict(preds)
    for _clip, frame_order in clips.items():
        sub = {f: preds.get(f, np.zeros((0, 7))) for f in frame_order}
        tracked = track_clip(
            sub,
            frame_order,
            iou_thr=float(tcfg.get("iou_thr", 0.3)),
            max_gap=int(tcfg.get("max_gap", 5)),
        )
        out.update(tracked)
    return out
