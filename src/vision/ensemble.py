"""Rotated Weighted Box Fusion for same-class OBB clusters."""

from __future__ import annotations

import numpy as np

from .geometry import DEFAULT_CONVENTION, AngleConvention, rotated_iou


def _fuse(cluster: np.ndarray) -> np.ndarray:
    w = cluster[:, 0]
    wsum = w.sum() or 1.0
    cls = int(np.bincount(cluster[:, 1].astype(int)).argmax())
    geo = (cluster[:, 2:6] * w[:, None]).sum(axis=0) / wsum
    # OBB angles are 180-degree periodic.
    ang2 = np.deg2rad(cluster[:, 6] * 2.0)
    mean_ang = (
        np.rad2deg(np.arctan2((np.sin(ang2) * w).sum(), (np.cos(ang2) * w).sum())) / 2.0
    )
    score = w.max()
    return np.array([score, cls, geo[0], geo[1], geo[2], geo[3], mean_ang % 180])


def wbf_frame(
    preds_list: list[np.ndarray],
    iou_thr: float = 0.55,
    conv: AngleConvention = DEFAULT_CONVENTION,
) -> np.ndarray:
    rows = [np.asarray(arr, dtype=np.float64).reshape(-1, 7) for arr in preds_list]
    allp = np.vstack(rows) if rows else np.zeros((0, 7))
    if len(allp) == 0:
        return np.zeros((0, 7))
    order = np.argsort(-allp[:, 0])
    allp = allp[order]
    used = np.zeros(len(allp), dtype=bool)
    fused = []
    for i in range(len(allp)):
        if used[i]:
            continue
        members = [i]
        used[i] = True
        for j in range(i + 1, len(allp)):
            if used[j] or allp[j, 1] != allp[i, 1]:
                continue
            if rotated_iou(allp[i, 2:7], allp[j, 2:7], conv) >= iou_thr:
                members.append(j)
                used[j] = True
        fused.append(_fuse(allp[members]))
    return np.asarray(fused, dtype=np.float64).reshape(-1, 7)


def wbf(
    pred_dicts: list[dict[str, np.ndarray]],
    iou_thr: float = 0.55,
    conv: AngleConvention = DEFAULT_CONVENTION,
) -> dict[str, np.ndarray]:
    frame_ids: set[str] = set().union(*[set(d.keys()) for d in pred_dicts])
    return {
        fid: wbf_frame(
            [d.get(fid, np.zeros((0, 7))) for d in pred_dicts], iou_thr, conv
        )
        for fid in frame_ids
    }
