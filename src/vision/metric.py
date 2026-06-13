"""Local reproduction of Macro AP-rIoU@[0.50:0.80]."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import CLASS_IDS, RIOU_THRESHOLDS
from .geometry import AngleConvention, DEFAULT_CONVENTION, rotated_iou_matrix


def _ap_all_points(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """Continuous AP excludes classes with no ground truth from the mean."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tp = is_tp[order].astype(np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # The precision envelope matches the COCO continuous AP variant.
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _group(records: dict, has_score: bool):
    """Group boxes by the frame and class matching boundary."""
    grouped: dict[tuple, list] = defaultdict(list)
    cls_col = 1 if has_score else 0
    for frame_id, arr in records.items():
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, 7 if has_score else 6)
        for row in arr:
            cls = int(row[cls_col])
            grouped[(frame_id, cls)].append(row)
    return {k: np.asarray(v, dtype=np.float64) for k, v in grouped.items()}


def evaluate(
    preds: dict,
    gts: dict,
    conv: AngleConvention = DEFAULT_CONVENTION,
    thresholds: tuple = RIOU_THRESHOLDS,
    classes: tuple = CLASS_IDS,
) -> dict:
    """
    Compute the official metric contract.

    Matches are per frame and per class. Each ground-truth box can be assigned
    once per threshold. Duplicate predictions and wrong-class predictions are
    false positives.
    """
    pred_g = _group(preds, has_score=True)
    gt_g = _group(gts, has_score=False)

    iou_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    pred_keys = set(pred_g.keys())
    for key in pred_keys:
        p = pred_g[key]
        order = np.argsort(-p[:, 0], kind="mergesort")
        p_sorted = p[order]
        g = gt_g.get(key)
        if g is None or len(g) == 0:
            iou_cache[key] = (p_sorted[:, 0], np.zeros((len(p_sorted), 0)))
            continue
        m = rotated_iou_matrix(p_sorted[:, 2:7], g[:, 1:6], conv=conv)
        iou_cache[key] = (p_sorted[:, 0], m)

    n_gt_per_cls = defaultdict(int)
    for (_, cls), g in gt_g.items():
        n_gt_per_cls[cls] += len(g)

    per_class_threshold: dict[int, dict[float, float]] = {}
    per_class: dict[int, float] = {}

    for cls in classes:
        entries = []
        for (frame_id, c), (scores, _m) in iou_cache.items():
            if c != cls:
                continue
            for li, sc in enumerate(scores):
                entries.append((sc, frame_id, li))
        entries.sort(key=lambda e: -e[0])
        scores_arr = np.array([e[0] for e in entries], dtype=np.float64)

        thr_aps = {}
        for thr in thresholds:
            assigned: dict = defaultdict(set)
            is_tp = np.zeros(len(entries), dtype=bool)
            for i, (_sc, frame_id, li) in enumerate(entries):
                _, m = iou_cache[(frame_id, cls)]
                if m.shape[1] == 0:
                    continue
                ious = m[li].copy()
                for used in assigned[frame_id]:
                    ious[used] = -1.0
                best = int(np.argmax(ious)) if ious.size else -1
                if best >= 0 and ious[best] >= thr:
                    is_tp[i] = True
                    assigned[frame_id].add(best)
            thr_aps[thr] = _ap_all_points(scores_arr, is_tp, n_gt_per_cls[cls])

        per_class_threshold[cls] = thr_aps
        vals = [v for v in thr_aps.values() if not np.isnan(v)]
        per_class[cls] = float(np.mean(vals)) if vals else float("nan")

    class_means = [per_class[c] for c in classes if not np.isnan(per_class[c])]
    score = float(np.mean(class_means)) if class_means else float("nan")
    return {
        "score": score,
        "per_class": per_class,
        "per_class_threshold": per_class_threshold,
    }
