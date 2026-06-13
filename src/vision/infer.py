"""Real-backend inference paths return project IR predictions."""

from __future__ import annotations

import numpy as np


def _yolo_to_ir(obb, conv_ccw: bool = True) -> np.ndarray:
    """Ultralytics OBB output uses zero-based classes and radians."""
    if obb is None or len(obb) == 0:
        return np.zeros((0, 7), dtype=np.float64)
    xywhr = obb.xywhr.cpu().numpy()
    conf = obb.conf.cpu().numpy()
    cls = obb.cls.cpu().numpy()
    rows = []
    for (cx, cy, w, h, r), sc, c in zip(xywhr, conf, cls):
        ang = np.rad2deg(r if conv_ccw else -r)
        rows.append(
            [
                float(sc),
                int(c) + 1,
                float(cx),
                float(cy),
                float(w),
                float(h),
                float(ang),
            ]
        )
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7)


def predict_plain(ckpt: str, frames: list[tuple[str, str]], cfg: dict) -> dict:
    from ultralytics import YOLO

    mcfg = cfg.get("model", {})
    model = YOLO(ckpt)
    conf = float(mcfg.get("conf", 0.05))  # AP favors recall before score sorting.
    imgsz = int(mcfg.get("imgsz", 640))
    out = {}
    paths = [p for _, p in frames]
    results = model.predict(
        paths,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
        augment=bool(cfg.get("tta", {}).get("enabled", False)),
    )
    for (fid, _), res in zip(frames, results):
        out[fid] = _yolo_to_ir(getattr(res, "obb", None))
    return out


def predict_sahi(ckpt: str, frames: list[tuple[str, str]], cfg: dict) -> dict:
    """SAHI deduplicates horizontal envelopes before OBBs are recovered."""
    import cv2
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    scfg = cfg.get("sahi", {})
    mcfg = cfg.get("model", {})
    tile = int(scfg.get("tile", 640))
    overlap = float(scfg.get("overlap", 0.2))
    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=ckpt,
        confidence_threshold=float(mcfg.get("conf", 0.05)),
    )
    out = {}
    for fid, path in frames:
        result = get_sliced_prediction(
            path,
            det_model,
            slice_height=tile,
            slice_width=tile,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )
        rows = []
        for op in result.object_prediction_list:
            seg = op.mask.segmentation[0] if op.mask is not None else None
            if seg is not None and len(seg) >= 8:
                pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
            else:
                # SAHI may return no mask, so the horizontal box becomes an OBB fallback.
                x1, y1, x2, y2 = op.bbox.to_xyxy()
                cx, cy, w, h, ang = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.0
            rows.append(
                [
                    op.score.value,
                    int(op.category.id) + 1,
                    float(cx),
                    float(cy),
                    float(w),
                    float(h),
                    float(ang),
                ]
            )
        out[fid] = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    return out
