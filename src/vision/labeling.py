"""Converters between project IR and X-AnyLabeling JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import CLASSES
from .geometry import (
    AngleConvention,
    DEFAULT_CONVENTION,
    corners_to_obb,
    obb_to_corners,
)

_NAME_TO_ID = {v: k for k, v in CLASSES.items()}


def export_pseudolabels(
    preds: dict[str, np.ndarray],
    images_dir: str | Path,
    out_dir: str | Path,
    conf_thr: float = 0.25,
    conv: AngleConvention = DEFAULT_CONVENTION,
) -> Path:
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fid, arr in preds.items():
        shapes = []
        for sc, cls, cx, cy, w, h, ang in np.asarray(arr).reshape(-1, 7):
            if sc < conf_thr:
                continue
            pts = obb_to_corners(cx, cy, w, h, ang, conv).tolist()
            shapes.append(
                {
                    "label": CLASSES.get(int(cls), str(int(cls))),
                    "points": pts,
                    "shape_type": "rotation",
                    "direction": float(np.deg2rad(ang)),
                    "score": float(sc),
                    "flags": {},
                }
            )
        doc = {
            "version": "2.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": f"{fid}.jpg",
            "imageHeight": None,
            "imageWidth": None,
        }
        (out_dir / f"{fid}.json").write_text(json.dumps(doc, indent=2))
    return out_dir


def import_corrections(
    json_dir: str | Path, conv: AngleConvention = DEFAULT_CONVENTION
) -> dict[str, np.ndarray]:
    """X-AnyLabeling rotation shapes are recovered through the project OBB convention."""
    json_dir = Path(json_dir)
    out = {}
    for jp in sorted(json_dir.glob("*.json")):
        doc = json.loads(jp.read_text())
        rows = []
        for sh in doc.get("shapes", []):
            cls = _NAME_TO_ID.get(sh["label"])
            if cls is None:
                continue
            cx, cy, w, h, ang = corners_to_obb(np.asarray(sh["points"]), conv)
            rows.append([cls, cx, cy, w, h, ang])
        out[jp.stem] = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    return out


def rank_by_uncertainty(preds: dict[str, np.ndarray]) -> list[str]:
    """Review empty and low-confidence frames before confident detections."""

    def key(item):
        arr = np.asarray(item[1]).reshape(-1, 7)
        if len(arr) == 0:
            return (0.0, 0)
        return (float(arr[:, 0].max()), len(arr))

    return [fid for fid, _ in sorted(preds.items(), key=key)]
