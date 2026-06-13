"""Render OBB overlays for angle-convention calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import CLASSES
from .geometry import AngleConvention, DEFAULT_CONVENTION, obb_to_corners

_PALETTE = [
    (66, 135, 245),
    (245, 66, 66),
    (66, 245, 132),
    (245, 209, 66),
    (197, 66, 245),
    (66, 245, 245),
    (245, 132, 66),
    (132, 66, 245),
    (180, 180, 180),
]


def draw_obbs(
    img: np.ndarray,
    boxes: np.ndarray,
    has_score: bool = False,
    conv: AngleConvention = DEFAULT_CONVENTION,
) -> np.ndarray:
    """Prediction boxes may include a leading confidence score."""
    import cv2

    out = img.copy()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7 if has_score else 6)
    off = 1 if has_score else 0
    for row in boxes:
        cls = int(row[off])
        cx, cy, w, h, ang = row[off + 1 : off + 6]
        corners = obb_to_corners(cx, cy, w, h, ang, conv).astype(np.int32)
        color = _PALETTE[(cls - 1) % len(_PALETTE)]
        cv2.polylines(out, [corners], isClosed=True, color=color, thickness=2)
        label = CLASSES.get(cls, str(cls))
        if has_score:
            label += f" {row[0]:.2f}"
        cv2.putText(
            out,
            label,
            tuple(corners[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def render_calibration(
    gts: dict[str, np.ndarray],
    images_dir: str | Path,
    out_dir: str | Path,
    n: int = 12,
    conv: AngleConvention = DEFAULT_CONVENTION,
) -> Path:
    """Calibration overlays must be inspected before training on real data."""
    import cv2

    images_dir, out_dir = Path(images_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fid in list(gts.keys())[:n]:
        for ext in (".jpg", ".jpeg", ".png"):
            p = images_dir / f"{fid}{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                cv2.imwrite(
                    str(out_dir / f"{fid}.jpg"), draw_obbs(img, gts[fid], conv=conv)
                )
                break
    return out_dir


if __name__ == "__main__":
    import sys

    from .convert import load_gts

    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    gts = load_gts(raw / "train.csv")
    out = render_calibration(gts, raw / "images", "runs/calibration")
    print(f"calibration overlays -> {out}  (inspect these before training!)")
