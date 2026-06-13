"""Conversion boundaries between competition cells, project IR, and YOLO labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES, CLASS_IDS
from .geometry import AngleConvention, DEFAULT_CONVENTION, is_valid_box, obb_to_corners

NONE_TOKEN = "none"


def parse_gt_cell(cell: str) -> np.ndarray:
    """Empty ground-truth cells normalize to an empty IR array."""
    cell = (cell or "").strip()
    if not cell or cell.lower() == NONE_TOKEN:
        return np.zeros((0, 6), dtype=np.float64)
    rows = []
    for det in cell.split(";"):
        det = det.strip()
        if not det:
            continue
        vals = [float(x) for x in det.split()]
        rows.append(vals[:6])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 6)


def parse_pred_cell(cell: str) -> np.ndarray:
    """Empty prediction cells normalize to an empty IR array."""
    cell = (cell or "").strip()
    if not cell or cell.lower() == NONE_TOKEN:
        return np.zeros((0, 7), dtype=np.float64)
    rows = []
    for det in cell.split(";"):
        det = det.strip()
        if not det:
            continue
        vals = [float(x) for x in det.split()]
        rows.append(vals[:7])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7)


def format_pred_cell(pred: np.ndarray, drop_invalid: bool = True) -> str:
    """Invalid predictions are omitted before writing a submission cell."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 7)
    parts = []
    for sc, cls, cx, cy, w, h, ang in pred:
        if drop_invalid and not (
            is_valid_box(cx, cy, w, h, ang) and 1 <= int(cls) <= 9
        ):
            continue
        parts.append(f"{sc:.4f} {int(cls)} {cx:.2f} {cy:.2f} {w:.2f} {h:.2f} {ang:.2f}")
    return ";".join(parts) if parts else NONE_TOKEN


def load_gts(
    csv_path: str | Path,
    id_col: str = "Id",
    target_col: str = "Target",
) -> dict[str, np.ndarray]:
    """Column names stay configurable until the real `train.csv` schema is known."""
    df = pd.read_csv(csv_path)
    return {str(r[id_col]): parse_gt_cell(str(r[target_col])) for _, r in df.iterrows()}


def load_preds(
    csv_path: str | Path, id_col: str = "Id", target_col: str = "Target"
) -> dict:
    df = pd.read_csv(csv_path)
    return {
        str(r[id_col]): parse_pred_cell(str(r[target_col])) for _, r in df.iterrows()
    }


def gt_to_yolo_lines(
    gt: np.ndarray, img_w: int, img_h: int, conv: AngleConvention = DEFAULT_CONVENTION
) -> list[str]:
    """YOLO labels use normalized corners and zero-based class indices."""
    lines = []
    for cls, cx, cy, w, h, ang in np.asarray(gt, dtype=np.float64).reshape(-1, 6):
        if not is_valid_box(cx, cy, w, h, ang):
            continue
        idx = int(cls) - 1
        c = obb_to_corners(cx, cy, w, h, ang, conv)
        c = c / np.array([img_w, img_h], dtype=np.float64)
        c = np.clip(c, 0.0, 1.0).reshape(-1)
        coords = " ".join(f"{v:.6f}" for v in c)
        lines.append(f"{idx} {coords}")
    return lines


def write_dataset_yaml(out_dir: str | Path, train_dir: str, val_dir: str) -> Path:
    """Ultralytics expects class names ordered by zero-based index."""
    out_dir = Path(out_dir)
    names = "\n".join(f"  {i}: {CLASSES[cid]}" for i, cid in enumerate(CLASS_IDS))
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: {train_dir}\n"
        f"val: {val_dir}\n"
        f"names:\n{names}\n"
    )
    return yaml_path


def build_submission(
    preds: dict[str, np.ndarray], frame_order: list[str], out_path: str | Path
) -> Path:
    """Every requested frame gets a submission row, even when no boxes are predicted."""
    rows = [
        {"Id": fid, "Target": format_pred_cell(preds.get(fid, np.zeros((0, 7))))}
        for fid in frame_order
    ]
    out_path = Path(out_path)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
