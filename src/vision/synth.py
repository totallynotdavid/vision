"""Synthetic clip data exercises the full OBB pipeline without real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASS_IDS
from .geometry import DEFAULT_CONVENTION, obb_to_corners

# Per-class size priors keep synthetic objects visually distinct.
_SIZE = {
    1: (60, 30),
    2: (90, 40),
    3: (110, 45),
    4: (130, 50),
    5: (170, 55),
    6: (240, 55),
    7: (160, 55),
    8: (45, 35),
    9: (35, 20),
}


def make_synthetic(
    out_dir: str | Path,
    n_clips: int = 4,
    frames_per_clip: int = 8,
    img_size: tuple[int, int] = (640, 384),
    objs_per_frame: int = 6,
    seed: int = 0,
) -> dict:
    """Write images and a train.csv-like file using the default angle convention."""
    import cv2

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    W, H = img_size
    rng = np.random.default_rng(seed)

    rows = []
    for clip in range(n_clips):
        # Persistent clip objects create near-duplicate frames for split tests.
        n_obj = objs_per_frame
        cls = rng.choice(CLASS_IDS, size=n_obj)
        cx = rng.uniform(0.15 * W, 0.85 * W, n_obj)
        cy = rng.uniform(0.15 * H, 0.85 * H, n_obj)
        ang = rng.uniform(0, 180, n_obj)
        vel = rng.uniform(-4, 4, (n_obj, 2))
        for fr in range(frames_per_clip):
            img = np.full((H, W, 3), 40, dtype=np.uint8)
            gt = []
            for i in range(n_obj):
                w, h = _SIZE[int(cls[i])]
                w *= rng.uniform(0.85, 1.15)
                h *= rng.uniform(0.85, 1.15)
                pcx = float(np.clip(cx[i] + vel[i, 0] * fr, 0, W))
                pcy = float(np.clip(cy[i] + vel[i, 1] * fr, 0, H))
                pang = float((ang[i] + fr * 2) % 180)
                corners = obb_to_corners(pcx, pcy, w, h, pang, DEFAULT_CONVENTION)
                color = tuple(int(x) for x in rng.integers(80, 255, 3))
                cv2.fillPoly(img, [corners.astype(np.int32)], color)
                gt.append([int(cls[i]), pcx, pcy, w, h, pang])
            frame_id = f"clip{clip:03d}_{fr:03d}"
            cv2.imwrite(str(img_dir / f"{frame_id}.jpg"), img)
            gt_arr = np.array(gt, dtype=np.float64)
            cell = ";".join(
                f"{int(c)} {x:.2f} {y:.2f} {ww:.2f} {hh:.2f} {a:.2f}"
                for c, x, y, ww, hh, a in gt_arr
            )
            rows.append({"Id": frame_id, "Target": cell or "none"})

    csv_path = out_dir / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"csv": csv_path, "images": img_dir, "img_w": W, "img_h": H}


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/synth"
    info = make_synthetic(out)
    print(f"synthetic dataset -> {info['csv']} ({info['img_w']}x{info['img_h']})")
