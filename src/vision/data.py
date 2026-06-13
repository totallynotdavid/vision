"""Inspect real data and derive config values that depend on the dataset."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from . import CLASS_IDS, CLASSES
from .convert import load_gts


def class_counts(gts: dict[str, np.ndarray]) -> dict[int, int]:
    counts = defaultdict(int)
    for arr in gts.values():
        for row in np.asarray(arr).reshape(-1, 6):
            counts[int(row[0])] += 1
    return {c: counts.get(c, 0) for c in CLASS_IDS}


def effective_num_weights(
    counts: dict[int, int], beta: float = 0.999
) -> dict[int, float]:
    """Class-balanced loss weights via 'effective number of samples' (Cui et al.).
    Preferred over inverse-frequency, which over-corrects and hurts common classes.
    """
    w = {}
    for c, n in counts.items():
        eff = (1.0 - beta**n) / (1.0 - beta) if n > 0 else 1.0
        w[c] = 1.0 / eff if eff > 0 else 0.0
    mean = np.mean([v for v in w.values() if v > 0]) or 1.0
    return {c: v / mean for c, v in w.items()}


def repeat_factors(gts: dict[str, np.ndarray], thr: float = 0.001) -> dict[str, float]:
    """Image-level Repeat Factor Sampling (Gupta et al.). An image's repeat
    factor is the max over the categories it contains of max(1, sqrt(thr/f_c)).
    """
    counts = class_counts(gts)
    total = sum(counts.values()) or 1
    freq = {c: counts[c] / total for c in CLASS_IDS}
    cat_rf = {
        c: max(1.0, np.sqrt(thr / freq[c])) if freq[c] > 0 else 1.0 for c in CLASS_IDS
    }
    out = {}
    for fid, arr in gts.items():
        present = {int(r[0]) for r in np.asarray(arr).reshape(-1, 6)}
        out[fid] = max((cat_rf[c] for c in present), default=1.0)
    return out


def angle_stats(gts: dict[str, np.ndarray]) -> dict:
    angs = [r[5] for arr in gts.values() for r in np.asarray(arr).reshape(-1, 6)]
    if not angs:
        return {}
    angs = np.array(angs)
    return {
        "min": float(angs.min()),
        "max": float(angs.max()),
        "mean": float(angs.mean()),
        "n": int(len(angs)),
    }


def image_size(images_dir: str | Path) -> tuple[int, int] | None:
    import cv2

    for p in sorted(Path(images_dir).glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img = cv2.imread(str(p))
            if img is not None:
                return img.shape[1], img.shape[0]
    return None


def suggest_tile_size(img_wh: tuple[int, int] | None) -> int | None:
    """Enable SAHI only when the image is larger than the model input footprint."""
    if img_wh is None:
        return None
    long_side = max(img_wh)
    if long_side <= 1280:
        return None
    return 1024 if long_side > 2048 else 640


def inspect(
    raw_dir: str | Path, csv_name: str = "train.csv", images_subdir: str = "images"
) -> dict:
    raw_dir = Path(raw_dir)
    gts = load_gts(raw_dir / csv_name)
    counts = class_counts(gts)
    img_wh = image_size(raw_dir / images_subdir)
    report = {
        "n_frames": len(gts),
        "image_size": img_wh,
        "suggested_tile": suggest_tile_size(img_wh),
        "class_counts": counts,
        "loss_weights": effective_num_weights(counts),
        "angle": angle_stats(gts),
    }
    return report


def _print_report(r: dict) -> None:
    print(
        f"frames: {r['n_frames']}   image_size: {r['image_size']}   "
        f"suggested SAHI tile: {r['suggested_tile']}"
    )
    print(f"angle: {r['angle']}")
    print("class                count     loss_w")
    for c in CLASS_IDS:
        print(
            f"  {c} {CLASSES[c]:<14} {r['class_counts'][c]:>7}   {r['loss_weights'][c]:.3f}"
        )


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    path = sys.argv[2] if len(sys.argv) > 2 else "data/raw"
    if cmd == "inspect":
        _print_report(inspect(path))
    else:
        raise SystemExit(f"unknown command: {cmd}")
