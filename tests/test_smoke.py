"""
End-to-end smoke test on synthetic data, zero GPU/network. Proves the harness
runs the moment real data arrives. Run: `uv run pytest`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from vision.convert import format_pred_cell, parse_gt_cell, parse_pred_cell
from vision.geometry import obb_to_corners, corners_to_obb, rotated_iou
from vision.metric import evaluate
from vision.splits import clip_of, make_folds
from vision.synth import make_synthetic


def test_rotated_iou_identity_and_disjoint():
    box = (100, 100, 40, 20, 30)
    assert rotated_iou(box, box) == pytest.approx(1.0, abs=1e-6)
    far = (1000, 1000, 40, 20, 30)
    assert rotated_iou(box, far) == 0.0


def test_corner_roundtrip():
    box = (100.0, 80.0, 60.0, 30.0, 25.0)
    corners = obb_to_corners(*box)
    cx, cy, w, h, _ang = corners_to_obb(corners)
    assert (cx, cy) == pytest.approx((100.0, 80.0), abs=1e-3)
    assert sorted((w, h)) == pytest.approx(sorted((60.0, 30.0)), abs=1e-3)


def test_cell_roundtrip():
    assert parse_gt_cell("none").shape == (0, 6)
    assert parse_pred_cell("").shape == (0, 7)
    cell = format_pred_cell(np.array([[0.9, 1, 10, 20, 30, 40, 15.0]]))
    assert cell.startswith("0.9000 1")
    # Invalid boxes must not reach submission output.
    assert format_pred_cell(np.array([[0.9, 1, 10, 20, 0, 40, 15.0]])) == "none"


def test_metric_perfect_and_partial():
    gts = {"f1": np.array([[1, 100, 100, 40, 20, 0.0], [9, 200, 150, 30, 15, 10.0]])}
    perfect = {
        "f1": np.array(
            [[0.99, 1, 100, 100, 40, 20, 0.0], [0.95, 9, 200, 150, 30, 15, 10.0]]
        )
    }
    assert evaluate(perfect, gts)["score"] == pytest.approx(1.0, abs=1e-6)
    empty = {"f1": np.zeros((0, 7))}
    assert evaluate(empty, gts)["score"] == 0.0


def test_clip_grouped_split_has_no_leak():
    frames = [f"clip{c:02d}_{f:03d}" for c in range(6) for f in range(10)]
    folds = make_folds(frames, k=3, seed=0)
    for fold in folds:
        train_clips = {clip_of(f) for f in fold["train"]}
        val_clips = {clip_of(f) for f in fold["val"]}
        assert train_clips.isdisjoint(val_clips), "clip leaked across split!"


def test_full_sweep_end_to_end(tmp_path):
    synth = tmp_path / "synth"
    info = make_synthetic(synth, n_clips=4, frames_per_clip=6, seed=0)
    assert info["csv"].exists()

    cfg = OmegaConf.load("configs/smoke.yaml")
    cfg.data.raw_dir = str(synth)
    cfg.cv.folds_file = str(tmp_path / "folds.json")
    cfg.sweep.results_csv = str(tmp_path / "results.csv")
    cfg.sweep.work_dir = str(tmp_path / "runs")

    from vision.sweep import run_sweep

    run_sweep(cfg)

    results = Path(cfg.sweep.results_csv)
    assert results.exists()
    with open(results) as f:
        rows = list(csv.DictReader(f))
    names = {r["exp"] for r in rows}
    assert "baseline" in names
    for r in rows:
        m = float(r["mean"])
        assert 0.0 <= m <= 1.0
    # The dummy backend should keep the baseline above chance.
    base = next(r for r in rows if r["exp"] == "baseline")
    assert float(base["mean"]) > 0.3
