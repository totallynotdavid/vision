"""Exercise the CPU pipeline on synthetic data without GPU or network access."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig

from vision.convert import format_pred_cell, parse_gt_cell, parse_pred_cell
from vision.geometry import corners_to_obb, obb_to_corners, rotated_iou
from vision.metric import evaluate
from vision.splits import clip_of, make_folds
from vision.sweep import git_sha, run_sweep


def test_rotated_iou_identity_and_disjoint() -> None:
    box = (100, 100, 40, 20, 30)
    assert rotated_iou(box, box) == pytest.approx(1.0, abs=1e-6)
    far = (1000, 1000, 40, 20, 30)
    assert rotated_iou(box, far) == 0.0


def test_corner_roundtrip() -> None:
    box = (100.0, 80.0, 60.0, 30.0, 25.0)
    corners = obb_to_corners(*box)
    cx, cy, w, h, _ang = corners_to_obb(corners)
    assert (cx, cy) == pytest.approx((100.0, 80.0), abs=1e-3)
    assert sorted((w, h)) == pytest.approx(sorted((60.0, 30.0)), abs=1e-3)


def test_cell_roundtrip() -> None:
    assert parse_gt_cell("none").shape == (0, 6)
    assert parse_pred_cell("").shape == (0, 7)
    cell = format_pred_cell(np.array([[0.9, 1, 10, 20, 30, 40, 15.0]]))
    assert cell.startswith("0.9000 1")
    # Invalid boxes must not reach submission output.
    assert format_pred_cell(np.array([[0.9, 1, 10, 20, 0, 40, 15.0]])) == "none"


def test_metric_perfect_and_partial() -> None:
    gts = {"f1": np.array([[1, 100, 100, 40, 20, 0.0], [9, 200, 150, 30, 15, 10.0]])}
    perfect = {
        "f1": np.array(
            [[0.99, 1, 100, 100, 40, 20, 0.0], [0.95, 9, 200, 150, 30, 15, 10.0]]
        )
    }
    assert evaluate(perfect, gts)["score"] == pytest.approx(1.0, abs=1e-6)
    empty = {"f1": np.zeros((0, 7))}
    assert evaluate(empty, gts)["score"] == 0.0


@pytest.mark.parametrize("failure", ["git_missing", "not_a_repository"])
def test_git_sha_without_a_usable_git_is_nogit(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if failure == "git_missing":
        monkeypatch.setenv("PATH", str(tmp_path))
    else:
        monkeypatch.setenv("GIT_DIR", str(tmp_path / "absent"))
    assert git_sha() == "nogit"


def test_clip_grouped_split_has_no_leak() -> None:
    frames = [f"clip{c:02d}_{f:03d}" for c in range(6) for f in range(10)]
    folds = make_folds(frames, k=3, seed=0)
    for fold in folds:
        train_clips = {clip_of(f) for f in fold["train"]}
        val_clips = {clip_of(f) for f in fold["val"]}
        assert train_clips.isdisjoint(val_clips), "clip leaked across split!"


def test_full_sweep_end_to_end(smoke_cfg: DictConfig) -> None:
    run_sweep(smoke_cfg)

    with open(smoke_cfg.sweep.results_csv) as f:
        rows = list(csv.DictReader(f))
    assert {"baseline", "final"} <= {r["exp"] for r in rows}
    for r in rows:
        assert 0.0 <= float(r["mean"]) <= 1.0
    base = next(r for r in rows if r["exp"] == "baseline")
    assert float(base["mean"]) > 0.3
