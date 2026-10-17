from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from vision.synth import make_synthetic


@pytest.fixture
def smoke_cfg(tmp_path: Path) -> DictConfig:
    synth = tmp_path / "synth"
    make_synthetic(synth, n_clips=4, frames_per_clip=6, seed=0)
    cfg = OmegaConf.load("configs/smoke.yaml")
    assert isinstance(cfg, DictConfig)
    cfg.data.raw_dir = str(synth)
    cfg.cv.folds_file = str(tmp_path / "folds.json")
    cfg.sweep.results_csv = str(tmp_path / "results.csv")
    cfg.sweep.work_dir = str(tmp_path / "runs")
    return cfg
