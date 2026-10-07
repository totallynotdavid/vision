"""The ablation win rule: ties, zero baseline std, margin boundary and NaN."""

from __future__ import annotations

import csv

import pytest
from omegaconf import OmegaConf

from vision.sweep import _winner_overrides, ablation_wins, merge_overrides, run_sweep
from vision.synth import make_synthetic

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize("margin", [0.0, 1.0, 2.5])
def test_tie_never_wins(margin):
    assert not ablation_wins(0.5, 0.5, 0.0, margin)
    assert not ablation_wins(0.5, 0.5, 0.1, margin)


@pytest.mark.parametrize("margin", [0.0, 1.0, 100.0])
def test_zero_baseline_std_lets_any_strict_gain_win(margin):
    assert ablation_wins(0.5 + 1e-12, 0.5, 0.0, margin)


def test_zero_margin_needs_a_strict_gain_only():
    assert ablation_wins(0.51, 0.5, 0.2, 0.0)
    assert not ablation_wins(0.49, 0.5, 0.2, 0.0)


def test_gain_exactly_at_the_margin_wins():
    assert ablation_wins(0.75, 0.5, 0.25, 1.0)
    assert not ablation_wins(0.7499, 0.5, 0.25, 1.0)


def test_loss_never_wins():
    assert not ablation_wins(0.4, 0.5, 0.0, 0.0)


@pytest.mark.parametrize(
    ("mean", "baseline_mean", "baseline_std"),
    [
        (NAN, 0.5, 0.1),
        (0.6, NAN, 0.1),
        (0.6, 0.5, NAN),
        (NAN, NAN, NAN),
        (INF, 0.5, 0.1),
    ],
)
@pytest.mark.parametrize("margin", [0.0, 1.0])
def test_non_finite_input_never_wins(mean, baseline_mean, baseline_std, margin):
    assert not ablation_wins(mean, baseline_mean, baseline_std, margin)


@pytest.mark.parametrize("margin", [NAN, -1.0, INF])
def test_invalid_margin_is_rejected(margin):
    with pytest.raises(ValueError, match="win_margin_std"):
        ablation_wins(0.6, 0.5, 0.1, margin)


def _cfg(margin):
    return OmegaConf.create(
        {
            "sweep": {
                "win_margin_std": margin,
                "ablations": {"a": ["x.a=1"], "b": ["x.b=1"], "c": ["x.c=1"]},
            }
        }
    )


def test_winner_overrides_collects_only_winners():
    baseline = {"mean": 0.5, "std": 0.0}
    ablations = {
        "a": {"mean": 0.6},
        "b": {"mean": 0.5},
        "c": {"mean": NAN},
    }
    assert _winner_overrides(_cfg(0.0), baseline, ablations) == ["x.a=1"]
    assert _winner_overrides(_cfg(1.0), baseline, ablations) == ["x.a=1"]


def test_winner_overrides_rejects_invalid_margin_without_ablations():
    with pytest.raises(ValueError, match="win_margin_std"):
        _winner_overrides(_cfg(-1.0), {"mean": 0.5, "std": 0.0}, {})


def test_winner_overrides_with_nan_baseline_finds_none():
    baseline = {"mean": NAN, "std": NAN}
    ablations = {"a": {"mean": 0.6}}
    assert _winner_overrides(_cfg(0.0), baseline, ablations) == []


def test_sweep_at_zero_margin_combines_only_strict_gains(tmp_path):
    """The smoke config's dummy backend ignores `model.size`, so `size_s` ties."""
    synth = tmp_path / "synth"
    make_synthetic(synth, n_clips=4, frames_per_clip=6, seed=0)
    cfg = OmegaConf.load("configs/smoke.yaml")
    cfg.data.raw_dir = str(synth)
    cfg.cv.folds_file = str(tmp_path / "folds.json")
    cfg.sweep.results_csv = str(tmp_path / "results.csv")
    cfg.sweep.work_dir = str(tmp_path / "runs")
    cfg.sweep.stages = ["baseline", "ablations", "combine"]
    cfg.sweep.win_margin_std = 0.0

    run_sweep(cfg)

    with open(cfg.sweep.results_csv) as f:
        means = {r["exp"]: float(r["mean"]) for r in csv.DictReader(f)}
    assert means["abl_size_s"] == means["baseline"]
    expected = [
        name
        for name in ("sahi", "tracking", "size_s")
        if means[f"abl_{name}"] > means["baseline"]
    ]
    assert ("combined" in means) == bool(expected)


def test_merge_overrides_keeps_order_and_dedupes_identical_items():
    merged = merge_overrides(
        {"a": ["x.a=1", "x.b=2"], "b": ["x.b=2", "x.c=3"], "c": []}
    )
    assert merged == ["x.a=1", "x.b=2", "x.c=3"]


def test_merge_overrides_rejects_a_key_set_to_two_values():
    with pytest.raises(ValueError, match=r"'x.b'.*a sets '1'.*b sets '2'"):
        merge_overrides({"a": ["x.b=1"], "b": ["x.b=2"]})


def test_winners_that_conflict_raise_at_combine_time():
    cfg = OmegaConf.create(
        {
            "sweep": {
                "win_margin_std": 0.0,
                "ablations": {"a": ["m.size=s"], "b": ["m.size=m"], "c": ["m.k=1"]},
            }
        }
    )
    baseline = {"mean": 0.5, "std": 0.0}
    both = {"a": {"mean": 0.6}, "b": {"mean": 0.7}, "c": {"mean": 0.6}}
    with pytest.raises(ValueError, match="ablation a.*ablation b"):
        _winner_overrides(cfg, baseline, both)
    # A conflicting ablation that does not win is not a conflict.
    one = {"a": {"mean": 0.6}, "b": {"mean": 0.5}, "c": {"mean": 0.6}}
    assert _winner_overrides(cfg, baseline, one) == ["m.size=s", "m.k=1"]


def _smoke_cfg(tmp_path):
    synth = tmp_path / "synth"
    make_synthetic(synth, n_clips=4, frames_per_clip=6, seed=0)
    cfg = OmegaConf.load("configs/smoke.yaml")
    cfg.data.raw_dir = str(synth)
    cfg.cv.folds_file = str(tmp_path / "folds.json")
    cfg.sweep.results_csv = str(tmp_path / "results.csv")
    cfg.sweep.work_dir = str(tmp_path / "runs")
    cfg.sweep.stages = ["baseline", "ablations", "combine"]
    cfg.sweep.win_margin_std = 0.0
    cfg.model.dummy_recall = 0.3
    return cfg


def test_sweep_stops_when_two_winners_set_one_key_differently(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    cfg.sweep.ablations = {
        "recall_hi": ["model.dummy_recall=0.95"],
        "recall_max": ["model.dummy_recall=0.99"],
    }
    with pytest.raises(ValueError, match="model.dummy_recall"):
        run_sweep(cfg)


def test_sweep_combines_winners_that_agree_on_a_key(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    cfg.sweep.ablations = {
        "recall_hi": ["model.dummy_recall=0.95"],
        "recall_hi_sahi": ["model.dummy_recall=0.95", "sahi.enabled=true"],
    }
    run_sweep(cfg)
    with open(cfg.sweep.results_csv) as f:
        assert "combined" in {r["exp"] for r in csv.DictReader(f)}
