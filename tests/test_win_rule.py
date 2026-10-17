"""The ablation win rule and how the sweep combines winners."""

from __future__ import annotations

import csv
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from vision.sweep import _winner_overrides, ablation_wins, merge_overrides, run_sweep

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize(
    ("mean", "baseline_mean", "baseline_std", "margin", "wins"),
    [
        pytest.param(0.5, 0.5, 0.0, 0.0, False, id="tie-at-zero-margin"),
        pytest.param(0.5, 0.5, 0.1, 1.0, False, id="tie-with-margin"),
        pytest.param(0.4, 0.5, 0.0, 0.0, False, id="loss"),
        pytest.param(0.5 + 1e-12, 0.5, 0.0, 100.0, True, id="zero-std-any-gain"),
        pytest.param(0.51, 0.5, 0.2, 0.0, True, id="zero-margin-gain"),
        pytest.param(0.49, 0.5, 0.2, 0.0, False, id="zero-margin-loss"),
        pytest.param(0.75, 0.5, 0.25, 1.0, True, id="gain-at-the-margin"),
        pytest.param(0.7499, 0.5, 0.25, 1.0, False, id="gain-below-the-margin"),
        pytest.param(NAN, 0.5, 0.1, 0.0, False, id="nan-mean"),
        pytest.param(0.6, NAN, 0.1, 0.0, False, id="nan-baseline-mean"),
        pytest.param(0.6, 0.5, NAN, 0.0, False, id="nan-baseline-std"),
        pytest.param(INF, 0.5, 0.1, 0.0, False, id="infinite-mean"),
    ],
)
def test_ablation_wins(
    mean: float, baseline_mean: float, baseline_std: float, margin: float, wins: bool
) -> None:
    assert ablation_wins(mean, baseline_mean, baseline_std, margin) is wins


@pytest.mark.parametrize("margin", [NAN, -1.0, INF])
def test_invalid_margin_is_rejected(margin: float) -> None:
    with pytest.raises(ValueError, match="win_margin_std"):
        ablation_wins(0.6, 0.5, 0.1, margin)


def _create(tree: dict[str, Any]) -> DictConfig:
    cfg = OmegaConf.create(tree)
    assert isinstance(cfg, DictConfig)
    return cfg


def _cfg(margin: float) -> DictConfig:
    return _create(
        {
            "sweep": {
                "win_margin_std": margin,
                "ablations": {"a": ["x.a=1"], "b": ["x.b=1"], "c": ["x.c=1"]},
            }
        }
    )


def test_winner_overrides_collects_only_winners() -> None:
    baseline = {"mean": 0.5, "std": 0.0}
    ablations = {
        "a": {"mean": 0.6},
        "b": {"mean": 0.5},
        "c": {"mean": NAN},
    }
    assert _winner_overrides(_cfg(0.0), baseline, ablations) == ["x.a=1"]


def test_winner_overrides_rejects_invalid_margin_without_ablations() -> None:
    with pytest.raises(ValueError, match="win_margin_std"):
        _winner_overrides(_cfg(-1.0), {"mean": 0.5, "std": 0.0}, {})


def test_sweep_at_zero_margin_combines_only_strict_gains(smoke_cfg: DictConfig) -> None:
    """The smoke config's dummy backend ignores `model.size`, so `size_s` ties."""
    smoke_cfg.sweep.stages = ["baseline", "ablations", "combine"]

    run_sweep(smoke_cfg)

    with open(smoke_cfg.sweep.results_csv) as f:
        means = {r["exp"]: float(r["mean"]) for r in csv.DictReader(f)}
    assert means["abl_size_s"] == means["baseline"]
    expected = [
        name
        for name in ("sahi", "tracking", "size_s")
        if means[f"abl_{name}"] > means["baseline"]
    ]
    assert ("combined" in means) == bool(expected)


def test_merge_overrides_keeps_order_and_dedupes_identical_items() -> None:
    merged = merge_overrides(
        {"a": ["x.a=1", "x.b=2"], "b": ["x.b=2", "x.c=3"], "c": []}
    )
    assert merged == ["x.a=1", "x.b=2", "x.c=3"]


def test_merge_overrides_rejects_a_key_set_to_two_values() -> None:
    with pytest.raises(ValueError, match=r"'x.b'.*a sets '1'.*b sets '2'"):
        merge_overrides({"a": ["x.b=1"], "b": ["x.b=2"]})


def test_winners_that_conflict_raise_at_combine_time() -> None:
    cfg = _create(
        {
            "sweep": {
                "win_margin_std": 0.0,
                "ablations": {"a": ["m.size=s"], "b": ["m.size=m"], "c": ["m.k=1"]},
            }
        }
    )
    baseline = {"mean": 0.5, "std": 0.0}
    both = {"a": {"mean": 0.6}, "b": {"mean": 0.7}, "c": {"mean": 0.6}}
    with pytest.raises(ValueError, match=r"ablation a.*ablation b"):
        _winner_overrides(cfg, baseline, both)
    # A conflicting ablation that does not win is not a conflict.
    one = {"a": {"mean": 0.6}, "b": {"mean": 0.5}, "c": {"mean": 0.6}}
    assert _winner_overrides(cfg, baseline, one) == ["m.size=s", "m.k=1"]


def test_sweep_stops_when_two_winners_set_one_key_differently(
    smoke_cfg: DictConfig,
) -> None:
    smoke_cfg.sweep.stages = ["baseline", "ablations", "combine"]
    smoke_cfg.model.dummy_recall = 0.3
    smoke_cfg.sweep.ablations = {
        "recall_hi": ["model.dummy_recall=0.95"],
        "recall_max": ["model.dummy_recall=0.99"],
    }
    with pytest.raises(ValueError, match=r"model\.dummy_recall"):
        run_sweep(smoke_cfg)


def test_sweep_combines_winners_that_agree_on_a_key(smoke_cfg: DictConfig) -> None:
    smoke_cfg.sweep.stages = ["baseline", "ablations", "combine"]
    smoke_cfg.model.dummy_recall = 0.3
    smoke_cfg.sweep.ablations = {
        "recall_hi": ["model.dummy_recall=0.95"],
        "recall_hi_sahi": ["model.dummy_recall=0.95", "sahi.enabled=true"],
    }
    run_sweep(smoke_cfg)
    with open(smoke_cfg.sweep.results_csv) as f:
        assert "combined" in {r["exp"] for r in csv.DictReader(f)}
