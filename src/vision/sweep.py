"""Run the configured A/B sweep and append resumable results."""

from __future__ import annotations

import csv as _csv
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from . import CLASS_IDS
from .backends import make_backend
from .convert import gt_to_yolo_lines, load_gts, write_dataset_yaml
from .data import image_size, repeat_factors
from .ensemble import wbf
from .metric import evaluate
from .splits import clip_of, load_folds, make_folds, save_folds
from .track import track_all


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError, OSError:
        return "nogit"


def config_hash(cfg) -> str:
    return hashlib.sha1(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()[:10]


def load_done(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            done.add((row["exp"], row["config_hash"]))
    return done


def append_result(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _frame_path(cfg, fid: str) -> str:
    return str(Path(cfg.data.raw_dir) / cfg.data.images / f"{fid}{cfg.data.ext}")


def prepare_fold_dataset(cfg, gts, fold, img_wh, work_dir: Path) -> str:
    """Real backends train on per-fold YOLO datasets with optional RFS repeats."""
    import os

    w, h = img_wh
    base = work_dir / f"fold{fold['fold']}_data"
    rfs = repeat_factors(gts, float(cfg.sampling.rfs_thr)) if cfg.sampling.rfs else None

    for split, frames in (("train", fold["train"]), ("val", fold["val"])):
        img_out = base / split / "images"
        lbl_out = base / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for fid in frames:
            reps = 1
            if rfs and split == "train":
                reps = max(1, round(rfs.get(fid, 1.0)))
            lines = gt_to_yolo_lines(gts.get(fid, np.zeros((0, 6))), w, h)
            for r in range(reps):
                tag = fid if r == 0 else f"{fid}__r{r}"
                src = Path(_frame_path(cfg, fid)).resolve()
                dst = img_out / f"{tag}{cfg.data.ext}"
                if not dst.exists():
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        pass
                (lbl_out / f"{tag}.txt").write_text("\n".join(lines))

    return str(write_dataset_yaml(base, "train/images", "val/images"))


def _clips_of(frame_ids, regex):
    clips = defaultdict(list)
    for fid in frame_ids:
        clips[clip_of(fid, regex)].append(fid)
    return {c: sorted(fs) for c, fs in clips.items()}


def _aggregate_per_class(per_class_list):
    agg = {}
    for c in CLASS_IDS:
        vals = [pc[c] for pc in per_class_list if not np.isnan(pc.get(c, np.nan))]
        agg[c] = float(np.mean(vals)) if vals else float("nan")
    return agg


def _fold_predictions(
    cfg, cfg_d, gts, fold, img_wh, fwork: Path, warm_ckpts=None
) -> tuple[dict, str]:
    is_dummy = cfg.model.backend == "dummy"
    yaml = None if is_dummy else prepare_fold_dataset(cfg, gts, fold, img_wh, fwork)
    val_frames = [(fid, _frame_path(cfg, fid)) for fid in fold["val"]]
    seeds = list(cfg.ensemble.seeds) if cfg.ensemble.enabled else [cfg.model.seed]
    seed_preds, primary_ckpt = [], None

    for si, seed in enumerate(seeds):
        scfg = json.loads(json.dumps(cfg_d))
        scfg["model"]["seed"] = int(seed)
        backend = make_backend(scfg, gts)
        warm = (warm_ckpts or {}).get(fold["fold"]) if si == 0 else None
        ckpt = backend.train(yaml, scfg, fwork, warm_start=warm)
        if si == 0:
            primary_ckpt = ckpt
        seed_preds.append(backend.predict(val_frames, scfg, ckpt))

    preds = (
        wbf(seed_preds, iou_thr=float(cfg.ensemble.iou_thr))
        if len(seed_preds) > 1
        else seed_preds[0]
    )
    return preds, primary_ckpt


def run_cv(name, cfg, gts, folds, img_wh, work_dir: Path, warm_ckpts=None) -> dict:
    fold_scores, per_class_list, ckpts = [], [], []
    cfg_d = OmegaConf.to_container(cfg, resolve=True)

    for fold in folds:
        fwork = work_dir / name / f"fold{fold['fold']}"
        fwork.mkdir(parents=True, exist_ok=True)
        preds, primary_ckpt = _fold_predictions(
            cfg, cfg_d, gts, fold, img_wh, fwork, warm_ckpts
        )

        if cfg.tracking.enabled:
            preds = track_all(preds, _clips_of(fold["val"], cfg.cv.clip_regex), cfg_d)

        val_gts = {fid: gts.get(fid, np.zeros((0, 6))) for fid in fold["val"]}
        res = evaluate(preds, val_gts)
        fold_scores.append(res["score"])
        per_class_list.append(res["per_class"])
        ckpts.append((fold["fold"], primary_ckpt))

    scores = np.array(fold_scores, dtype=np.float64)
    scored = scores[~np.isnan(scores)]
    return {
        "name": name,
        "mean": float(scored.mean()) if scored.size else float("nan"),
        "std": float(scored.std()) if scored.size else float("nan"),
        "fold_scores": [float(s) for s in scores],
        "per_class": _aggregate_per_class(per_class_list),
        "ckpts": dict(ckpts),
        "config_hash": config_hash(cfg),
    }


def _log(csv_path: Path, res: dict, sha: str) -> None:
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exp": res["name"],
        "mean": round(res["mean"], 5),
        "std": round(res["std"], 5),
        "folds": json.dumps([round(s, 4) for s in res["fold_scores"]]),
        "per_class": json.dumps({k: round(v, 4) for k, v in res["per_class"].items()}),
        "config_hash": res["config_hash"],
        "git_sha": sha,
    }
    append_result(csv_path, row)
    print(f"  [{res['name']}] mean={res['mean']:.4f} +/- {res['std']:.4f}")


def _load_sweep_inputs(cfg):
    gts = load_gts(
        Path(cfg.data.raw_dir) / cfg.data.csv, cfg.data.id_col, cfg.data.target_col
    )
    frame_ids = list(gts.keys())

    folds_file = Path(cfg.cv.folds_file)
    if folds_file.exists():
        folds = load_folds(folds_file)
    else:
        folds = make_folds(
            frame_ids, int(cfg.cv.k), int(cfg.cv.seed), cfg.cv.clip_regex
        )
        save_folds(folds, folds_file)

    img_wh = image_size(Path(cfg.data.raw_dir) / cfg.data.images) or (640, 384)
    return gts, folds, img_wh


def _maybe_run(name, cfg, gts, folds, img_wh, work_dir, results_csv, sha, done):
    if (name, config_hash(cfg)) in done:
        print(f"  [{name}] skip (already done)")
        return None
    res = run_cv(name, cfg, gts, folds, img_wh, work_dir)
    _log(results_csv, res, sha)
    return res


def _run_baseline(cfg, gts, folds, img_wh, work_dir, results_csv, sha, done):
    print("== baseline ==")
    res = run_cv("baseline", cfg, gts, folds, img_wh, work_dir)
    if ("baseline", res["config_hash"]) not in done:
        _log(results_csv, res, sha)
    else:
        print("  [baseline] skip (already done)")
    return res


def _run_ablations(
    cfg, baseline_res, gts, folds, img_wh, work_dir, results_csv, sha, done
):
    print("== ablations (single-variable, warm-started from baseline) ==")
    results = {}
    warm = baseline_res["ckpts"] if baseline_res else None
    for name, overrides in cfg.sweep.ablations.items():
        ablation_cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        exp_name = f"abl_{name}"
        if (exp_name, config_hash(ablation_cfg)) in done:
            print(f"  [{exp_name}] skip (already done)")
            continue
        res = run_cv(
            exp_name, ablation_cfg, gts, folds, img_wh, work_dir, warm_ckpts=warm
        )
        results[name] = res
        _log(results_csv, res, sha)
    return results


def _check_margin(margin_std: float) -> None:
    if not (np.isfinite(margin_std) and margin_std >= 0):
        raise ValueError(
            f"sweep.win_margin_std must be finite and >= 0, got {margin_std}"
        )


def ablation_wins(
    mean: float, baseline_mean: float, baseline_std: float, margin_std: float
) -> bool:
    """Apply the ablation win rule using baseline standard deviations.

    The gain must be strictly positive, so a tie never wins, even at margin 0.
    A zero baseline standard deviation makes the threshold zero, so any strict
    gain wins. A NaN or infinite input never wins.
    """
    _check_margin(margin_std)
    if not all(np.isfinite([mean, baseline_mean, baseline_std])):
        return False
    gain = mean - baseline_mean
    return bool(gain > 0 and gain >= margin_std * baseline_std)


def merge_overrides(groups: dict[str, list[str]]) -> list[str]:
    """Validate and merge overrides in source order.

    An identical `key=value` repeated by several sources appears once. A key
    that two sources set to different values raises `ValueError`. Each source
    was only validated on its own.
    """
    owner: dict[str, tuple[str, str]] = {}
    merged = []
    for source, overrides in groups.items():
        for item in overrides:
            key, _, value = item.partition("=")
            if key in owner:
                first, first_value = owner[key]
                if first_value != value:
                    raise ValueError(
                        f"overrides conflict on {key!r}: {first} sets "
                        f"{first_value!r}, {source} sets {value!r}"
                    )
                continue
            owner[key] = (source, value)
            merged.append(item)
    return merged


def _winner_overrides(cfg, baseline_res, ablation_results):
    print("== combine winners ==")
    winners = {}
    margin = float(cfg.sweep.win_margin_std)
    _check_margin(margin)
    for name, res in ablation_results.items():
        if ablation_wins(
            res["mean"], baseline_res["mean"], baseline_res["std"], margin
        ):
            gain = res["mean"] - baseline_res["mean"]
            winners[f"ablation {name}"] = list(cfg.sweep.ablations[name])
            print(f"  winner: {name} (+{gain:.4f})")
    if not winners:
        print("  no ablation cleared the margin")
    return merge_overrides(winners)


def run_sweep(config) -> None:
    cfg = OmegaConf.load(config) if isinstance(config, (str, Path)) else config
    sha = git_sha()
    results_csv = Path(cfg.sweep.results_csv)
    work_dir = Path(cfg.sweep.work_dir)
    done = load_done(results_csv)

    gts, folds, img_wh = _load_sweep_inputs(cfg)

    stages = list(cfg.sweep.stages)
    baseline_res = None
    ablation_results = {}

    if "baseline" in stages:
        baseline_res = _run_baseline(
            cfg, gts, folds, img_wh, work_dir, results_csv, sha, done
        )

    if "ablations" in stages:
        ablation_results = _run_ablations(
            cfg, baseline_res, gts, folds, img_wh, work_dir, results_csv, sha, done
        )

    winners = []
    if "combine" in stages and baseline_res:
        winners = _winner_overrides(cfg, baseline_res, ablation_results)
        if winners:
            c = OmegaConf.merge(cfg, OmegaConf.from_dotlist(winners))
            _maybe_run(
                "combined", c, gts, folds, img_wh, work_dir, results_csv, sha, done
            )

    if "final" in stages and baseline_res:
        print("== final (winners + TTA + tracking + ensemble) ==")
        final_over = merge_overrides(
            {
                "the winners": winners,
                "the final stage": [
                    "tta.enabled=true",
                    "tracking.enabled=true",
                    "ensemble.enabled=true",
                ],
            }
        )
        c = OmegaConf.merge(cfg, OmegaConf.from_dotlist(final_over))
        _maybe_run("final", c, gts, folds, img_wh, work_dir, results_csv, sha, done)

    print(f"\ndone -> {results_csv}")


if __name__ == "__main__":
    cfg_path = (
        sys.argv[sys.argv.index("--config") + 1]
        if "--config" in sys.argv
        else "configs/base.yaml"
    )
    run_sweep(cfg_path)
