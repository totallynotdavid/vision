# Sweep

```bash
uv run python -m vision.sweep --config configs/smoke.yaml
```

`--config` is the only option; it defaults to `configs/base.yaml`.
[`configs/smoke.yaml`](../configs/smoke.yaml) uses the CPU dummy backend on
synthetic data. [`configs/base.yaml`](../configs/base.yaml) trains YOLO26-OBB
through Ultralytics and needs the `train` extra and a GPU:

```bash
uv sync --extra train
uv run python -m vision.sweep --config configs/base.yaml
```

The code is [`src/vision/sweep.py`](../src/vision/sweep.py).

## Stages

`sweep.stages` lists the stages to run, in this order:

1. `baseline` cross-validates the config as written.
2. `ablations` runs each entry of `sweep.ablations` as the baseline plus one
   dotlist override, for example `sahi: ["sahi.enabled=true"]`. Ablations
   warm-start from the baseline checkpoint of the same fold.
3. `combine` merges the overrides of every ablation that wins and runs them
   together as `combined`. An ablation wins when its mean beats the baseline
   mean by at least `sweep.win_margin_std` baseline standard deviations. If none
   wins, nothing runs.

   - The gain must be strictly positive. A tie never wins, even at margin 0.
   - A gain exactly equal to the margin wins.
   - When the baseline standard deviation is 0, the threshold is 0, so any
     strictly positive gain wins.
   - A NaN or infinite mean or standard deviation never wins. A fold with no
     ground truth scores NaN and is left out of the mean and standard
     deviation.
   - Two winners that set the same key to different text stop the sweep with an
     error naming the key and both sources. For example, `model.size=s` and
     `model.size=m` conflict because neither was tested with the other's value.
     The same `key=value` from two winners is applied once. `final` applies the
     same check between the winners and its own `tta`, `tracking` and `ensemble`
     overrides. Values compare as written, so `1` and `1.0` conflict.
   - `win_margin_std` must be finite and at least 0, or the sweep stops with an
     error.
4. `final` runs the winners plus `tta.enabled=true`, `tracking.enabled=true` and
   `ensemble.enabled=true` as `final`.

`combine` and `final` need `baseline`. `combine` compares only the ablations
that ran in the current invocation. An ablation skipped because
[`results.csv`](#resultscsv) already holds it is not a winner, so a re-run can
combine fewer overrides than the first.

Each run is `cv.k` folds. A fold trains on its train frames, predicts its
validation frames, and is scored with [the local metric](metric.md). The run
reports the mean and standard deviation of the fold scores.

## `results.csv`

Each run that is not skipped (see below) appends one row to the file named by
`sweep.results_csv`. The smoke config writes `runs/smoke/results.csv`; the base
config writes `results.csv` in the working directory. Both paths are
git-ignored.

```text
timestamp,exp,mean,std,folds,per_class,config_hash,git_sha
2026-10-06 12:24:55,baseline,0.77315,0.06244,"[0.8276, 0.6857, 0.8061]","{""1"": 0.6946, ""2"": 0.8287, ""3"": 0.7112, ""4"": 0.75, ""5"": 0.75, ""6"": 0.8958, ""7"": 0.7812, ""8"": 0.9056, ""9"": 0.6036}",9525277799,9358520
```

| Column        | Content                                                    |
| ------------- | ---------------------------------------------------------- |
| `exp`         | `baseline`, `abl_<name>`, `combined`, or `final`.          |
| `mean`, `std` | Mean and standard deviation of the per-fold scores.        |
| `folds`       | JSON list of per-fold scores.                              |
| `per_class`   | JSON map from class id to AP, averaged over folds.         |
| `config_hash` | First 10 hex digits of the SHA-1 of the resolved config.   |
| `git_sha`     | Short `HEAD` at run time, or `nogit` outside a repository. |

A run whose `(exp, config_hash)` is already in the file is skipped and appends
no row. The baseline is the exception: it is recomputed on every invocation,
because ablations need its checkpoints, and is logged only when its pair is new.

Running a sweep again therefore skips the completed runs, but `combine` and
`final` see only the ablations that ran in that invocation. Their configs can
differ from the earlier pass, and a different config has a different
`config_hash`, so the run appends a new row. Delete the file to score every run
again.

## Folds

The first run writes the folds to `cv.folds_file` and later runs reuse them.
Delete the file to regenerate it after the data or `cv.k` changes.
`cv.clip_regex` maps a frame id to its clip: group 1 is the clip key, and an id
that does not match is its own clip. The default `^(.*)_\d+$` turns
`clip003_007` into `clip003`.

## Levers

Ablations override these keys. Defaults are in
[`configs/base.yaml`](../configs/base.yaml).

| Key                                      | Effect                                                                                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `model.size`                             | YOLO26-OBB size; the checkpoint is `yolo26<size>-obb.pt`.                                                                               |
| `model.epochs`, `imgsz`, `batch`, `seed` | Passed to Ultralytics training.                                                                                                         |
| `model.conf`                             | Confidence floor at inference. It is low so AP sees every ranked box.                                                                   |
| `model.cls_gain`, `patience`             | Ultralytics classification loss gain and early-stopping patience.                                                                       |
| `aug.mosaic`, `mixup`, `copy_paste`      | Ultralytics augmentation probabilities.                                                                                                 |
| `sampling.rfs`, `rfs_thr`                | Repeat-factor sampling: a training image repeats `round(f)` times, where `f` is the largest repeat factor of its classes.               |
| `sahi.enabled`, `tile`, `overlap`        | Tiled inference with SAHI.                                                                                                              |
| `tta.enabled`                            | Ultralytics test-time augmentation. SAHI inference does not use it.                                                                     |
| `tracking.enabled`                       | Per-clip tracking of detections across consecutive frames, with Hungarian matching on rotated IoU.                                      |
| `ensemble.enabled`, `seeds`, `iou_thr`   | Train one model per seed and fuse same-class boxes whose rotated IoU reaches `iou_thr` with [`ensemble.py`](../src/vision/ensemble.py). |

The dummy backend reads `model.seed`, `model.dummy_noise`, `model.dummy_recall`,
and `sahi.enabled`. It ignores the other model keys, so `abl_size_s` scores the
same as the baseline in the smoke config.

## Backends

`model.backend` selects the backend: `ultralytics` or `dummy`
([`backends.py`](../src/vision/backends.py)). A backend implements
`train(dataset_yaml, cfg, work_dir, warm_start)`, which returns a checkpoint id,
and `predict(frames, cfg, ckpt)`, which returns predictions as
[IR](formats.md#internal-ir).

For `ultralytics`, each fold gets a YOLO dataset under
`<work_dir>/<exp>/fold<n>/fold<n>_data/` with symlinked images and normalized
labels. The checkpoint is `<work_dir>/<exp>/fold<n>/train/weights/best.pt`.
