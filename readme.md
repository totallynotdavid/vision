# vision

Oriented vehicle detection for MTC Peru intersections. The task is to find
vehicles in traffic-camera frames as rotated boxes in nine classes (auto, combi,
microbus, minibus, omnibus, articulado, camion, mototaxi, motocicleta). The
official metric is Macro AP-rIoU@[0.50:0.80].

This repository is the experiment harness for that task. It reproduces the
metric locally, splits frames into clip-grouped cross-validation folds, and
scores one training or inference change at a time. It has no command that
predicts a test set.

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Try it without data or a GPU

Generate a synthetic dataset and sweep it with the CPU-only dummy backend:

```console
$ uv run python -m vision.synth data/synth
synthetic dataset -> data/synth/train.csv (640x384)
$ uv run python -m vision.sweep --config configs/smoke.yaml
== baseline ==
  [baseline] mean=0.7732 +/- 0.0624
== ablations (single-variable, warm-started from baseline) ==
  [abl_sahi] mean=0.7867 +/- 0.0507
  [abl_tracking] mean=0.7732 +/- 0.0624
  [abl_size_s] mean=0.7732 +/- 0.0624
== combine winners ==
  winner: sahi (+0.0135)
  [combined] mean=0.7867 +/- 0.0507
== final (winners + TTA + tracking + ensemble) ==
  [final] mean=0.9373 +/- 0.0294

done -> runs/smoke/results.csv
```

Each line is a cross-validated Macro AP-rIoU, mean and standard deviation over
folds. [`docs/sweep.md`](docs/sweep.md) explains the stages and the
`results.csv` columns.

## Features

- **Local metric.** Macro AP-rIoU@[0.50:0.80] over 9 classes and 7 rotated-IoU
  thresholds ([`docs/metric.md`](docs/metric.md)).
- **Clip-grouped folds.** Frames from one clip never straddle training and
  validation folds.
- **Sweep.** Baseline, single-variable ablations, a combination of the winners,
  and a final run with test-time levers. Each run appends a row to `results.csv`
  unless the same experiment and config are already in it.
- **Long-tail and small-object levers.** Repeat-factor sampling, mixup,
  copy-paste, SAHI tiled inference, test-time augmentation, tracking, and a
  multi-seed rotated box-fusion ensemble.
- **Two backends.** YOLO26-OBB through Ultralytics for real runs; a
  deterministic dummy backend for CPU runs.
- **Format converters.** Competition text cells, YOLO-OBB labels, and
  X-AnyLabeling JSON ([`docs/formats.md`](docs/formats.md),
  [`docs/labeling.md`](docs/labeling.md)).
- **Data inspection and angle calibration.** Class counts, suggested settings,
  and box overlays for checking the angle convention
  ([`docs/real-data.md`](docs/real-data.md)).

## Documentation

The manual starts at [`docs/readme.md`](docs/readme.md). The module map is in
[`architecture.md`](architecture.md). To work on the code, read
[`contributing.md`](contributing.md).
