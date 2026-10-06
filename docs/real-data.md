# Real data

The commands here read `train.csv` and a folder of frames. The examples run on
the synthetic dataset from `uv run python -m vision.synth data/synth`; replace
`data/synth` with `data/raw` for real data.

`vision.data inspect` and `vision.viz` take only the data directory. They read
`<dir>/train.csv` with columns `Id` and `Target` and the frames in
`<dir>/images/`, whatever the `data.*` keys say. The sweep reads those keys.

## Layout

```text
data/raw/
  train.csv        columns Id, Target
  images/          one <Id>.jpg per row
```

Each `Target` cell is a ground-truth list in the
[competition text format](formats.md#competition-text-cells). For the sweep, the
file name, column names, image folder and extension are the `data.*` keys of
[`configs/base.yaml`](../configs/base.yaml). `data/raw/`, `data/yolo/`,
`data/external/` and `data/synth/` are git-ignored.

## Inspect

```console
$ uv run python -m vision.data inspect data/synth
frames: 32   image_size: (640, 384)   suggested SAHI tile: None
angle: {'min': 5.1, 'max': 178.76, 'mean': 101.48541666666667, 'n': 192}
class                count     loss_w
  1 auto                24   0.677
  2 combi               40   0.409
  3 microbus            24   0.677
  4 minibus              8   2.014
  5 omnibus              8   2.014
  6 articulado          32   0.510
  7 camion              24   0.677
  8 mototaxi            16   1.011
  9 motocicleta         16   1.011
```

The command prints values; it changes no file. Copy what you need into the
config by hand:

- `image_size` guides `model.imgsz`.
- The suggested SAHI tile guides `sahi.tile`. It is `None` when the longest side
  is 1280 or less, 640 up to 2048, and 1024 above that.
- `loss_w` are class-balanced weights from the effective number of samples (beta
  0.999), normalized to mean 1. No config key reads them.

The code is [`src/vision/data.py`](../src/vision/data.py).

## Calibrate the angle convention

```console
$ uv run python -m vision.viz data/synth
calibration overlays -> runs/calibration  (inspect these before training!)
```

The command draws the ground-truth boxes of the first 12 frames onto copies of
the images in `runs/calibration/`. Open them and check that every box lies on
its vehicle. If the boxes are rotated the wrong way, flip the `ccw` default of
`AngleConvention` in [`src/vision/geometry.py`](../src/vision/geometry.py). That
setting is the default for corner conversion, rotated IoU, YOLO labels, overlays
and the labeling converters. It does not reach the angles that the Ultralytics
backend predicts: inference reads them with a fixed sign, so the metric and the
labels follow the flip and the predictions do not. See
[the angle convention](formats.md#angle-convention).

## Configure and run

1. Point `data.raw_dir` in `configs/base.yaml` at the data.
2. Set `cv.clip_regex` to the real frame-id format so that all frames of a clip
   share group 1. See [Folds](sweep.md#folds).
3. Run the [sweep](sweep.md) with `configs/base.yaml`.
