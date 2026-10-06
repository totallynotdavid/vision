# Architecture

The package is `src/vision/`. Modules exchange boxes as the
[IR](docs/formats.md#internal-ir), a mapping from frame id to a NumPy array.
Configuration is OmegaConf YAML in [`configs/`](configs/).

## Module map

| Module                                  | Responsibility                                                                                          |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`__init__.py`](src/vision/__init__.py) | Class ids and names, `NUM_CLASSES`, and the rotated-IoU thresholds.                                     |
| [`geometry.py`](src/vision/geometry.py) | `AngleConvention`, box to corners and back, rotated IoU, box validity. The angle convention lives here. |
| [`convert.py`](src/vision/convert.py)   | Competition text cells, IR, and YOLO-OBB labels; `dataset.yaml`; submission CSV.                        |
| [`metric.py`](src/vision/metric.py)     | Macro AP-rIoU.                                                                                          |
| [`splits.py`](src/vision/splits.py)     | Clip key from a frame id, clip-grouped K-fold, folds file read and write.                               |
| [`data.py`](src/vision/data.py)         | Class counts, loss weights, repeat factors, angle statistics, image size, tile suggestion, `inspect`.   |
| [`backends.py`](src/vision/backends.py) | `Backend` interface; `DummyOBB` on CPU; `UltralyticsOBB` for YOLO26-OBB.                                |
| [`infer.py`](src/vision/infer.py)       | Ultralytics inference, plain and SAHI-tiled, converted to IR.                                           |
| [`ensemble.py`](src/vision/ensemble.py) | Rotated weighted box fusion across prediction sets.                                                     |
| [`track.py`](src/vision/track.py)       | Per-clip tracking of predictions across frames.                                                         |
| [`sweep.py`](src/vision/sweep.py)       | Stage orchestration, per-fold training and scoring, `results.csv`, skipping runs it already holds.      |
| [`synth.py`](src/vision/synth.py)       | Synthetic clips with ground truth, for runs without data.                                               |
| [`viz.py`](src/vision/viz.py)           | Box overlays and the calibration renderer.                                                              |
| [`labeling.py`](src/vision/labeling.py) | X-AnyLabeling JSON export and import, review ordering.                                                  |

`python -m vision.<module>` runs `sweep`, `synth`, `data` (`inspect`), and `viz`.
The remaining modules provide library APIs.

## Path of a sweep

For each fold and run, `sweep.py` follows these steps:

```text
train.csv ──load_gts──▶ IR ground truth
                          │
splits.make_folds ────────┤  folds (clip-grouped)
                          ▼
   backend.train ──▶ checkpoint ──▶ backend.predict ──▶ IR predictions
    (YOLO dataset from                 (infer.py: plain or SAHI)
    convert and RFS repeats
    from data)
                          │  per seed; ensemble.wbf fuses seeds
                          ▼
              track.track_all (if tracking is on)
                          ▼
                  metric.evaluate ──▶ fold score ──▶ results.csv row
```

`sweep.py` imports `backends`, `convert`, `data`, `ensemble`, `metric`, `splits`,
and `track`. `infer.py` and the Ultralytics and SAHI imports load only when a
real backend runs. The dummy backend and tests therefore need only the base
dependencies.

## Boundaries

- Box-to-corner conversion and rotated IoU live in `geometry.py`. Other modules
  call it with an `AngleConvention`.
- Class ids are 1 to 9 everywhere except YOLO labels, which are 0 to 8.
  `convert.py` and `infer.py` shift them.
- A backend returns IR predictions and nothing else, so the sweep treats the
  dummy and Ultralytics backends the same.
