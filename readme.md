# Oriented vehicle detection

Detect and classify 9 vehicle classes with oriented bounding boxes.
Official metric: **Macro AP-rIoU@[0.50:0.80]**. Average precision is averaged
over 9 classes (equal weight) and 7 rotated-IoU thresholds (0.50 to 0.80, step
0.05).

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/). `mise` pins the
toolchain (`mise.toml`); otherwise `uv` fetches the interpreter on first sync.

## Quickstart (no data, no GPU)

The quickstart uses synthetic data and a CPU-only dummy backend. It validates
the pipeline before real data or a GPU is available.

```bash
uv sync --extra dev
uv run pytest
uv run python -m vision.synth data/synth
uv run python -m vision.sweep --config configs/smoke.yaml
```

## Set up real data

The training extra installs the model backend and tiled-inference dependencies.
The training backend pulls torch and expects a GPU.

1. Drop the data in `data/raw/` (frames plus `train.csv`).
2. `uv run python -m vision.data inspect data/raw`. This derives constants from
   image size and class counts (image size, tile size, loss weights).
3. `uv run python -m vision.viz data/raw`. Review the overlays in
   `runs/calibration/` and **confirm the angle convention before training**. If
   boxes rotate the wrong way, flip `ccw` in `vision.geometry`.
4. Point `configs/base.yaml:data.raw_dir` at the data, set `cv.clip_regex` to the
   real frame-id format, then start the real-data sweep:

   ```bash
   uv sync --extra train
   uv run python -m vision.sweep --config configs/base.yaml
   ```

   Read `results.csv` when it returns. See `configs/base.yaml` for the full lever list.

## Sweep stages

The sweep runs `baseline`, then single-variable `ablations`, then `combine`,
then `final`. Ablations warm-start from the baseline. `combine` keeps only the
ablations that beat the baseline by the configured std margin. `final` layers on
the late inference levers: TTA, tracking, and rotated-WBF ensemble.

Every run appends one row to `results.csv` (per-fold scores, mean/std, per-class
AP, git SHA). Completed `(name, config_hash)` pairs are skipped, so an
interrupted sweep resumes where it left off.

## Box representations

The code moves between three box representations:

1. Competition text cells: `score cls cx cy w h angle;...` (predictions) or
   `cls cx cy w h angle;...` (ground truth).
2. Internal IR: a mapping from `frame_id` to an `np.ndarray` of boxes.
3. YOLO-OBB labels: normalized corner coordinates with zero-based class
   indices.

Competition class ids are `1..9`; YOLO class indices are `0..8`.

## Labeling loop

`vision.labeling` provides X-AnyLabeling converters only (not a live tool
integration). The intended loop: pseudo-label external images, review the
low-confidence frames in X-AnyLabeling, then import the corrected JSON back into
the project IR.
