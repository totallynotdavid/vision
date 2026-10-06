# Agent rules

- Run `mise run check` before finishing. It must pass.
- Pass an `AngleConvention` from `src/vision/geometry.py` wherever a box angle
  is converted. Do not pick an angle sign locally.
- Class ids are 1 to 9 in IR and competition text; YOLO labels use 0 to 8.
- Import `ultralytics`, `sahi` and `torch` inside functions, never at module top
  level.
- Do not commit `data/`, `runs/`, checkpoints or `results.csv`.
- The module map is `architecture.md`. Update it when you add or remove a
  module.
