# Formats

Boxes appear in three representations. The converters are in
[`src/vision/convert.py`](../src/vision/convert.py).

## Classes

Competition class ids run 1 to 9
([`src/vision/__init__.py`](../src/vision/__init__.py)). YOLO class indices run
0 to 8, in the same order.

| Id  | Name     | Id  | Name       | Id  | Name        |
| --- | -------- | --- | ---------- | --- | ----------- |
| 1   | auto     | 4   | minibus    | 7   | camion      |
| 2   | combi    | 5   | omnibus    | 8   | mototaxi    |
| 3   | microbus | 6   | articulado | 9   | motocicleta |

## Competition text cells

One cell per frame, boxes separated by `;`, fields by spaces. Angles are in
degrees.

```text
ground truth   cls cx cy w h angle;cls cx cy w h angle
prediction     score cls cx cy w h angle;score cls cx cy w h angle
```

An empty cell or `none` means no boxes. `format_pred_cell` writes `none` for a
frame with no valid box and drops boxes that are not finite, have a non-positive
size, or have a class outside 1 to 9.
`build_submission(preds, frame_order, out_path)` writes a CSV with columns `Id`
and `Target` and one row per requested frame. No command calls it.

## Internal IR

A mapping from `frame_id` to a float64 `np.ndarray`. Ground-truth rows are
`[cls, cx, cy, w, h, angle]`, shape `(n, 6)`. Prediction rows are
`[score, cls, cx, cy, w, h, angle]`, shape `(n, 7)`. `load_gts` and `load_preds`
read a CSV into this form.

## YOLO-OBB labels

One line per box: `<class index> x1 y1 x2 y2 x3 y3 x4 y4`. The four corners are
normalized by image width and height, clipped to 0 to 1, and written with six
decimals. `gt_to_yolo_lines` skips invalid boxes. `write_dataset_yaml` writes
the Ultralytics `dataset.yaml` with class names ordered by index.

## Angle convention

[`src/vision/geometry.py`](../src/vision/geometry.py) owns the box geometry.
Angle 0 puts the width on the image x-axis and the height on the y-axis. The
angle is in degrees. `AngleConvention.ccw` sets the rotation direction; the
default is counter-clockwise. Corner conversion, rotated IoU, label writing,
overlays and the labeling converters all take an `AngleConvention` and default
to `DEFAULT_CONVENTION`.

Inference does not take one. `_yolo_to_ir` in
[`src/vision/infer.py`](../src/vision/infer.py) reads Ultralytics angles
counter-clockwise whatever `ccw` is, and the SAHI path takes the angle from
OpenCV's `minAreaRect` unchanged.

Rotated IoU uses Shapely polygons. A box that is not a valid polygon is repaired
with `buffer(0)`, and a box with zero area has IoU 0.

To check the convention against real annotations, see
[Real data](real-data.md#calibrate-the-angle-convention).
