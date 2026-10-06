# Metric

[`src/vision/metric.py`](../src/vision/metric.py) reproduces Macro
AP-rIoU@[0.50:0.80] locally.

```python
from vision.metric import evaluate

res = evaluate(preds, gts)
res["score"]  # Macro AP-rIoU
res["per_class"]  # {class id: AP averaged over thresholds}
res["per_class_threshold"]  # {class id: {threshold: AP}}
```

`preds` and `gts` are [IR](formats.md#internal-ir) mappings from frame id to
arrays.

## Computation

1. Boxes match within one frame and one class. A prediction of another class is
   a false positive and never matches.
2. For each class and each threshold in `RIOU_THRESHOLDS` (0.50 to 0.80 in steps
   of 0.05, in [`src/vision/__init__.py`](../src/vision/__init__.py)), the
   predictions of all frames are sorted by descending score.
3. Each prediction takes the unmatched ground-truth box of its frame with the
   highest rotated IoU. It is a true positive when that IoU reaches the
   threshold; otherwise it is a false positive. A ground-truth box matches once,
   so a duplicate prediction is a false positive.
4. AP is the area under the precision-recall curve, with the precision envelope
   made monotone, as in continuous COCO AP.
5. A class's AP is the mean over thresholds. The score is the mean over classes.

A class with no ground-truth box has no AP and is left out of the mean. A class
with ground truth and no predictions has AP 0.

Rotated IoU and its angle convention are described in
[Formats](formats.md#angle-convention).
