# Labeling

[`src/vision/labeling.py`](../src/vision/labeling.py) converts between
[IR](formats.md#internal-ir) and X-AnyLabeling JSON files. It only reads and
writes files. It does not drive the tool, and no command calls it. Use it from
Python.

The loop:

1. Run a model to get predictions on unlabeled images.
2. `rank_by_uncertainty(preds)` returns frame ids in review order: frames with
   no predictions first, then ascending by their highest score, then by box
   count.
3. `export_pseudolabels(preds, images_dir, out_dir, conf_thr=0.25)` writes one
   `<frame id>.json` per frame to `out_dir`. It keeps predictions scored at
   `conf_thr` or higher. Each is a `rotation` shape with the class name as its
   label, four corner points, a `direction` in radians, and the `score`.
   `imagePath` is `<frame id>.jpg`.
4. Open `out_dir` in X-AnyLabeling and correct the boxes.
5. `import_corrections(json_dir)` reads the corrected files back into a
   ground-truth IR mapping, with the box recovered from the corner points. A
   shape whose label is not one of the nine [class names](formats.md#classes) is
   skipped.
