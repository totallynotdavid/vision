"""Real-backend inference converts model output to IR in the given convention."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from vision.geometry import AngleConvention, obb_to_corners
from vision.infer import predict_plain, predict_sahi

CCW = AngleConvention(ccw=True)
CW = AngleConvention(ccw=False)
FRAMES = [("clip_0001", "clip_0001.png")]


class _Tensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _FakeObb:
    """One box at 30 degrees counter-clockwise in class 3 (index 2)."""

    xywhr = _Tensor([[50.0, 40.0, 20.0, 10.0, np.deg2rad(30.0)]])
    conf = _Tensor([0.9])
    cls = _Tensor([2])

    def __len__(self) -> int:
        return 1


def _install_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    class YOLO:
        def __init__(self, ckpt: str) -> None:
            self.ckpt = ckpt

        def predict(self, paths: list[str], **kwargs: Any) -> list[Any]:
            return [types.SimpleNamespace(obb=_FakeObb()) for _ in paths]

    module = types.ModuleType("ultralytics")
    module.__dict__["YOLO"] = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def _install_sahi(monkeypatch: pytest.MonkeyPatch) -> None:
    corners = obb_to_corners(50.0, 40.0, 20.0, 10.0, 30.0, CCW)
    prediction = types.SimpleNamespace(
        mask=types.SimpleNamespace(segmentation=[corners.ravel().tolist()]),
        score=types.SimpleNamespace(value=0.8),
        category=types.SimpleNamespace(id=2),
    )

    class AutoDetectionModel:
        @staticmethod
        def from_pretrained(**kwargs: Any) -> object:
            return object()

    def get_sliced_prediction(*args: Any, **kwargs: Any) -> Any:
        return types.SimpleNamespace(object_prediction_list=[prediction])

    sahi = types.ModuleType("sahi")
    sahi.__dict__["AutoDetectionModel"] = AutoDetectionModel
    predict = types.ModuleType("sahi.predict")
    predict.__dict__["get_sliced_prediction"] = get_sliced_prediction
    monkeypatch.setitem(sys.modules, "sahi", sahi)
    monkeypatch.setitem(sys.modules, "sahi.predict", predict)


def test_plain_prediction_is_ir_in_the_default_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ultralytics(monkeypatch)
    row = predict_plain("best.pt", FRAMES, {})["clip_0001"][0]
    assert row == pytest.approx([0.9, 3, 50.0, 40.0, 20.0, 10.0, 30.0])


def test_plain_prediction_flips_the_angle_for_a_clockwise_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ultralytics(monkeypatch)
    ccw = predict_plain("best.pt", FRAMES, {}, CCW)["clip_0001"][0]
    cw = predict_plain("best.pt", FRAMES, {}, CW)["clip_0001"][0]
    assert cw[6] == pytest.approx(-ccw[6])
    assert cw[:6] == pytest.approx(ccw[:6])


def test_sahi_prediction_flips_the_angle_for_a_clockwise_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sahi(monkeypatch)
    ccw = predict_sahi("best.pt", FRAMES, {}, CCW)["clip_0001"][0]
    cw = predict_sahi("best.pt", FRAMES, {}, CW)["clip_0001"][0]
    assert ccw[6] != 0.0
    assert cw[6] == pytest.approx(-ccw[6])
    assert cw[:6] == pytest.approx(ccw[:6])
