"""Detection backends behind the sweep interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Backend:
    """Backends train a checkpoint and return project IR predictions."""

    def train(
        self,
        dataset_yaml: str,
        cfg: dict,
        work_dir: Path,
        warm_start: str | None = None,
    ) -> str:
        raise NotImplementedError

    def predict(self, frames: list[tuple[str, str]], cfg: dict, ckpt: str) -> dict:
        raise NotImplementedError


class DummyOBB(Backend):
    """Deterministic CPU backend for exercising sweep selection logic."""

    def __init__(self, gts: dict[str, np.ndarray]):
        self.gts = gts

    def train(self, dataset_yaml, cfg, work_dir, warm_start=None):
        return f"dummy::{cfg.get('model', {}).get('size', 'n')}"

    def predict(self, frames, cfg, ckpt):
        mcfg = cfg.get("model", {})
        seed = int(mcfg.get("seed", 0)) + (
            1 if cfg.get("sahi", {}).get("enabled") else 0
        )
        rng = np.random.default_rng(seed)
        noise = float(mcfg.get("dummy_noise", 3.0))
        recall = float(mcfg.get("dummy_recall", 0.8))
        out = {}
        for fid, _path in frames:
            gt = np.asarray(self.gts.get(fid, np.zeros((0, 6)))).reshape(-1, 6)
            rows = []
            for cls, cx, cy, w, h, ang in gt:
                if rng.random() > recall:
                    continue
                rows.append(
                    [
                        rng.uniform(0.5, 1.0),
                        cls,
                        cx + rng.normal() * noise,
                        cy + rng.normal() * noise,
                        max(1.0, w + rng.normal() * noise),
                        max(1.0, h + rng.normal() * noise),
                        ang + rng.normal() * noise,
                    ]
                )
            out[fid] = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
        return out


class UltralyticsOBB(Backend):
    """Real YOLO26-OBB. Heavy deps imported lazily so importing this module is cheap."""

    def train(self, dataset_yaml, cfg, work_dir, warm_start=None):
        from ultralytics import YOLO

        mcfg = cfg.get("model", {})
        size = mcfg.get("size", "n")
        weights = warm_start or f"yolo26{size}-obb.pt"
        model = YOLO(weights)
        aug = cfg.get("aug", {})
        results = model.train(
            data=str(dataset_yaml),
            epochs=int(mcfg.get("epochs", 100)),
            imgsz=int(mcfg.get("imgsz", 640)),
            batch=int(mcfg.get("batch", 16)),
            project=str(work_dir),
            name="train",
            seed=int(mcfg.get("seed", 0)),
            mosaic=float(aug.get("mosaic", 1.0)),
            mixup=float(aug.get("mixup", 0.0)),
            copy_paste=float(aug.get("copy_paste", 0.0)),
            cls=float(mcfg.get("cls_gain", 0.5)),
            patience=int(mcfg.get("patience", 30)),
            exist_ok=True,
            verbose=False,
        )
        return str(Path(results.save_dir) / "weights" / "best.pt")

    def predict(self, frames, cfg, ckpt):
        from .infer import predict_plain, predict_sahi

        if cfg.get("sahi", {}).get("enabled"):
            return predict_sahi(ckpt, frames, cfg)
        return predict_plain(ckpt, frames, cfg)


def make_backend(cfg: dict, gts: dict | None = None) -> Backend:
    name = cfg.get("model", {}).get("backend", "ultralytics")
    if name == "dummy":
        assert gts is not None, "DummyOBB needs gts"
        return DummyOBB(gts)
    if name == "ultralytics":
        return UltralyticsOBB()
    raise ValueError(f"unknown backend: {name}")
