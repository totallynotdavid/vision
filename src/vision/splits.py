"""Clip-grouped K-fold cross-validation prevents frame-level leakage."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_CLIP_REGEX = r"^(.*)_\d+$"


def clip_of(frame_id: str, regex: str = DEFAULT_CLIP_REGEX) -> str:
    """The regex must map all frames from one clip to one group key."""
    m = re.match(regex, str(frame_id))
    return m.group(1) if m else str(frame_id)


def make_folds(
    frame_ids: list[str], k: int = 5, seed: int = 0, regex: str = DEFAULT_CLIP_REGEX
) -> list[dict]:
    """Balance frame counts without splitting any clip across folds."""
    by_clip: dict[str, list[str]] = defaultdict(list)
    for fid in frame_ids:
        by_clip[clip_of(fid, regex)].append(fid)

    clips = sorted(by_clip.keys(), key=lambda c: (-len(by_clip[c]), c))
    import random

    rng = random.Random(seed)
    # The seeded tie-break avoids depending on lexicographic order for equal sizes.
    clips.sort(key=lambda c: (-len(by_clip[c]), rng.random()))

    fold_clips: list[list[str]] = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for c in clips:
        i = min(range(k), key=lambda j: fold_sizes[j])
        fold_clips[i].append(c)
        fold_sizes[i] += len(by_clip[c])

    folds = []
    for i in range(k):
        val_clips = set(fold_clips[i])
        val = [f for c in fold_clips[i] for f in by_clip[c]]
        train = [f for c, fs in by_clip.items() if c not in val_clips for f in fs]
        folds.append(
            {
                "fold": i,
                "val_clips": sorted(val_clips),
                "train": sorted(train),
                "val": sorted(val),
            }
        )
    return folds


def save_folds(folds: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(folds, indent=2))
    return path


def load_folds(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
