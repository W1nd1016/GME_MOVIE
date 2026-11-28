"""Access converted MovieLens interactions and item-only metadata."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import (
    FEATURE_WIDTH, HOLDOUT_VIEWS, MOVIE_INDEX, NEIGHBOR_COUNT, SLOTS, VOCAB_SIZE,
    Purpose, Split, require_access,
)
from .splits import validate_holdout


def _read_arrays(path, fields):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(fields):
            raise ValueError(f"{path}: expected fields {sorted(fields)}, got {archive.files}")
        return {name: archive[name] for name in fields}


def _validate_ids(name, values):
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer IDs")
    if values.size and (values.min() < 0 or values.max() >= VOCAB_SIZE):
        raise ValueError(f"{name} IDs must be in [0, {VOCAB_SIZE})")


def _load_samples(root, split):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["splits"][split.value]["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("manifest row count must be a nonnegative integer")
    arrays = _read_arrays(root / f"{split.value}.npz", ("features", "label"))
    features, labels = arrays["features"], arrays["label"]
    if features.shape != (rows, FEATURE_WIDTH) or labels.shape != (rows,):
        raise ValueError("sample shapes disagree with manifest row count or feature schema")
    _validate_ids("features", features)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("expected binary labels")
    return features, labels


class MovieLensDataset(Dataset):
    """One split, opened for an explicit purpose. Each row is (features, label)."""

    def __init__(self, root, split, purpose, *, holdout_path=None):
        self.split = Split(split)
        purpose = Purpose(purpose)
        require_access(self.split, purpose)
        root = Path(root)
        source, group = HOLDOUT_VIEWS.get((self.split, purpose), (self.split, None))
        self.holdout_bytes = None
        if group is not None:
            if holdout_path is None:
                raise ValueError(f"{self.split.value} requires holdout_path for {purpose.value}")
            self.holdout_bytes = Path(holdout_path).read_bytes()
        elif holdout_path is not None:
            raise ValueError(f"{self.split.value} does not use a holdout for {purpose.value}")

        features, labels = _load_samples(root, source)
        if group is not None:
            movies = features[:, MOVIE_INDEX]
            holdout = validate_holdout(self.holdout_bytes, movies, load_items(root)["movie_id"])
            keep = np.isin(movies, holdout[group])
            features, labels = features[keep], labels[keep]
        self.features = torch.from_numpy(features.astype(np.int64, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.float32, copy=False))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def load_items(root):
    """Return aligned item arrays, with one row per unique movie ID."""
    fields = ("movie_id", "year", "title", "genres", "neighbors")
    items = _read_arrays(Path(root) / "items.npz", fields)
    ids = items["movie_id"]
    if ids.ndim != 1 or len(np.unique(ids)) != ids.size or np.any(ids == 0):
        raise ValueError("expected one row per non-padding movie ID")
    rows = len(ids)
    widths = dict(SLOTS)
    shapes = {
        "movie_id": (rows,), "year": (rows,),
        "title": (rows, widths["title"]), "genres": (rows, widths["genres"]),
        "neighbors": (rows, NEIGHBOR_COUNT),
    }
    for name, values in items.items():
        _validate_ids(name, values)
        if values.shape != shapes[name]:
            raise ValueError(f"{name}: expected shape {shapes[name]}, got {values.shape}")
    if not np.isin(items["neighbors"], ids).all():
        raise ValueError("neighbors must refer to movies in the item table")
    return items
