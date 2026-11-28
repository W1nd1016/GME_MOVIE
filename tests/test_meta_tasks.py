"""Contracts for movie-level support/query tasks and visible positive feedback."""

import json

import numpy as np
import pytest
import torch

from coldstart.data.meta_tasks import load_meta_tasks, make_meta_split
from coldstart.data.schema import FEATURE_WIDTH, MOVIE_INDEX, USER_INDEX
from coldstart.data.splits import make_holdout
from coldstart.data.visible_hist import positive_users


def make_meta_source(root, old_count=20, split_seed=42):
    """Write movie metadata, fixed holdout and shuffled A/B with 20 rows per movie."""
    root.mkdir(parents=True, exist_ok=True)
    old = np.arange(100, 100 + old_count, dtype=np.int32)
    items = np.arange(100, 101 + old_count, dtype=np.int32)[::-1]
    holdout = make_holdout(old, items, split_seed)
    path = root / "holdout.json"
    path.write_text(json.dumps(holdout))
    title = np.tile([10, 11, 0, 0, 0], (len(items), 1))
    genres = np.tile([20, 21, 22, 0, 0], (len(items), 1))
    neighbors = np.full((len(items), 10), old[0], dtype=np.int32)
    np.savez(root / "items.npz", movie_id=items, year=items + 1000,
             title=title, genres=genres, neighbors=neighbors)

    manifest = {"splits": {}}
    for batch, name in enumerate(("meta_a", "meta_b")):
        movies = np.repeat(old, 20)
        features = np.zeros((len(movies), FEATURE_WIDTH), dtype=np.int32)
        features[:, MOVIE_INDEX] = movies
        features[:, USER_INDEX] = np.tile(np.arange(20), len(old)) + 300 + batch * 20
        labels = (features[:, USER_INDEX] % 2).astype(np.uint8)
        rows = np.random.default_rng(batch).permutation(len(movies))
        np.savez(root / f"{name}.npz", features=features[rows], label=labels[rows])
        manifest["splits"][name] = {"rows": len(movies)}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, path, holdout["pseudo_movies"]


@pytest.fixture
def meta_source(tmp_path):
    return make_meta_source(tmp_path)


def test_meta_tasks_pair_by_movie(meta_source):
    root, path, pseudo = meta_source
    tasks = load_meta_tasks(root, path)
    assert set(tasks) == set(pseudo)
    for movie, task in tasks.items():
        assert task.movie_id == movie
        assert task.attributes.tolist() == [movie + 1000, 10, 11, 0, 0, 0, 20, 21, 22, 0, 0]
        for name, features, labels in (
            ("meta_a", task.support_features, task.support_labels),
            ("meta_b", task.query_features, task.query_labels),
        ):
            with np.load(root / f"{name}.npz") as rows:
                keep = rows["features"][:, MOVIE_INDEX] == movie
                np.testing.assert_array_equal(features, rows["features"][keep])
                np.testing.assert_array_equal(labels, rows["label"][keep])
            assert features.dtype == task.attributes.dtype == torch.long
            assert features.shape == (20, FEATURE_WIDTH)
            assert labels.dtype == torch.float32 and labels.shape == (20,)


@pytest.mark.parametrize("problem", ["missing_movie", "short", "padding", "duplicate", "overlap"])
def test_meta_tasks_reject_invalid_feedback(meta_source, problem):
    root, path, pseudo = meta_source
    with np.load(root / "meta_b.npz") as rows:
        features, labels = rows["features"], rows["label"]
    matches = np.flatnonzero(features[:, MOVIE_INDEX] == pseudo[0])
    if problem == "missing_movie":
        features[matches, MOVIE_INDEX] = pseudo[1]
    elif problem == "short":
        features[matches[0], MOVIE_INDEX] = pseudo[1]
    elif problem == "padding":
        features[matches[0], USER_INDEX] = 0
    elif problem == "duplicate":
        features[matches[0], USER_INDEX] = features[matches[1], USER_INDEX]
    elif problem == "overlap":
        features[matches[0], USER_INDEX] = 300
    np.savez(root / "meta_b.npz", features=features, label=labels)
    with pytest.raises(ValueError):
        load_meta_tasks(root, path)


def test_meta_movie_split_is_fixed_and_nested(tmp_path):
    root, path, movies = make_meta_source(tmp_path, old_count=1058)
    train, val = make_meta_split(load_meta_tasks(root, path))
    expected = np.random.Generator(np.random.PCG64(2026)).permutation(sorted(movies)).tolist()
    assert val == expected[:21]
    assert train == expected[21:]
    assert set(train[:25]) < set(train[:50]) < set(train)
    assert set(val).isdisjoint(train)
    for name in ("meta_a", "meta_b"):
        with np.load(root / f"{name}.npz") as rows:
            features, labels = rows["features"], rows["label"]
        np.savez(root / f"{name}.npz", features=features[::-1], label=1 - labels[::-1])
    assert make_meta_split(load_meta_tasks(root, path)) == (train, val)
    assert make_meta_split(reversed(movies)) == (train, val)
    with pytest.raises(ValueError, match="split_seed"):
        make_meta_split(movies, None)
    with pytest.raises(ValueError):
        make_meta_split(movies[:-1] + [movies[0]])


def test_visible_hist_uses_only_open_positive_users():
    features = torch.zeros(5, FEATURE_WIDTH, dtype=torch.long)
    features[:, MOVIE_INDEX] = 44
    features[:, USER_INDEX] = torch.tensor([91, 92, 91, 93, 94])
    labels = torch.tensor([1., 0., 1., 0., 1.])
    assert sorted(positive_users(features, labels).tolist()) == [91, 94]
    for empty in (positive_users(features[:0], labels[:0]),
                  positive_users(features, torch.zeros_like(labels))):
        assert empty.shape == (0,) and empty.dtype == torch.long
    features[-1, MOVIE_INDEX] = 45
    with pytest.raises(ValueError):
        positive_users(features, labels)
