"""Contracts for the fixed MovieLens item holdout and its modeling views."""
import json
from pathlib import Path

import numpy as np
import pytest

from coldstart.data.movielens import MovieLensDataset
from coldstart.data.schema import FEATURE_WIDTH, MOVIE_INDEX
from test_movielens import converted


VIEWS = (
    ("base_train", "base_train", "base_candidate", "base_movies"),
    ("meta_a", "meta_support", "meta_a", "pseudo_movies"),
    ("meta_b", "meta_query", "meta_b", "pseudo_movies"),
)


def make_small_dataset(root):
    """Build three real-format source splits with unique row IDs and no final_eval."""
    root.mkdir(exist_ok=True)
    old = np.arange(100, 110, dtype=np.int32)
    all_movies = np.arange(100, 113, dtype=np.int32)
    manifest = {"splits": {}}
    for index, name in enumerate(("base_candidate", "meta_a", "meta_b")):
        movies = np.concatenate((old[::-1], old))
        features = np.zeros((len(movies), FEATURE_WIDTH), dtype=np.int32)
        features[:, 0] = np.arange(len(movies)) + 500 + index * 100
        features[:, MOVIE_INDEX] = movies
        labels = np.arange(len(movies), dtype=np.uint8) % 2
        np.savez(root / f"{name}.npz", features=features, label=labels)
        manifest["splits"][name] = {"rows": len(movies), "source": name + ".tfrecord"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    # PCG64 with seed 42 holds out 105. Test items 110/111/112 retain 10/6/0 slots.
    neighbors = np.full((len(all_movies), 10), 100, dtype=np.int32)
    neighbors[5, :2] = 105
    neighbors[11, :4] = 105
    neighbors[12, :] = 105
    tokens = np.zeros((len(all_movies), 5), dtype=np.int32)
    np.savez(root / "items.npz", movie_id=all_movies, year=all_movies * 0,
             title=tokens, genres=tokens, neighbors=neighbors)


def test_holdout_is_reproducible_and_label_independent(tmp_path):
    from scripts.prepare_holdout import prepare_holdout

    root, output = tmp_path / "source", tmp_path / "split"
    make_small_dataset(root)
    prepare_holdout(root, output, 42)
    files = {name: (output / name).read_bytes() for name in ("holdout.json", "neighbors.json")}
    holdout = json.loads(files["holdout.json"])
    assert holdout["version"] == 1 and holdout["algorithm"] == "PCG64"
    assert holdout["split_seed"] == 42
    assert holdout["pseudo_movies"] == [105]
    assert holdout["base_movies"] == [m for m in range(100, 110) if m != 105]
    assert holdout["test_movies"] == [110, 111, 112]

    for _, _, source, _ in VIEWS:
        path = root / (source + ".npz")
        with np.load(path) as samples:
            features, labels = samples["features"], samples["label"]
        np.savez(path, features=features[::-1], label=1 - labels[::-1])
    prepare_holdout(root, output, 42)
    assert all((output / name).read_bytes() == data for name, data in files.items())
    with pytest.raises(FileExistsError):
        prepare_holdout(root, output, 7)
    with np.load(root / "meta_b.npz") as samples:
        features, labels = samples["features"], samples["label"]
    features[features[:, MOVIE_INDEX] == 109, MOVIE_INDEX] = 108
    np.savez(root / "meta_b.npz", features=features, label=labels)
    with pytest.raises(ValueError):
        prepare_holdout(root, output, 42)
    assert all((output / name).read_bytes() == data for name, data in files.items())


def test_holdout_views_match_source_rows(converted, tmp_path):
    from scripts.prepare_holdout import prepare_holdout

    output = tmp_path / "split"
    prepare_holdout(converted, output, 42)
    path = output / "holdout.json"
    holdout = json.loads(path.read_bytes())
    groups = [set(holdout[key]) for key in ("base_movies", "pseudo_movies", "test_movies")]
    assert [len(group) for group in groups] == [952, 106, 1127]
    assert len(set.union(*groups)) == sum(map(len, groups))
    with np.load(converted / "items.npz") as items:
        assert set(items["movie_id"]) == set.union(*groups)
    for split in ("warm_a", "warm_b"):
        with np.load(converted / (split + ".npz")) as samples:
            assert set(samples["features"][:, MOVIE_INDEX]) == groups[2]
    for split, purpose, source, group_name in VIEWS:
        dataset = MovieLensDataset(converted, split, purpose, holdout_path=path)
        with np.load(converted / (source + ".npz")) as samples:
            features = samples["features"]
            keep = np.isin(features[:, MOVIE_INDEX], holdout[group_name])
            np.testing.assert_array_equal(dataset.features.numpy(), features[keep])
            np.testing.assert_array_equal(dataset.labels.numpy(), samples["label"][keep])
        if group_name == "pseudo_movies":
            assert len(dataset) == 2120
            movies, counts = np.unique(dataset.features[:, MOVIE_INDEX], return_counts=True)
            np.testing.assert_array_equal(movies, holdout["pseudo_movies"])
            assert (counts == 20).all()


@pytest.mark.parametrize(("split", "purpose", "source", "group_name"), VIEWS)
def test_modeling_requires_verified_holdout(tmp_path, split, purpose, source, group_name):
    from scripts.prepare_holdout import prepare_holdout

    root, output = tmp_path / "source", tmp_path / "split"
    make_small_dataset(root)
    with pytest.raises(ValueError):
        MovieLensDataset(root, split, purpose)
    prepare_holdout(root, output, 42)
    path = output / "holdout.json"
    original = path.read_bytes()
    for key, wrong_value in (("version", 2), ("algorithm", "MT19937"), ("split_seed", 7)):
        holdout = json.loads(original)
        holdout[key] = wrong_value
        path.write_text(json.dumps(holdout))
        with pytest.raises(ValueError):
            MovieLensDataset(root, split, purpose, holdout_path=path)
    holdout = json.loads(original)
    holdout["base_movies"][0], holdout["pseudo_movies"][0] = 105, 100
    holdout["base_movies"].sort()
    path.write_text(json.dumps(holdout))
    with pytest.raises(ValueError):
        MovieLensDataset(root, split, purpose, holdout_path=path)

    path.write_bytes(original)
    source_path = root / (source + ".npz")
    with np.load(source_path) as samples:
        features, labels = samples["features"], samples["label"]
    features[features[:, MOVIE_INDEX] == 100, MOVIE_INDEX] = 110
    np.savez(source_path, features=features, label=labels)
    with pytest.raises(ValueError):
        MovieLensDataset(root, split, purpose, holdout_path=path)


def test_holdout_does_not_open_final_eval(tmp_path, monkeypatch):
    from scripts.prepare_holdout import prepare_holdout

    root, output = tmp_path / "source", tmp_path / "split"
    make_small_dataset(root)
    assert not (root / "final_eval.npz").exists()
    prepare_holdout(root, output, 42)

    # Also guard the normal case where the file exists. Reading even its features
    # would let the evaluation set influence either the split or the training data.
    final_eval = root / "final_eval.npz"
    final_eval.touch()
    original_load = np.load

    def guard_final_eval(path, *args, **kwargs):
        assert Path(path) != final_eval, "opened final_eval during holdout construction"
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guard_final_eval)
    output = tmp_path / "split-with-eval"
    prepare_holdout(root, output, 42)
    for split, purpose, _, _ in VIEWS:
        view = MovieLensDataset(root, split, purpose, holdout_path=output / "holdout.json")
        assert len(view) > 0


def test_neighbor_coverage_counts_only_base_movies(tmp_path):
    from scripts.prepare_holdout import prepare_holdout

    root, output = tmp_path / "source", tmp_path / "split"
    make_small_dataset(root)
    original = (root / "items.npz").read_bytes()
    prepare_holdout(root, output, 42)
    assert (root / "items.npz").read_bytes() == original
    stats = json.loads((output / "neighbors.json").read_bytes())
    for name, counts, pseudo, total in (
        ("pseudo", {"105": 8}, 2, 10),
        ("test", {"110": 10, "111": 6, "112": 0}, 14, 30),
    ):
        group = stats[name]
        assert group["available_by_movie"] == counts
        assert group["histogram"] == [list(counts.values()).count(n) for n in range(11)]
        assert group["pseudo_positions"] == pseudo
        assert group["total_positions"] == total
        assert group["pseudo_fraction"] == pytest.approx(pseudo / total)
