"""Verify the MovieLens feedback available at each of the three warm-up rounds."""
import hashlib
import json
from itertools import combinations

import numpy as np
import pytest

from coldstart.data.movielens import MovieLensDataset
from coldstart.data.schema import FEATURE_WIDTH, MOVIE_INDEX, SLOT_SLICES
from scripts.prepare_holdout import prepare_holdout
from test_item_holdout import make_small_dataset
from test_movielens import converted


USER_INDEX = SLOT_SLICES["user"].start
ROUNDS = (("warm_1", "warm_a", 20), ("warm_2", "warm_b", 10), ("warm_3", "warm_b", 10))


def read_rows(root, split):
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"features", "label"}
        return archive["features"], archive["label"]


def interaction_keys(features):
    return list(zip(features[:, USER_INDEX].tolist(), features[:, MOVIE_INDEX].tolist()))


def write_source(root, split, features, labels):
    np.savez(root / f"{split}.npz", features=features, label=labels)
    path = root / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["splits"][split] = {"source": split + ".tfrecord", "rows": len(labels)}
    path.write_text(json.dumps(manifest))


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "source"
    make_small_dataset(root)
    holdouts = tmp_path / "holdouts"
    prepare_holdout(root, holdouts, 42)
    # The same users rate different movies. Interleaving the movies makes row
    # order observable separately from both hash order and per-movie grouping.
    for split, first_user in (("warm_a", 6000), ("warm_b", 7000)):
        features = np.zeros((60, FEATURE_WIDTH), dtype=np.int32)
        features[:, USER_INDEX] = np.repeat(np.arange(first_user, first_user + 20), 3)
        features[:, MOVIE_INDEX] = np.tile([110, 111, 112], 20)
        features[:, 2:] = np.arange(60)[:, None] + np.arange(FEATURE_WIDTH - 2)
        labels = (np.arange(60) % 7 < 3).astype(np.uint8)
        write_source(root, split, features, labels)
    return root, holdouts / "holdout.json"


def test_b_assignment_uses_only_interaction_keys(sources, tmp_path):
    from scripts.prepare_warm_arrivals import prepare_warm_arrivals

    root, holdout = sources
    expected = {
        110: {7000, 7001, 7004, 7005, 7006, 7007, 7009, 7010, 7014, 7018},
        111: {7000, 7003, 7005, 7007, 7008, 7009, 7015, 7017, 7018, 7019},
        112: {7002, 7004, 7005, 7006, 7010, 7012, 7013, 7014, 7016, 7017},
    }
    state = np.random.get_state()
    try:
        for seed in (123, 97):
            np.random.seed(seed)
            features, labels = read_rows(root, "warm_b")
            order = np.random.permutation(len(features))
            write_source(root, "warm_b", features[order], 1 - labels[order])
            output = tmp_path / str(seed)
            prepare_warm_arrivals(root, holdout, output)
            second = set(interaction_keys(read_rows(output, "warm_2")[0]))
            third = set(interaction_keys(read_rows(output, "warm_3")[0]))
            assert second == {(user, movie) for movie, users in expected.items() for user in users}
            assert third == set(interaction_keys(features)) - second
    finally:
        np.random.set_state(state)


def test_arrival_views_match_source_rows(sources, tmp_path):
    from scripts.prepare_warm_arrivals import prepare_warm_arrivals

    root, holdout = sources
    output = tmp_path / "arrivals"
    prepare_warm_arrivals(root, holdout, output)
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["algorithm"] == "gme-warm-v1"
    assert manifest["holdout_sha256"] == hashlib.sha256(holdout.read_bytes()).hexdigest()
    assert set(manifest["splits"]) == {name for name, _, _ in ROUNDS}
    for split, source, per_movie in ROUNDS:
        entry = manifest["splits"][split]
        assert (entry["source"], entry["rows"], entry["rows_per_movie"]) == (
            source, per_movie * 3, per_movie)
        features, labels = read_rows(output, split)
        assert features.dtype == np.int32 and labels.dtype == np.uint8
        keys = set(interaction_keys(features))
        original, original_labels = read_rows(root, source)
        keep = [key in keys for key in interaction_keys(original)]
        np.testing.assert_array_equal(features, original[keep])
        np.testing.assert_array_equal(labels, original_labels[keep])
        view = MovieLensDataset(output, split, "warm_update")
        np.testing.assert_array_equal(view.features.numpy(), original[keep])
        np.testing.assert_array_equal(view.labels.numpy(), original_labels[keep])
        movie_ids, counts = np.unique(features[:, MOVIE_INDEX], return_counts=True)
        assert movie_ids.tolist() == [110, 111, 112]
        assert counts.tolist() == [per_movie] * 3

    third = set(interaction_keys(read_rows(output, "warm_3")[0]))
    original, labels = read_rows(root, "warm_b")
    flip = [key in third for key in interaction_keys(original)]
    labels[flip] = 1 - labels[flip]
    write_source(root, "warm_b", original, labels)
    changed = tmp_path / "changed"
    prepare_warm_arrivals(root, holdout, changed)
    for split, _, _ in ROUNDS:
        features, before = read_rows(output, split)
        current, after = read_rows(changed, split)
        np.testing.assert_array_equal(current, features)
        np.testing.assert_array_equal(after, 1 - before if split == "warm_3" else before)


def test_warm_update_reads_only_requested_round(sources, tmp_path):
    root, holdout = sources
    # A nonexistent root makes an unintended disk access fail the test.
    for source in ("warm_a", "warm_b"):
        with pytest.raises(ValueError, match=source):
            MovieLensDataset(tmp_path / "absent", source, "warm_update")
    for split, _, _ in ROUNDS:
        for forbidden in ("base_train", "score", "partition", "meta_support", "meta_query"):
            with pytest.raises(ValueError, match=split):
                MovieLensDataset(tmp_path / "absent", split, forbidden)

    from scripts.prepare_warm_arrivals import prepare_warm_arrivals

    assert not (root / "final_eval.npz").exists()
    output = tmp_path / "arrivals"
    prepare_warm_arrivals(root, holdout, output)
    (output / "warm_1.npz").unlink()
    (output / "warm_3.npz").unlink()
    view = MovieLensDataset(output, "warm_2", "warm_update")
    assert len(list(view)) == 30


@pytest.mark.parametrize("problem", [
    "wrong_movies", "19_rows", "21_rows", "padding_user", "padding_movie",
    "duplicate_key", "overlap_with_a", "conflicting_output",
])
def test_arrivals_reject_invalid_sources(sources, tmp_path, problem):
    from scripts.prepare_warm_arrivals import prepare_warm_arrivals

    root, holdout = sources
    output = tmp_path / "arrivals"
    prepare_warm_arrivals(root, holdout, output)
    snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
    prepare_warm_arrivals(root, holdout, output)
    split = "warm_a" if problem in ("19_rows", "padding_movie", "duplicate_key") else "warm_b"
    features, labels = read_rows(root, split)
    expected_error = ValueError
    if problem == "wrong_movies":
        features[features[:, MOVIE_INDEX] == 110, MOVIE_INDEX] = 101
    elif problem == "19_rows":
        features, labels = features[1:], labels[1:]
    elif problem == "21_rows":
        extra = features[:1].copy()
        extra[0, USER_INDEX] = 9000
        features = np.concatenate([features, extra])
        labels = np.append(labels, labels[:1])
    elif problem == "padding_user":
        features[0, USER_INDEX] = 0
    elif problem == "padding_movie":
        features[0, MOVIE_INDEX] = 0
    elif problem == "duplicate_key":
        features[0, USER_INDEX] = features[3, USER_INDEX]
    elif problem == "overlap_with_a":
        features[0, USER_INDEX] = 6000
    elif problem == "conflicting_output":
        labels[0] = 1 - labels[0]
        expected_error = FileExistsError
    write_source(root, split, features, labels)
    with pytest.raises(expected_error):
        prepare_warm_arrivals(root, holdout, output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == snapshot


def test_real_arrivals_are_disjoint_from_final_eval(converted, tmp_path):
    from scripts.prepare_warm_arrivals import prepare_warm_arrivals

    holdouts, output = tmp_path / "holdouts", tmp_path / "arrivals"
    holdout = prepare_holdout(converted, holdouts, 42)
    movies = holdout["test_movies"]
    assert len(movies) == 1127
    prepare_warm_arrivals(converted, holdouts / "holdout.json", output)
    groups = {}
    for split, source, per_movie in ROUNDS:
        features, labels = read_rows(output, split)
        keys = interaction_keys(features)
        groups[split] = set(keys)
        movie_ids, counts = np.unique(features[:, MOVIE_INDEX], return_counts=True)
        assert len(groups[split]) == len(keys) == len(movies) * per_movie
        assert movie_ids.tolist() == movies and (counts == per_movie).all()
        original, original_labels = read_rows(converted, source)
        keep = [key in groups[split] for key in interaction_keys(original)]
        np.testing.assert_array_equal(features, original[keep])
        np.testing.assert_array_equal(labels, original_labels[keep])

    # Independent audit after publication. Only access the features member.
    with np.load(converted / "final_eval.npz") as final:
        keys = interaction_keys(final["features"])
    groups["final_eval"] = set(keys)
    assert len(groups["final_eval"]) == len(keys) == 123787
    assert {movie for _, movie in keys} == set(movies)
    for left, right in combinations(groups, 2):
        assert groups[left].isdisjoint(groups[right]), (left, right)
