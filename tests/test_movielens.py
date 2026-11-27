"""Verify the MovieLens conversion against the read-only data shipped in the repo."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "base_candidate": ("data/big_train_main", 765669),
    "meta_a": ("data_with_hist/train_oneshot_a_w_ngb_hist", 21160),
    "meta_b": ("data_with_hist/train_oneshot_b_w_ngb_hist", 21160),
    "warm_a": ("data_with_hist/test_oneshot_a_hist", 22540),
    "warm_b": ("data_with_hist/test_oneshot_b_hist", 22540),
    "final_eval": ("data_with_hist/test_test_w_ngb_hist", 123787),
}


def run_legacy(*args):
    subprocess.run(["conda", "run", "-n", "gme_movie_tf1", "python", *args],
                   cwd=ROOT, check=True)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    output = tmp_path_factory.mktemp("movielens") / "converted"
    run_legacy("scripts/convert_movielens.py", "--output", str(output))
    return output


def test_conversion_preserves_rows_and_fields(converted):
    from torch.utils.data import DataLoader, Subset

    from coldstart.data.movielens import MovieLensDataset
    from coldstart.data.schema import Purpose, Split

    manifest = json.loads((converted / "manifest.json").read_text())["splits"]
    assert set(manifest) == set(SOURCES)
    for name, (source, count) in SOURCES.items():
        assert manifest[name]["source"] == source + ".tfrecord"
        assert manifest[name]["rows"] == count
        with np.load(converted / f"{name}.npz") as rows:
            assert set(rows.files) == {"label", "features"}
            assert rows["features"].shape == (count, 16)
            assert rows["label"].shape == (count,)
            actual = np.column_stack((rows["label"], rows["features"]))
        if name == "final_eval":
            # This split has no CSV. Digest all original TF Example labels and
            # current features, in source order, encoded as little-endian uint16.
            assert hashlib.sha256(actual.astype("<u2").tobytes()).hexdigest() == (
                "577753d3589cd2cb0e11ba02022c8a937fa9ce636be61a68b4c1cb88ae53fe58"
            )
        else:
            expected = np.loadtxt(ROOT / (source + ".csv"), delimiter=",",
                                  dtype=np.int32, usecols=range(17))
            np.testing.assert_array_equal(actual, expected)

    dataset = MovieLensDataset(converted, Split.FINAL_EVAL, purpose=Purpose.SCORE)
    assert len(dataset) == 123787
    tail = Subset(dataset, range(123648, len(dataset)))
    batches = list(DataLoader(tail, batch_size=128, shuffle=False))
    assert [len(x) for x, _ in batches] == [128, 11]
    features, labels = batches[-1]
    assert features[-1].tolist() == [
        10545, 1180, 8, 3, 2227, 11126, 2760, 2761, 0, 0, 0, 21, 0, 0, 0, 0
    ]
    assert labels[-1].item() == 0


def test_item_table_matches_sources(converted):
    from coldstart.data.movielens import load_items

    items = load_items(converted)
    assert set(items) == {"movie_id", "year", "title", "genres", "neighbors"}
    ids = items["movie_id"]
    assert len(np.unique(ids)) == len(ids) == 2185
    # Sort once so this check is independent of the converter's file ordering.
    order = np.argsort(ids)
    sorted_ids = ids[order]
    with np.load(converted / "base_candidate.npz") as base:
        old_ids = np.unique(base["features"][:, 1])
    assert len(old_ids) == 1058
    assert np.isin(items["neighbors"], old_ids).all()
    assert items["neighbors"].shape == (2185, 10)
    attributes = np.column_stack((items["year"], items["title"], items["genres"]))
    for name, (source, _) in SOURCES.items():
        with np.load(converted / f"{name}.npz") as data:
            features = data["features"]
        rows = order[np.searchsorted(sorted_ids, features[:, 1])]
        np.testing.assert_array_equal(ids[rows], features[:, 1])
        np.testing.assert_array_equal(attributes[rows], features[:, 5:16])
        if name in ("meta_a", "meta_b"):
            # CSV column 0 is the label; each of ten neighbors has 16 fields.
            neighbors = np.loadtxt(ROOT / (source + ".csv"), delimiter=",",
                                   dtype=np.int32, usecols=range(18, 177, 16))
            np.testing.assert_array_equal(items["neighbors"][rows], neighbors)
    # Test-movie neighbor IDs have no CSV. Hash sorted (movie, 10 neighbors)
    # tuples extracted independently from the original final TFRecord.
    new_rows = order[~np.isin(sorted_ids, old_ids)]
    new_neighbors = np.column_stack((ids[new_rows], items["neighbors"][new_rows]))
    assert hashlib.sha256(new_neighbors.astype("<u2").tobytes()).hexdigest() == (
        "ff124880f1dffacdef9618a43e8eb0a3da561f9ae36396748451525057b1af31"
    )


def test_legacy_hist_is_excluded():
    # Run in the TF1 environment to mutate the actual protobuf field. The
    # decoder's output is the only data passed to the NPZ and item writers.
    run_legacy("-c", """
import tensorflow as tf
from scripts.convert_movielens import decode_record
path = 'data_with_hist/test_test_w_ngb_hist.tfrecord'
raw = next(tf.io.tf_record_iterator(path))
example = tf.train.Example.FromString(raw)
hist = example.features.feature['hist'].int64_list.value
assert len(hist) == 20 and any(hist)
hist[:] = [0] * 20
assert decode_record(raw, True) == decode_record(example.SerializeToString(), True)
""")


@pytest.mark.parametrize("problem", ["extra_hist", "float_ids", "out_of_range", "bad_label"])
def test_rejects_malformed_samples(tmp_path, problem):
    from coldstart.data.movielens import MovieLensDataset
    from coldstart.data.schema import Purpose, Split, VOCAB_SIZE

    arrays = {"features": np.ones((1, 16), dtype=np.int32),
              "label": np.ones(1, dtype=np.int32)}
    if problem == "extra_hist":
        arrays["hist"] = np.zeros((1, 20), dtype=np.int32)
    elif problem == "float_ids":
        arrays["features"] = arrays["features"].astype(float)
    elif problem == "out_of_range":
        arrays["features"][0, 1] = VOCAB_SIZE
    else:
        arrays["label"][0] = 2
    np.savez(tmp_path / "final_eval.npz", **arrays)
    manifest = {"splits": {"final_eval": {"source": SOURCES["final_eval"][0] + ".tfrecord",
                                          "rows": 1}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, TypeError)):
        MovieLensDataset(tmp_path, Split.FINAL_EVAL, purpose=Purpose.SCORE)


@pytest.mark.parametrize("purpose", ["partition", "meta_support", "meta_query", "warm_update"])
def test_final_eval_rejects_non_scoring_access_before_io(tmp_path, purpose):
    from coldstart.data.movielens import MovieLensDataset
    from coldstart.data.schema import Purpose, Split

    with pytest.raises(ValueError, match="final_eval"):
        MovieLensDataset(tmp_path, Split.FINAL_EVAL, purpose=Purpose(purpose))
