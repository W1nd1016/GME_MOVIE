"""Convert the repository's MovieLens TFRecords in the gme_movie_tf1 environment."""

import argparse
import errno
import filecmp
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

# The legacy Python environment needs the shared schema, but does not need torch.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coldstart.data.schema import (
    FEATURE_WIDTH, GENRES_SLICE, MOVIE_INDEX, NEIGHBOR_COUNT, Split, TITLE_SLICE,
    VOCAB_SIZE, YEAR_INDEX,
)


# (source, expected number of examples, whether the data field contains neighbors)
SOURCES = {
    Split.BASE_CANDIDATE: ("data/big_train_main.tfrecord", 765669, False),
    Split.META_A: ("data_with_hist/train_oneshot_a_w_ngb_hist.tfrecord", 21160, True),
    Split.META_B: ("data_with_hist/train_oneshot_b_w_ngb_hist.tfrecord", 21160, True),
    Split.WARM_A: ("data_with_hist/test_oneshot_a_hist.tfrecord", 22540, False),
    Split.WARM_B: ("data_with_hist/test_oneshot_b_hist.tfrecord", 22540, False),
    Split.FINAL_EVAL: ("data_with_hist/test_test_w_ngb_hist.tfrecord", 123787, True),
}


def decode_record(raw, has_neighbors):
    """Read only a binary label, the current features, and neighbor movie IDs."""
    fields = tf.train.Example.FromString(raw).features.feature
    labels = fields["label"].int64_list.value
    if len(labels) != 1 or labels[0] not in (0, 1):
        raise ValueError("expected one binary label")
    data = fields["data"].int64_list.value
    width = FEATURE_WIDTH * (1 + NEIGHBOR_COUNT) if has_neighbors else FEATURE_WIDTH
    if len(data) != width:
        raise ValueError("expected {} data fields, found {}".format(width, len(data)))
    features = tuple(data[:FEATURE_WIDTH])
    if min(features) < 0 or max(features) >= VOCAB_SIZE or features[MOVIE_INDEX] == 0:
        raise ValueError("invalid feature or movie ID")
    neighbors = tuple(data[FEATURE_WIDTH + MOVIE_INDEX::FEATURE_WIDTH]) if has_neighbors else ()
    return labels[0], features, neighbors


def convert_split(source, count, has_neighbors, output, attributes, neighbors):
    labels = np.empty(count, dtype=np.uint8)
    features = np.empty((count, FEATURE_WIDTH), dtype=np.int32)
    rows = 0
    for rows, raw in enumerate(tf.io.tf_record_iterator(str(source)), start=1):
        if rows > count:
            raise ValueError("{} has more than {} rows".format(source, count))
        try:
            label, current, nearby = decode_record(raw, has_neighbors)
            movie = current[MOVIE_INDEX]
            item = (current[YEAR_INDEX], current[TITLE_SLICE], current[GENRES_SLICE])
            if attributes.setdefault(movie, item) != item:
                raise ValueError("inconsistent attributes for movie {}".format(movie))
            if has_neighbors and neighbors.setdefault(movie, nearby) != nearby:
                raise ValueError("inconsistent neighbors for movie {}".format(movie))
        except ValueError as error:
            raise ValueError("{} row {}: {}".format(source, rows, error)) from error
        labels[rows - 1] = label
        features[rows - 1] = current
    if rows != count:
        raise ValueError("{}: expected {} rows, found {}".format(source, count, rows))
    np.savez(output, label=labels, features=features)
    return set(np.unique(features[:, MOVIE_INDEX]))


def write_items(output, attributes, neighbors, old_movies):
    if len(attributes) != 2185 or len(old_movies) != 1058 or attributes.keys() != neighbors.keys():
        raise ValueError("expected metadata and neighbors for all 1058 old and 1127 new movies")
    if any(mid not in old_movies for group in neighbors.values() for mid in group):
        raise ValueError("all neighbors must be movies from base_candidate")
    movies = sorted(attributes)
    years, titles, genres = zip(*(attributes[mid] for mid in movies))
    np.savez(output, movie_id=np.array(movies, dtype=np.int32),
             year=np.array(years, dtype=np.int32), title=np.array(titles, dtype=np.int32),
             genres=np.array(genres, dtype=np.int32),
             neighbors=np.array([neighbors[mid] for mid in movies], dtype=np.int32))


def convert(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Readers see either the complete dataset or the previous directory. A failed
    # input check never exposes a mix of splits from different conversions.
    with TemporaryDirectory(prefix=".movielens-", dir=str(output.parent)) as scratch:
        staged = Path(scratch) / "dataset"
        staged.mkdir()
        attributes, neighbors, movie_ids, manifest = {}, {}, {}, {}
        for split, (source, count, has_neighbors) in SOURCES.items():
            movie_ids[split] = convert_split(
                ROOT / source, count, has_neighbors, staged / (split.value + ".npz"),
                attributes, neighbors,
            )
            manifest[split.value] = {"source": source, "rows": count}
        write_items(staged / "items.npz", attributes, neighbors, movie_ids[Split.BASE_CANDIDATE])
        (staged / "manifest.json").write_text(
            json.dumps({"splits": manifest}, indent=2) + "\n", encoding="utf-8"
        )
        try:
            staged.rename(output)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            for entry in staged.iterdir():
                existing = output / entry.name
                if not existing.is_file() or not filecmp.cmp(entry, existing, shallow=False):
                    raise FileExistsError("different artifacts already exist at {}".format(output))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name, entry in convert(args.output).items():
        print("{}: {} rows".format(name, entry["rows"]))
