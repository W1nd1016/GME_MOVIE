"""Prepare the fixed MovieLens item split and neighbor coverage report."""

import argparse
import errno
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from coldstart.data.movielens import MovieLensDataset, load_items
from coldstart.data.schema import MOVIE_INDEX, Purpose, Split
from coldstart.data.splits import make_holdout, neighbor_coverage


def _source_movies(root, split):
    dataset = MovieLensDataset(root, split, Purpose.PARTITION)
    return np.unique(dataset.features[:, MOVIE_INDEX].numpy())


def _publish(output, documents):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        staged = Path(scratch) / "holdout"
        staged.mkdir()
        for name, document in documents.items():
            content = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
            (staged / name).write_text(content + "\n", encoding="utf-8")
        try:
            staged.rename(output)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR):
                raise
            if not output.is_dir() or {p.name for p in output.iterdir()} != documents.keys():
                raise FileExistsError(f"conflicting holdout directory at {output}") from error
            for name in documents:
                existing = output / name
                if not existing.is_file() or existing.read_bytes() != (staged / name).read_bytes():
                    raise FileExistsError(f"conflicting holdout artifact at {existing}") from error


def prepare_holdout(root, output, split_seed):
    """Validate the three old-movie sources and atomically publish both JSON files."""
    old = _source_movies(root, Split.BASE_CANDIDATE)
    for split in (Split.META_A, Split.META_B):
        if not np.array_equal(_source_movies(root, split), old):
            raise ValueError(f"{split.value} movie IDs differ from base_candidate")
    items = load_items(root)
    holdout = make_holdout(old, items["movie_id"], split_seed)
    coverage = neighbor_coverage(items, holdout)
    _publish(output, {"holdout.json": holdout, "neighbors.json": coverage})
    return holdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    args = parser.parse_args()
    split = prepare_holdout(args.root, args.output, args.split_seed)
    for name in ("base_movies", "pseudo_movies", "test_movies"):
        print(f"{name}: {len(split[name])}")
