"""Prepare three independent MovieLens feedback batches for warm-up updates."""

import argparse
import errno
import filecmp
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from coldstart.data.movielens import MovieLensDataset
from coldstart.data.schema import MOVIE_INDEX, USER_INDEX, Purpose, Split
from coldstart.data.splits import WARM_ARRIVAL_ALGORITHM, warm_second_round_mask


def _publish(staged, output):
    """Publish the complete directory or verify a previous identical publication."""
    if not output.exists() and not output.is_symlink():
        try:
            staged.rename(output)
            return
        except OSError as error:
            # Another preparer may have published after the path check.
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR):
                raise

    if output.is_symlink() or not output.is_dir():
        raise FileExistsError(f"conflicting arrivals path at {output}")
    if {p.name for p in output.iterdir()} != {p.name for p in staged.iterdir()}:
        raise FileExistsError(f"conflicting arrivals directory at {output}")
    for entry in staged.iterdir():
        target = output / entry.name
        if target.is_symlink() or not target.is_file() or not filecmp.cmp(entry, target, shallow=False):
            raise FileExistsError(f"conflicting arrivals artifact at {target}")


def prepare_warm_arrivals(root, holdout_path, output):
    """Split warm A/B by interaction key and publish rows in their source order."""
    holdout_bytes = Path(holdout_path).read_bytes()
    holdout = json.loads(holdout_bytes)
    if not isinstance(holdout, dict) or "test_movies" not in holdout:
        raise ValueError("holdout must include test_movies")
    warm_a = MovieLensDataset(root, Split.WARM_A, Purpose.PARTITION)
    warm_b = MovieLensDataset(root, Split.WARM_B, Purpose.PARTITION)
    columns = [USER_INDEX, MOVIE_INDEX]
    second = warm_second_round_mask(
        warm_a.features.numpy()[:, columns], warm_b.features.numpy()[:, columns],
        holdout["test_movies"],
    )
    rounds = (
        (Split.WARM_1, warm_a, slice(None), 20),
        (Split.WARM_2, warm_b, second, 10),
        (Split.WARM_3, warm_b, ~second, 10),
    )
    manifest = {
        "algorithm": WARM_ARRIVAL_ALGORITHM,
        "holdout_sha256": hashlib.sha256(holdout_bytes).hexdigest(),
        "splits": {},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".warm-arrivals-", dir=output.parent) as scratch:
        staged = Path(scratch) / "arrivals"
        staged.mkdir()
        for split, dataset, keep, per_movie in rounds:
            features = dataset.features.numpy()[keep].astype(np.int32, copy=False)
            labels = dataset.labels.numpy()[keep].astype(np.uint8, copy=False)
            np.savez(staged / f"{split.value}.npz", features=features, label=labels)
            manifest["splits"][split.value] = {
                "source": dataset.split.value, "rows": len(labels), "rows_per_movie": per_movie,
            }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        _publish(staged, output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--holdout-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrivals = prepare_warm_arrivals(args.root, args.holdout_path, args.output)
    for name, entry in arrivals["splits"].items():
        print(f"{name}: {entry['rows']} rows ({entry['rows_per_movie']} per movie)")
