"""Build and verify MovieLens item holdouts and warm-up arrivals."""

import hashlib
import json

import numpy as np

from .schema import NEIGHBOR_COUNT, VOCAB_SIZE


WARM_ARRIVAL_ALGORITHM = "gme-warm-v1"


def _movie_ids(values):
    ids = np.asarray(values)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("movie IDs must be a one-dimensional integer array")
    if np.any(ids <= 0) or np.any(ids >= VOCAB_SIZE):
        raise ValueError("movie IDs must be non-padding IDs within the vocabulary")
    return np.unique(ids)


def make_holdout(old_ids, item_ids, split_seed):
    """Partition old movies 9:1, then take the remaining item IDs as test movies."""
    if type(split_seed) is not int or not 0 <= split_seed < 2**32:
        raise ValueError("holdout split_seed must be an integer in [0, 2**32)")
    old = _movie_ids(old_ids)
    items = _movie_ids(item_ids)
    if items.size != len(item_ids) or not np.isin(old, items).all():
        raise ValueError("item table must include every old movie and have unique IDs")
    test = np.setdiff1d(items, old, assume_unique=True)
    if old.size < 2 or test.size == 0:
        raise ValueError("holdout needs nonempty base, pseudo and test movie groups")

    rng = np.random.Generator(np.random.PCG64(split_seed))
    shuffled = rng.permutation(old)
    pseudo_count = max(1, (old.size + 5) // 10)
    return {
        "version": 1,
        "algorithm": "PCG64",
        "split_seed": split_seed,
        "base_movies": np.sort(shuffled[pseudo_count:]).tolist(),
        "pseudo_movies": np.sort(shuffled[:pseudo_count]).tolist(),
        "test_movies": test.tolist(),
    }


def validate_holdout(raw_bytes, old_ids, item_ids):
    """Verify the recorded assignment against the source movies and recorded seed."""
    record = json.loads(raw_bytes)
    if not isinstance(record, dict) or type(record.get("version")) is not int:
        raise ValueError("invalid holdout format")
    for key in ("base_movies", "pseudo_movies", "test_movies"):
        movies = record.get(key)
        if not isinstance(movies, list) or any(type(movie) is not int for movie in movies):
            raise ValueError(f"holdout {key} must be a list of integer movie IDs")
    expected = make_holdout(old_ids, item_ids, record.get("split_seed"))
    if record != expected:
        raise ValueError("holdout does not match its format, source movie IDs or split seed")
    return record


def neighbor_coverage(items, holdout):
    """Count eligible neighbor positions by target movie for pseudo and test groups."""
    neighbors = items["neighbors"]
    base = holdout["base_movies"]
    if not np.isin(neighbors, base + holdout["pseudo_movies"]).all():
        raise ValueError("neighbor IDs must refer to old movies")

    report = {}
    for group in ("pseudo", "test"):
        selected = np.isin(items["movie_id"], holdout[group + "_movies"])
        targets = items["movie_id"][selected]
        nearby = neighbors[selected]
        counts = np.count_nonzero(np.isin(nearby, base), axis=1)
        pseudo_positions = int(nearby.size - counts.sum())
        report[group] = {
            "available_by_movie": {str(mid): int(n) for mid, n in zip(targets, counts)},
            "histogram": np.bincount(counts, minlength=NEIGHBOR_COUNT + 1).tolist(),
            "pseudo_positions": pseudo_positions,
            "total_positions": nearby.size,
            "pseudo_fraction": pseudo_positions / nearby.size,
        }
    return report


def _warm_users_by_movie(keys, movies, source):
    """Collect the 20 unique users for each expected movie in a warm source."""
    keys = np.asarray(keys)
    if keys.ndim != 2 or keys.shape[1] != 2 or not np.issubdtype(keys.dtype, np.integer):
        raise ValueError(f"{source}: expected integer (user, movie) pairs")
    grouped = {movie: set() for movie in movies}
    for user, movie in keys.tolist():
        if movie not in grouped:
            raise ValueError(f"{source}: movie {movie} is outside the test movie list")
        if not 0 < user < VOCAB_SIZE:
            raise ValueError(f"{source}: movie {movie} has invalid user ID {user}")
        users = grouped[movie]
        if user in users:
            raise ValueError(f"{source}: repeated user {user} for movie {movie}")
        users.add(user)
    for movie, users in grouped.items():
        if len(users) != 20:
            raise ValueError(f"{source}: movie {movie} has {len(users)} users, expected 20")
    return grouped


def warm_second_round_mask(a_keys, b_keys, test_movies):
    """Verify A/B interaction keys and return the round-two mask in B source order."""
    if not isinstance(test_movies, list) or not test_movies or any(
        type(movie) is not int for movie in test_movies
    ):
        raise ValueError("holdout test_movies must be a nonempty list of integer IDs")
    if len(_movie_ids(test_movies)) != len(test_movies):
        raise ValueError("holdout test_movies contains duplicate movie IDs")

    a_users = _warm_users_by_movie(a_keys, test_movies, "warm_a")
    b_users = _warm_users_by_movie(b_keys, test_movies, "warm_b")
    second_round = {}
    for movie, users in b_users.items():
        overlap = users & a_users[movie]
        if overlap:
            raise ValueError(f"warm_a and warm_b overlap for movie {movie}, user {min(overlap)}")

        def rank(user):
            key = f"{WARM_ARRIVAL_ALGORITHM}/{movie}/{user}".encode("ascii")
            return hashlib.sha256(key).digest(), user

        second_round[movie] = set(sorted(users, key=rank)[:10])
    return np.array([user in second_round[movie] for user, movie in b_keys], dtype=bool)
