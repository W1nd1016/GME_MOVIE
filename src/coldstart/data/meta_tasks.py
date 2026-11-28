"""Group the verified pseudo-movie views into MovieLens support/query tasks."""

from dataclasses import dataclass

import numpy as np
import torch

from .movielens import MovieLensDataset, load_items
from .schema import MOVIE_INDEX, USER_INDEX, VOCAB_SIZE, Purpose, Split


@dataclass(frozen=True)
class MovieTask:
    movie_id: int
    attributes: torch.Tensor
    support_features: torch.Tensor
    support_labels: torch.Tensor
    query_features: torch.Tensor
    query_labels: torch.Tensor


def _feedback_by_movie(dataset):
    movies = dataset.features[:, MOVIE_INDEX]
    groups = {}
    for movie in torch.unique(movies).tolist():
        keep = movies == movie
        features = dataset.features[keep]
        users = features[:, USER_INDEX]
        if len(users) != 20 or torch.unique(users).numel() != 20 or (users == 0).any():
            raise ValueError(f"{dataset.split.value}: movie {movie} needs 20 distinct non-padding users")
        groups[movie] = (features, dataset.labels[keep])
    return groups


def load_meta_tasks(root, holdout_path) -> dict[int, MovieTask]:
    """Pair A and B by movie ID, preserving row order separately in each batch."""
    support = _feedback_by_movie(MovieLensDataset(
        root, Split.META_A, Purpose.META_SUPPORT, holdout_path=holdout_path
    ))
    query = _feedback_by_movie(MovieLensDataset(
        root, Split.META_B, Purpose.META_QUERY, holdout_path=holdout_path
    ))
    if support.keys() != query.keys():
        raise ValueError("meta_a and meta_b must cover the same pseudo movies")

    items = load_items(root)
    rows = {int(movie): i for i, movie in enumerate(items["movie_id"])}
    attributes = np.column_stack((items["year"], items["title"], items["genres"]))
    tasks = {}
    for movie, (a_features, a_labels) in support.items():
        b_features, b_labels = query[movie]
        if torch.isin(a_features[:, USER_INDEX], b_features[:, USER_INDEX]).any():
            raise ValueError(f"meta_a and meta_b share a user for movie {movie}")
        tasks[movie] = MovieTask(
            movie, torch.as_tensor(attributes[rows[movie]], dtype=torch.long),
            a_features, a_labels, b_features, b_labels,
        )
    return tasks


def make_meta_split(movie_ids, split_seed=2026) -> tuple[list[int], list[int]]:
    """Return 85 training IDs (nested subsets use prefixes) and 21 validation IDs."""
    if type(split_seed) is not int or not 0 <= split_seed < 2**32:
        raise ValueError("split_seed must be an integer in [0, 2**32)")
    ids = np.asarray(list(movie_ids))
    if (ids.shape != (106,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids <= 0) or np.any(ids >= VOCAB_SIZE) or np.unique(ids).size != 106):
        raise ValueError("stage 0 meta split requires 106 distinct non-padding movie IDs")
    shuffled = np.random.Generator(np.random.PCG64(split_seed)).permutation(np.sort(ids)).tolist()
    return shuffled[21:], shuffled[:21]
