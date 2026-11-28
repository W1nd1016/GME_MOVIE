"""Build a movie's user history from feedback already available for that movie."""

import torch

from .schema import FEATURE_WIDTH, MOVIE_INDEX, USER_INDEX


def positive_users(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the distinct positive user IDs for a single movie's open interactions."""
    if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH or labels.shape != features.shape[:1]:
        raise ValueError("expected one feature row and one label per open interaction")
    if torch.unique(features[:, MOVIE_INDEX]).numel() > 1:
        raise ValueError("visible feedback must belong to a single movie")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("visible feedback must contain binary labels")
    users = features[:, USER_INDEX]
    if (users <= 0).any():
        raise ValueError("open interactions must have non-padding user IDs")
    return users[labels == 1].unique().long()
