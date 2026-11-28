"""One-step adaptation and query scoring on MovieLens movie tasks."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from coldstart.data.meta_tasks import MovieTask
from coldstart.data.visible_hist import positive_users
from coldstart.models.base_dnn import BaseDNN
from coldstart.models.generators.attribute import MetaGenerator


@dataclass(frozen=True)
class TaskScores:
    initial: torch.Tensor
    adapted: torch.Tensor
    calibrated: torch.Tensor
    history_means: torch.Tensor
    support_logits: torch.Tensor
    query_logits: torch.Tensor


def _score_interactions(backbone, features, movie_vectors):
    movies, rows = features.shape[:2]
    vectors = movie_vectors[:, None, :].expand(-1, rows, -1)
    logits = backbone(features.flatten(0, 1), movie_embedding=vectors.flatten(0, 1))
    return logits.reshape(movies, rows)


def score_tasks(backbone: BaseDNN, generator: MetaGenerator, tasks: Sequence[MovieTask],
                inner_lr: float, *, training: bool) -> TaskScores:
    """Build movie vectors using A, then predict B without accessing its labels."""
    if not tasks:
        raise ValueError("meta scoring needs at least one movie")
    if any(param.requires_grad for param in backbone.parameters()):
        raise ValueError("meta scoring requires a frozen backbone")
    attributes = torch.stack([task.attributes for task in tasks])
    support_features = torch.stack([task.support_features for task in tasks])
    support_labels = torch.stack([task.support_labels for task in tasks])

    with torch.set_grad_enabled(training):
        initial = generator.initial(attributes, backbone.embedding)
    # Validation still needs the support derivative w.r.t. its temporary movie vector.
    with torch.enable_grad():
        support_vector = initial if training else initial.detach().requires_grad_()
        support_logits = _score_interactions(backbone, support_features, support_vector)
        losses = F.binary_cross_entropy_with_logits(
            support_logits, support_labels.to(support_logits), reduction="none"
        )
        per_movie = losses.mean(dim=1)
        gradient, = torch.autograd.grad(per_movie.sum(), support_vector, create_graph=training)

    with torch.set_grad_enabled(training):
        adapted = support_vector - inner_lr * gradient
        histories = []
        for task in tasks:
            users = positive_users(task.support_features, task.support_labels)
            # Empty histories sum to zero; each distinct positive user has equal weight.
            histories.append(backbone.embedding(users).sum(dim=0) / max(1, users.numel()))
        history_means = torch.stack(histories)
        calibrated = generator.calibrate(adapted, history_means)
        query_features = torch.stack([task.query_features for task in tasks])
        query_logits = _score_interactions(backbone, query_features, calibrated)
    if not training:
        initial = initial.detach()
        support_logits = support_logits.detach()
    return TaskScores(initial, adapted, calibrated, history_means, support_logits, query_logits)


def query_loss(scores: TaskScores, tasks: Sequence[MovieTask]) -> torch.Tensor:
    """Mean per-movie query BCE, computed after all B logits are available."""
    labels = torch.stack([task.query_labels for task in tasks]).to(scores.query_logits)
    return F.binary_cross_entropy_with_logits(scores.query_logits, labels)
