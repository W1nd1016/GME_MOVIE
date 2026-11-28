"""Attribute and shared cold starts with calibration from visible feedback."""

import torch
from torch import nn

from coldstart.data.schema import SLOTS
from coldstart.models.base_dnn import pool_slots


_ITEM_SLOTS = ("year", "title", "genres")
_ITEM_WIDTHS = tuple(dict(SLOTS)[name] for name in _ITEM_SLOTS)


class MetaGenerator(nn.Module):
    def __init__(self, embedding_dim: int, method: str):
        super().__init__()
        self.method = method
        if method == "attribute":
            self.attribute = nn.Linear(len(_ITEM_SLOTS) * embedding_dim, embedding_dim)
            nn.init.zeros_(self.attribute.weight)
            nn.init.zeros_(self.attribute.bias)
        elif method == "shared":
            self.shared = nn.Parameter(torch.zeros(embedding_dim))
        else:
            raise ValueError(f"unknown initial embedding method: {method}")
        self.calibration = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.zeros_(self.calibration.weight)

    def initial(self, attributes: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        """Return one cold vector per movie from its year, title and genre IDs."""
        if attributes.ndim != 2 or attributes.shape[1] != sum(_ITEM_WIDTHS):
            raise ValueError(f"expected attributes with shape [movies, {sum(_ITEM_WIDTHS)}]")
        if self.method == "shared":
            return self.shared.expand(attributes.shape[0], -1)
        slots = pool_slots(attributes, embedding, _ITEM_WIDTHS)
        return self.attribute(torch.cat(slots, dim=1))

    def initial_parameters(self):
        """Parameters whose query gradients measure learning of cold vectors."""
        if self.method == "shared":
            return (self.shared,)
        return self.attribute.parameters()

    def calibrate(self, adapted: torch.Tensor, history_means: torch.Tensor) -> torch.Tensor:
        """Add a learned residual for each movie's mean positive-user embedding."""
        if (adapted.ndim != 2 or adapted.shape != history_means.shape
                or adapted.shape[1] != self.calibration.in_features):
            raise ValueError("adapted vectors and history means must both have shape [movies, dim]")
        return adapted + self.calibration(history_means)
