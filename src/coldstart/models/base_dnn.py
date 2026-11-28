"""Shared-embedding CTR backbone for MovieLens interactions."""

import torch
from torch import nn

from coldstart.data.schema import SLOTS, VOCAB_SIZE


def pool_slots(ids, embedding, widths):
    """Embed IDs and sum within each slot, masking zero even for loaded weights."""
    embedded = embedding(ids).masked_fill(ids.eq(0).unsqueeze(-1), 0)
    return [slot.sum(dim=1) for slot in embedded.split(widths, dim=1)]


class BaseDNN(nn.Module):
    def __init__(self, slots=SLOTS, vocab_size=VOCAB_SIZE, embedding_dim=10):
        super().__init__()
        slots = tuple(slots)
        self.slot_names = tuple(name for name, _ in slots)
        self.slot_widths = tuple(width for _, width in slots)
        self.feature_width = sum(self.slot_widths)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(len(self.slot_widths) * embedding_dim, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 1, bias=False),
        )
        nn.init.normal_(self.embedding.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(
        self, features: torch.Tensor, *, movie_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score interactions, optionally replacing each row's pooled movie slot."""
        if features.ndim != 2 or features.shape[1] != self.feature_width:
            raise ValueError(f"expected features with shape [batch, {self.feature_width}]")
        pooled = pool_slots(features, self.embedding, self.slot_widths)
        if movie_embedding is not None:
            expected = (features.shape[0], self.embedding.embedding_dim)
            if movie_embedding.shape != expected:
                raise ValueError(f"expected movie_embedding with shape {expected}")
            pooled[self.slot_names.index("movie")] = movie_embedding
        return self.mlp(torch.cat(pooled, dim=1)).squeeze(-1)
