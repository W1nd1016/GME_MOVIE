"""Shared-embedding CTR backbone for MovieLens interactions."""

import torch
from torch import nn

from coldstart.data.schema import SLOTS, VOCAB_SIZE


class BaseDNN(nn.Module):
    def __init__(self, slots=SLOTS, vocab_size=VOCAB_SIZE, embedding_dim=10):
        super().__init__()
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map [batch, feature_width] integer IDs to [batch] raw logits."""
        if features.ndim != 2 or features.shape[1] != self.feature_width:
            raise ValueError(f"expected features with shape [batch, {self.feature_width}]")
        embedded = self.embedding(features)
        embedded = embedded.masked_fill(features.eq(0).unsqueeze(-1), 0)
        slots = embedded.split(self.slot_widths, dim=1)
        pooled = torch.cat([slot.sum(dim=1) for slot in slots], dim=1)
        return self.mlp(pooled).squeeze(-1)
