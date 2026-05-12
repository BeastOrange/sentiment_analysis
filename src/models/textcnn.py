"""TextCNN：Kim 2014 多尺度卷积核架构。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 200,
        num_classes: int = 2,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        num_filters: int = 96,
        dropout: float = 0.4,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, **_) -> torch.Tensor:
        x = self.embedding(input_ids)             # (B, L, E)
        x = x.transpose(1, 2)                     # (B, E, L)
        feats = [F.relu(conv(x)) for conv in self.convs]
        feats = [F.max_pool1d(f, f.size(-1)).squeeze(-1) for f in feats]
        cat = torch.cat(feats, dim=1)             # (B, F * len(K))
        return self.fc(self.dropout(cat))
