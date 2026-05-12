"""BiLSTM + 加性 Attention。"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (B, L, H), mask: (B, L) True=有效
        scores = self.v(torch.tanh(self.w(h))).squeeze(-1)        # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)                       # (B, L)
        pooled = (h * attn.unsqueeze(-1)).sum(dim=1)               # (B, H)
        return pooled, attn


class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.4,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = AdditiveAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, **_) -> torch.Tensor:
        x = self.embedding(input_ids)
        h, _ = self.lstm(x)
        mask = input_ids != self.padding_idx
        if not mask.any():
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        pooled, _ = self.attn(h, mask)
        return self.fc(self.dropout(pooled))

    def forward_with_attn(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        h, _ = self.lstm(x)
        mask = input_ids != self.padding_idx
        pooled, attn = self.attn(h, mask)
        return self.fc(self.dropout(pooled)), attn
