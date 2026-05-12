"""PyTorch Dataset 封装。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IndexedTextDataset(Dataset):
    """基于词表索引的 Dataset，用于 TextCNN / BiLSTM。"""

    def __init__(self, ids: np.ndarray, lengths: np.ndarray, labels: np.ndarray):
        self.ids = torch.from_numpy(ids).long()
        self.lengths = torch.from_numpy(lengths).long()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return self.ids.size(0)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.ids[i],
            "length": self.lengths[i],
            "label": self.labels[i],
        }


class BertTextDataset(Dataset):
    """直接持有原始文本 + 标签，tokenizer 在 collate 时调用。"""

    def __init__(self, texts: list[str], labels: np.ndarray | list[int]):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        return {"text": self.texts[i], "label": int(self.labels[i])}


def make_bert_collate_fn(tokenizer, max_len: int):
    def _collate(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return _collate


def df_to_indexed(
    df: pd.DataFrame, vocab: dict[str, int], max_len: int
) -> IndexedTextDataset:
    from src.data.preprocess import encode_batch
    ids, lens = encode_batch(df["text"].tolist(), vocab, max_len)
    labels = df["label"].to_numpy(dtype=np.int64)
    return IndexedTextDataset(ids, lens, labels)
