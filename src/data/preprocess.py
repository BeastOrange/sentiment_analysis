"""文本预处理：清洗、分词、词表。"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import jieba
import numpy as np
import pandas as pd

jieba.setLogLevel(40)  # 静默 jieba 初始化日志

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_AT = re.compile(r"@[\w一-龥]+")
_RE_TOPIC = re.compile(r"#[^#]+#")
_RE_REPEAT = re.compile(r"(.)\1{3,}")
_RE_SPACES = re.compile(r"\s+")
_RE_NON_USEFUL = re.compile(r"[【】\[\]()（）「」『』〈〉《》]")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _RE_URL.sub(" ", text)
    text = _RE_AT.sub(" ", text)
    text = _RE_TOPIC.sub(" ", text)
    text = _RE_NON_USEFUL.sub(" ", text)
    text = _RE_REPEAT.sub(r"\1\1\1", text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    text = clean_text(text)
    return [t for t in jieba.lcut(text) if t.strip()]


def build_vocab(
    texts: list[str], min_freq: int = 2, max_size: int = 50000
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(tokenize(t))
    most_common = [w for w, c in counter.most_common(max_size) if c >= min_freq]
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w in most_common:
        vocab[w] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> tuple[np.ndarray, int]:
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    length = len(ids)
    if length < max_len:
        ids = ids + [vocab[PAD_TOKEN]] * (max_len - length)
    return np.asarray(ids, dtype=np.int64), max(length, 1)


def encode_batch(
    texts: list[str], vocab: dict[str, int], max_len: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(texts)
    out = np.zeros((n, max_len), dtype=np.int64)
    lens = np.ones(n, dtype=np.int64)
    for i, t in enumerate(texts):
        out[i], lens[i] = encode(t, vocab, max_len)
    return out, lens


def split_dataframe(
    df: pd.DataFrame,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按比例采样并切分（用于压缩到 <2h 训练预算）。返回 train/val/test。"""
    rng = np.random.RandomState(seed)
    if stratify:
        parts_train, parts_val, parts_test = [], [], []
        labels = sorted(df["label"].unique())
        n_per_train = train_size // len(labels)
        n_per_val = val_size // len(labels)
        n_per_test = test_size // len(labels)
        for lab in labels:
            sub = df[df["label"] == lab]
            need = n_per_train + n_per_val + n_per_test
            if len(sub) < need:
                idx = rng.choice(sub.index, size=need, replace=True)
            else:
                idx = rng.choice(sub.index, size=need, replace=False)
            sub = sub.loc[idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
            parts_train.append(sub.iloc[:n_per_train])
            parts_val.append(sub.iloc[n_per_train:n_per_train + n_per_val])
            parts_test.append(sub.iloc[n_per_train + n_per_val:])
        train = pd.concat(parts_train).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val = pd.concat(parts_val).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test = pd.concat(parts_test).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train = shuffled.iloc[:train_size]
        val = shuffled.iloc[train_size:train_size + val_size]
        test = shuffled.iloc[train_size + val_size:train_size + val_size + test_size]
    return train, val, test


def save_processed(
    splits: dict[str, pd.DataFrame], out_dir: Path, prefix: str
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in splits.items():
        p = out_dir / f"{prefix}_{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
    return paths
