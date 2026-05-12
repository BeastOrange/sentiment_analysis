"""脚本 03：预处理 + 拆分 + 词表构建。

产出：
- data/processed/binary_{train,val,test}.parquet
- data/processed/emotion_{train,val,test}.parquet
- data/processed/vocab_binary.json, vocab_emotion.json
- outputs/figures/preprocess_*.png
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CFG, FIGURES_DIR, PROCESSED_DIR, SEED
from src.data.loader import load_binary, load_emotion
from src.data.preprocess import (
    build_vocab,
    clean_text,
    save_processed,
    split_dataframe,
    tokenize,
)
from src.utils import dump_json, get_logger, set_seed, timed
from src.viz import apply_style, palette, save_fig

log = get_logger("03_preprocess")


def viz_preprocess_effect(df: pd.DataFrame, fname: str) -> None:
    sample = df.sample(min(2000, len(df)), random_state=SEED)
    raw_lens = sample["text"].str.len()
    cleaned = sample["text"].apply(clean_text)
    clean_lens = cleaned.str.len()
    token_counts = cleaned.apply(lambda x: len(tokenize(x)))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    ax = axes[0]
    ax.hist([raw_lens.clip(upper=200), clean_lens.clip(upper=200)],
            bins=30, label=["原始", "清洗后"],
            color=palette(2, "primary"), edgecolor="white", alpha=0.85)
    ax.set_title("清洗前后字符长度对比")
    ax.set_xlabel("字符数"); ax.set_ylabel("样本数"); ax.legend()

    ax = axes[1]
    diff = (raw_lens - clean_lens).clip(lower=0, upper=120)
    ax.hist(diff, bins=30, color=palette(3, "primary")[2], edgecolor="white", alpha=0.85)
    ax.set_title("被清洗掉的字符数")
    ax.set_xlabel("差值"); ax.set_ylabel("样本数")

    ax = axes[2]
    ax.hist(token_counts.clip(upper=80), bins=30,
            color=palette(4, "primary")[3], edgecolor="white", alpha=0.85)
    ax.axvline(CFG.data.max_len, color="#DC2626", linestyle="--", linewidth=1.5,
               label=f"截断长度={CFG.data.max_len}")
    ax.set_title("分词 Token 数分布")
    ax.set_xlabel("Token 数"); ax.set_ylabel("样本数"); ax.legend()
    fig.suptitle("文本预处理效果", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def viz_vocab(vocab: dict[str, int], texts: list[str], fname: str) -> None:
    counter: Counter[str] = Counter()
    for t in texts[:5000]:
        counter.update(tokenize(t))
    freqs = np.array(sorted(counter.values(), reverse=True))[:5000]
    rank = np.arange(1, len(freqs) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.loglog(rank, freqs, color=palette(2, "primary")[0], linewidth=1.8)
    ax.set_xlabel("词频排名 (log)")
    ax.set_ylabel("出现次数 (log)")
    ax.set_title("Zipf 分布（词频-排名）")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    cum = np.cumsum(freqs) / freqs.sum()
    ax.plot(rank, cum * 100, color=palette(2, "primary")[1], linewidth=2)
    for thresh in (50, 80, 95):
        idx = int(np.searchsorted(cum, thresh / 100))
        ax.axhline(thresh, color="#9CA3AF", linestyle="--", linewidth=0.8)
        ax.text(idx, thresh + 1, f"前 {idx} 词覆盖 {thresh}%", fontsize=9, color="#1F2937")
    ax.set_xlabel("词表大小")
    ax.set_ylabel("覆盖率 (%)")
    ax.set_xscale("log")
    ax.set_title(f"词表覆盖率（实际词表大小：{len(vocab)}）")

    fig.suptitle("词表构建分析", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def viz_split(splits: dict[str, pd.DataFrame], names: list[str], fname: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    sizes = [len(splits[k]) for k in ("train", "val", "test")]
    ax.bar(["训练集", "验证集", "测试集"], sizes,
           color=palette(3, "primary"), edgecolor="white")
    for i, v in enumerate(sizes):
        ax.text(i, v + max(sizes) * 0.01, str(v), ha="center", fontsize=10)
    ax.set_ylabel("样本数"); ax.set_title("各 Split 样本量")

    ax = axes[1]
    width = 0.27
    x = np.arange(len(names))
    for i, (split, name) in enumerate([("train", "训练集"), ("val", "验证集"), ("test", "测试集")]):
        cnt = splits[split]["label"].value_counts().sort_index()
        vals = [cnt.get(j, 0) for j in range(len(names))]
        ax.bar(x + (i - 1) * width, vals, width, label=name,
               color=palette(3, "primary")[i], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel("样本数"); ax.set_title("各类别在 Split 内的分布")
    ax.legend()
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def main() -> None:
    set_seed(SEED)
    apply_style()

    # ===== 二分类 =====
    with timed("二分类：清洗 + 切分"):
        df_bin = load_binary()
        df_bin["text"] = df_bin["text"].apply(clean_text)
        df_bin = df_bin[df_bin["text"].str.len() >= 2].reset_index(drop=True)
        train, val, test = split_dataframe(
            df_bin,
            train_size=CFG.data.binary_train_size,
            val_size=CFG.data.binary_val_size,
            test_size=CFG.data.binary_test_size,
            seed=SEED,
        )
        splits_bin = {"train": train, "val": val, "test": test}
        save_processed(splits_bin, PROCESSED_DIR, "binary")

    with timed("二分类：构建词表"):
        vocab_bin = build_vocab(train["text"].tolist(),
                                min_freq=CFG.data.min_freq,
                                max_size=CFG.data.vocab_size_cap)
        dump_json(vocab_bin, PROCESSED_DIR / "vocab_binary.json")
        log.info(f"词表大小：{len(vocab_bin)}")

    with timed("二分类：可视化预处理 / 词表 / 切分"):
        viz_preprocess_effect(df_bin, "preprocess_01_binary_effect")
        viz_vocab(vocab_bin, train["text"].tolist(), "preprocess_02_binary_vocab")
        viz_split(splits_bin, list(CFG.binary_label_names_zh),
                  "preprocess_03_binary_split", "二分类·训练/验证/测试切分")

    # ===== 六情绪 =====
    with timed("六情绪：清洗 + 切分"):
        em = load_emotion()
        for k, v in em.items():
            em[k] = v.assign(text=v["text"].apply(clean_text))
            em[k] = em[k][em[k]["text"].str.len() >= 2].reset_index(drop=True)

        df_emo_full = pd.concat([em["train"], em["val"], em["test"]], ignore_index=True)
        train, val, test = split_dataframe(
            df_emo_full,
            train_size=min(CFG.data.emotion_train_size, len(df_emo_full) - 3000),
            val_size=CFG.data.emotion_val_size,
            test_size=CFG.data.emotion_test_size,
            seed=SEED,
        )
        splits_emo = {"train": train, "val": val, "test": test}
        save_processed(splits_emo, PROCESSED_DIR, "emotion")

    with timed("六情绪：构建词表"):
        vocab_emo = build_vocab(train["text"].tolist(),
                                min_freq=CFG.data.min_freq,
                                max_size=CFG.data.vocab_size_cap)
        dump_json(vocab_emo, PROCESSED_DIR / "vocab_emotion.json")
        log.info(f"词表大小：{len(vocab_emo)}")

    with timed("六情绪：可视化"):
        viz_preprocess_effect(df_emo_full, "preprocess_04_emotion_effect")
        viz_vocab(vocab_emo, train["text"].tolist(), "preprocess_05_emotion_vocab")
        viz_split(splits_emo, list(CFG.emotion_label_names_zh),
                  "preprocess_06_emotion_split", "六情绪·训练/验证/测试切分")

    # 概览统计落盘
    summary = {
        "binary": {k: int(len(v)) for k, v in splits_bin.items()} | {
            "vocab_size": len(vocab_bin),
        },
        "emotion": {k: int(len(v)) for k, v in splits_emo.items()} | {
            "vocab_size": len(vocab_emo),
        },
    }
    dump_json(summary, PROCESSED_DIR / "summary.json")
    log.info("✓ 预处理完成")
    log.info(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
