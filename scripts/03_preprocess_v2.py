"""脚本 03：预处理 + 拆分 + 词表构建（独立图表、科研风格）。

产出：
- data/processed/binary_{train,val,test}.parquet
- data/processed/emotion_{train,val,test}.parquet
- data/processed/vocab_binary.json, vocab_emotion.json
- outputs/figures/preprocess_*.png（每个指标独立一张图）
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

log = get_logger("03_preprocess_v2")


def viz_clean_length_compare(df: pd.DataFrame, fname: str, title: str) -> None:
    """清洗前后字符长度对比（独立图）"""
    sample = df.sample(min(2000, len(df)), random_state=SEED)
    raw_lens = sample["text"].str.len()
    cleaned = sample["text"].apply(clean_text)
    clean_lens = cleaned.str.len()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = palette(2, "sci")
    ax.hist([raw_lens.clip(upper=200), clean_lens.clip(upper=200)],
            bins=40, label=["原始", "清洗后"],
            color=colors, edgecolor="white", alpha=0.75, linewidth=0.8)
    ax.set_xlabel("字符数", fontsize=13)
    ax.set_ylabel("样本数", fontsize=13)
    ax.set_title(f"{title}·清洗前后字符长度对比", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def viz_clean_diff(df: pd.DataFrame, fname: str, title: str) -> None:
    """被清洗掉的字符数（独立图）"""
    sample = df.sample(min(2000, len(df)), random_state=SEED)
    raw_lens = sample["text"].str.len()
    clean_lens = sample["text"].apply(clean_text).str.len()
    diff = (raw_lens - clean_lens).clip(lower=0, upper=120)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = palette(1, "sci")
    ax.hist(diff, bins=40, color=colors[0], edgecolor="white", alpha=0.8, linewidth=0.8)
    ax.set_xlabel("被清洗掉的字符数", fontsize=13)
    ax.set_ylabel("样本数", fontsize=13)
    ax.set_title(f"{title}·清洗效果", fontsize=16, fontweight="600", pad=15)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def viz_token_count_dist(df: pd.DataFrame, fname: str, title: str) -> None:
    """分词 Token 数分布（独立图）"""
    sample = df.sample(min(2000, len(df)), random_state=SEED)
    cleaned = sample["text"].apply(clean_text)
    token_counts = cleaned.apply(lambda x: len(tokenize(x)))

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = palette(1, "sci")
    ax.hist(token_counts.clip(upper=80), bins=40,
            color=colors[0], edgecolor="white", alpha=0.8, linewidth=0.8)
    ax.axvline(CFG.data.max_len, color="#DC2626", linestyle="--", linewidth=2.5,
               label=f"截断长度 = {CFG.data.max_len}")
    ax.set_xlabel("Token 数", fontsize=13)
    ax.set_ylabel("样本数", fontsize=13)
    ax.set_title(f"{title}·分词 Token 数分布", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def viz_vocab_zipf(vocab: dict[str, int], texts: list[str], fname: str, title: str) -> None:
    """Zipf 分布（独立图）"""
    counter: Counter[str] = Counter()
    for t in texts[:5000]:
        counter.update(tokenize(t))
    freqs = np.array(sorted(counter.values(), reverse=True))[:5000]
    rank = np.arange(1, len(freqs) + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = palette(1, "sci")
    ax.loglog(rank, freqs, color=colors[0], linewidth=2.5, alpha=0.8)
    ax.set_xlabel("词频排名（对数）", fontsize=13)
    ax.set_ylabel("出现次数（对数）", fontsize=13)
    ax.set_title(f"{title}·Zipf 分布（词频-排名）", fontsize=16, fontweight="600", pad=15)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.8)
    save_fig(fig, fname, FIGURES_DIR)


def viz_vocab_coverage(vocab: dict[str, int], texts: list[str], fname: str, title: str) -> None:
    """词表覆盖率（独立图）"""
    counter: Counter[str] = Counter()
    for t in texts[:5000]:
        counter.update(tokenize(t))
    freqs = np.array(sorted(counter.values(), reverse=True))[:5000]
    rank = np.arange(1, len(freqs) + 1)
    cum = np.cumsum(freqs) / freqs.sum()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = palette(1, "sci")
    ax.plot(rank, cum * 100, color=colors[0], linewidth=2.5, alpha=0.8)

    for thresh in (50, 80, 95):
        idx = int(np.searchsorted(cum, thresh / 100))
        ax.axhline(thresh, color="#9CA3AF", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(idx * 1.2, thresh + 1.5, f"前 {idx} 词覆盖 {thresh}%",
                fontsize=11, color="#1A202C", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", edgecolor="#9CA3AF", alpha=0.8))

    ax.set_xlabel("词表大小", fontsize=13)
    ax.set_ylabel("覆盖率 (%)", fontsize=13)
    ax.set_xscale("log")
    ax.set_title(f"{title}·词表覆盖率（实际词表：{len(vocab)} 词）",
                 fontsize=16, fontweight="600", pad=15)
    ax.grid(True, alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def viz_split_sizes(splits: dict[str, pd.DataFrame], fname: str, title: str) -> None:
    """各 Split 样本量（独立图）"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = [len(splits[k]) for k in ("train", "val", "test")]
    colors = palette(3, "sci")
    bars = ax.bar(["训练集", "验证集", "测试集"], sizes,
                   color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(sizes) * 0.01,
                f"{int(h)}", ha="center", fontsize=12, color="#1A202C")

    ax.set_ylabel("样本数", fontsize=13)
    ax.set_title(f"{title}·数据切分", fontsize=16, fontweight="600", pad=15)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def viz_split_class_dist(splits: dict[str, pd.DataFrame], names: list[str],
                          fname: str, title: str) -> None:
    """各 Split 内类别分布（独立图）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    x = np.arange(len(names))
    colors = palette(3, "sci")

    for i, (split, name) in enumerate([("train", "训练集"), ("val", "验证集"), ("test", "测试集")]):
        cnt = splits[split]["label"].value_counts().sort_index()
        vals = [cnt.get(j, 0) for j in range(len(names))]
        ax.bar(x + (i - 1) * width, vals, width, label=name,
               color=colors[i], edgecolor="white", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("样本数", fontsize=13)
    ax.set_title(f"{title}·各类别在 Split 内的分布", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
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

    with timed("二分类：可视化"):
        viz_clean_length_compare(df_bin, "preprocess_01_binary_clean_length", "二分类")
        viz_clean_diff(df_bin, "preprocess_02_binary_clean_diff", "二分类")
        viz_token_count_dist(df_bin, "preprocess_03_binary_token_count", "二分类")
        viz_vocab_zipf(vocab_bin, train["text"].tolist(),
                       "preprocess_04_binary_vocab_zipf", "二分类")
        viz_vocab_coverage(vocab_bin, train["text"].tolist(),
                           "preprocess_05_binary_vocab_coverage", "二分类")
        viz_split_sizes(splits_bin, "preprocess_06_binary_split_sizes", "二分类")
        viz_split_class_dist(splits_bin, list(CFG.binary_label_names_zh),
                             "preprocess_07_binary_split_dist", "二分类")

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
        viz_clean_length_compare(df_emo_full, "preprocess_08_emotion_clean_length", "六情绪")
        viz_clean_diff(df_emo_full, "preprocess_09_emotion_clean_diff", "六情绪")
        viz_token_count_dist(df_emo_full, "preprocess_10_emotion_token_count", "六情绪")
        viz_vocab_zipf(vocab_emo, train["text"].tolist(),
                       "preprocess_11_emotion_vocab_zipf", "六情绪")
        viz_vocab_coverage(vocab_emo, train["text"].tolist(),
                           "preprocess_12_emotion_vocab_coverage", "六情绪")
        viz_split_sizes(splits_emo, "preprocess_13_emotion_split_sizes", "六情绪")
        viz_split_class_dist(splits_emo, list(CFG.emotion_label_names_zh),
                             "preprocess_14_emotion_split_dist", "六情绪")

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
