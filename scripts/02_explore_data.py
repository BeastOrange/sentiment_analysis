"""脚本 02：数据集探索性分析 + 可视化（中文图）。

产出：outputs/figures/eda_*.png
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

from src.config import CFG, FIGURES_DIR
from src.data.loader import load_binary, load_emotion
from src.data.preprocess import clean_text, tokenize
from src.utils import get_logger, timed
from src.viz import annotate_bars, apply_style, palette, save_fig

log = get_logger("02_eda")


def _find_cn_font_path() -> str | None:
    for p in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ):
        if Path(p).exists():
            return p
    return None


def viz_label_distribution(df: pd.DataFrame, names: list[str], title: str, fname: str) -> None:
    counts = df["label"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = palette(len(counts), "primary")

    ax = axes[0]
    bars = ax.bar(
        [names[i] for i in counts.index], counts.values, color=colors, edgecolor="white"
    )
    ax.set_title(title + "·样本数量")
    ax.set_ylabel("数量")
    annotate_bars(ax)

    ax = axes[1]
    ax.pie(
        counts.values,
        labels=[names[i] for i in counts.index],
        autopct="%1.1f%%",
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10},
    )
    ax.set_title(title + "·占比")
    fig.suptitle("数据集类别分布", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def viz_text_length(df_bin: pd.DataFrame, df_emo: pd.DataFrame) -> None:
    lens_bin = df_bin["text"].str.len()
    lens_emo = df_emo["text"].str.len()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.hist(
        [lens_bin[df_bin["label"] == 0].clip(upper=200),
         lens_bin[df_bin["label"] == 1].clip(upper=200)],
        bins=40, stacked=False, alpha=0.75,
        label=list(CFG.binary_label_names_zh),
        color=palette(2, "primary"), edgecolor="white",
    )
    ax.set_xlabel("字符长度")
    ax.set_ylabel("频次")
    ax.set_title("二分类·文本长度分布")
    ax.legend()

    ax = axes[1]
    box_data = [lens_emo[df_emo["label"] == i].clip(upper=300) for i in sorted(df_emo["label"].unique())]
    box_names = [CFG.emotion_label_names_zh[i] for i in sorted(df_emo["label"].unique())]
    bp = ax.boxplot(box_data, tick_labels=box_names, patch_artist=True, widths=0.55,
                    medianprops={"color": "#1F2937"})
    for patch, c in zip(bp["boxes"], palette(len(box_data), "primary")):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_ylabel("字符长度")
    ax.set_title("六情绪·各类别文本长度对比")
    save_fig(fig, "eda_02_length", FIGURES_DIR)


def viz_token_stats(df_bin: pd.DataFrame, sample_n: int = 5000) -> None:
    sample = df_bin.sample(min(sample_n, len(df_bin)), random_state=42)
    pos_tokens: list[str] = []
    neg_tokens: list[str] = []
    pos_lens, neg_lens = [], []
    for _, row in sample.iterrows():
        toks = tokenize(row["text"])
        if row["label"] == 1:
            neg_tokens.extend(toks)
            neg_lens.append(len(toks))
        else:
            pos_tokens.extend(toks)
            pos_lens.append(len(toks))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.arange(0, 50, 2)
    ax = axes[0]
    ax.hist(
        [np.clip(pos_lens, 0, 48), np.clip(neg_lens, 0, 48)],
        bins=bins, label=list(CFG.binary_label_names_zh),
        color=palette(2, "primary"), edgecolor="white", alpha=0.8,
    )
    ax.set_xlabel("分词后 Token 数")
    ax.set_ylabel("样本数")
    ax.set_title("分词长度分布（jieba）")
    ax.legend()

    ax2 = axes[1]
    stop = set("的了是我也都很就不和在还有也但被把没没有都也已也我们你们他们对啊吗哦呢吧"
               "一个这个那个这些那些这样那样真的就是".split())
    counter_pos = Counter(w for w in pos_tokens if w not in stop and len(w) > 1)
    counter_neg = Counter(w for w in neg_tokens if w not in stop and len(w) > 1)
    common_pos = [w for w, _ in counter_pos.most_common(10)]
    common_neg = [w for w, _ in counter_neg.most_common(10)]
    n = min(len(common_pos), len(common_neg))
    common_pos, common_neg = common_pos[:n], common_neg[:n]
    y_pos = np.arange(n)
    counts_pos = [counter_pos[w] for w in common_pos]
    counts_neg = [counter_neg[w] for w in common_neg]
    width = 0.4
    ax2.barh(y_pos - width / 2, counts_pos, width, label=CFG.binary_label_names_zh[0],
             color=palette(2, "primary")[0])
    ax2.barh(y_pos + width / 2, counts_neg, width, label=CFG.binary_label_names_zh[1],
             color=palette(2, "primary")[1])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{p}  ←  →  {n}" for p, n in zip(common_pos, common_neg)], fontsize=9)
    ax2.set_xlabel("出现次数")
    ax2.set_title("各类别高频词对比（左为非抑郁、右为抑郁）")
    ax2.legend(loc="lower right")
    ax2.invert_yaxis()
    save_fig(fig, "eda_03_tokens", FIGURES_DIR)


def viz_wordclouds(df_bin: pd.DataFrame, df_emo: pd.DataFrame) -> None:
    font_path = _find_cn_font_path()
    if font_path is None:
        log.warning("找不到中文字体，跳过词云")
        return

    def build_text(rows: pd.DataFrame) -> str:
        toks = []
        stop = set("的了是我也都很就不和在还有但是这个那个真的就是".split())
        for t in rows["text"].sample(min(len(rows), 3000), random_state=42):
            for tok in tokenize(t):
                if len(tok) > 1 and tok not in stop and not tok.isdigit():
                    toks.append(tok)
        return " ".join(toks)

    # 二分类词云
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, name, cmap in [
        (axes[0], 0, CFG.binary_label_names_zh[0], "Blues"),
        (axes[1], 1, CFG.binary_label_names_zh[1], "Reds"),
    ]:
        text = build_text(df_bin[df_bin["label"] == label])
        wc = WordCloud(
            font_path=font_path, width=600, height=400,
            background_color="white", colormap=cmap,
            max_words=120, prefer_horizontal=0.9,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{name}·高频词云", fontsize=13)
    fig.suptitle("二分类语料词云对比", fontsize=15, fontweight="bold", y=1.0)
    save_fig(fig, "eda_04_wordcloud_binary", FIGURES_DIR)

    # 六情绪词云
    labels = sorted(df_emo["label"].unique())
    cols = 3
    rows = (len(labels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows))
    axes = np.atleast_2d(axes).flatten()
    cmaps = ["YlOrBr", "Greens", "Reds", "Blues", "Purples", "PuBu"]
    for i, lab in enumerate(labels):
        text = build_text(df_emo[df_emo["label"] == lab])
        if not text.strip():
            axes[i].axis("off")
            continue
        wc = WordCloud(
            font_path=font_path, width=500, height=380,
            background_color="white", colormap=cmaps[i % len(cmaps)],
            max_words=80, prefer_horizontal=0.9,
        ).generate(text)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(CFG.emotion_label_names_zh[lab], fontsize=12)
    for j in range(len(labels), len(axes)):
        axes[j].axis("off")
    fig.suptitle("六情绪语料词云", fontsize=15, fontweight="bold", y=1.0)
    save_fig(fig, "eda_05_wordcloud_emotion", FIGURES_DIR)


def viz_overview(df_bin: pd.DataFrame, df_emo: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    metrics = ["样本总数", "唯一文本数", "平均字符数", "中位字符数"]
    bin_vals = [len(df_bin), df_bin["text"].nunique(),
                df_bin["text"].str.len().mean(), df_bin["text"].str.len().median()]
    emo_vals = [len(df_emo), df_emo["text"].nunique(),
                df_emo["text"].str.len().mean(), df_emo["text"].str.len().median()]
    x = np.arange(len(metrics))
    w = 0.38
    ax.bar(x - w/2, bin_vals, w, label="二分类微博", color=palette(2, "primary")[0])
    ax.bar(x + w/2, emo_vals, w, label="六情绪微博", color=palette(2, "primary")[1])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yscale("log")
    ax.set_title("数据集规模与文本统计（对数刻度）")
    ax.legend()
    for i, (b, e) in enumerate(zip(bin_vals, emo_vals)):
        ax.text(i - w/2, b * 1.05, f"{b:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, e * 1.05, f"{e:.0f}", ha="center", fontsize=8)

    ax = axes[1]
    pct_dup_bin = 1 - df_bin["text"].nunique() / len(df_bin)
    pct_dup_emo = 1 - df_emo["text"].nunique() / len(df_emo)
    avg_clean_bin = df_bin["text"].sample(2000, random_state=1).apply(
        lambda x: len(clean_text(x)) / max(len(x), 1)
    ).mean()
    avg_clean_emo = df_emo["text"].sample(min(2000, len(df_emo)), random_state=1).apply(
        lambda x: len(clean_text(x)) / max(len(x), 1)
    ).mean()
    cats = ["重复率", "清洗保留率"]
    vals_bin = [pct_dup_bin, avg_clean_bin]
    vals_emo = [pct_dup_emo, avg_clean_emo]
    x = np.arange(len(cats))
    ax.bar(x - w/2, vals_bin, w, label="二分类微博", color=palette(2, "primary")[0])
    ax.bar(x + w/2, vals_emo, w, label="六情绪微博", color=palette(2, "primary")[1])
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.05)
    ax.set_title("数据质量指标")
    for i, (b, e) in enumerate(zip(vals_bin, vals_emo)):
        ax.text(i - w/2, b + 0.02, f"{b*100:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, e + 0.02, f"{e*100:.1f}%", ha="center", fontsize=9)
    ax.legend()
    fig.suptitle("数据集概览", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, "eda_01_overview", FIGURES_DIR)


def main() -> None:
    apply_style()
    df_bin = load_binary()
    em = load_emotion()
    df_emo_full = pd.concat([em["train"], em["val"], em["test"]], ignore_index=True)
    log.info(f"二分类：{len(df_bin)}，六情绪：{len(df_emo_full)}")

    with timed("数据集概览"):
        viz_overview(df_bin, df_emo_full)
    with timed("标签分布（二分类）"):
        viz_label_distribution(
            df_bin, list(CFG.binary_label_names_zh),
            "二分类情感", "eda_06_label_binary",
        )
    with timed("标签分布（六情绪）"):
        viz_label_distribution(
            df_emo_full, list(CFG.emotion_label_names_zh),
            "六情绪", "eda_07_label_emotion",
        )
    with timed("文本长度分布"):
        viz_text_length(df_bin, df_emo_full)
    with timed("分词与高频词"):
        viz_token_stats(df_bin)
    with timed("词云"):
        viz_wordclouds(df_bin, df_emo_full)

    log.info(f"✓ EDA 完成，图存于：{FIGURES_DIR}")


if __name__ == "__main__":
    main()
