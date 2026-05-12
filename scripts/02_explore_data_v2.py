"""脚本 02：数据集探索性分析 + 可视化（独立图表、科研风格）。

产出：outputs/figures/eda_*.png（每个指标独立一张图）
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

log = get_logger("02_eda_v2")


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


def viz_dataset_size(df_bin: pd.DataFrame, df_emo: pd.DataFrame) -> None:
    """数据集规模对比（独立图）"""
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ["样本总数", "唯一文本数", "平均字符数", "中位字符数"]
    bin_vals = [len(df_bin), df_bin["text"].nunique(),
                df_bin["text"].str.len().mean(), df_bin["text"].str.len().median()]
    emo_vals = [len(df_emo), df_emo["text"].nunique(),
                df_emo["text"].str.len().mean(), df_emo["text"].str.len().median()]

    x = np.arange(len(metrics))
    w = 0.35
    colors = palette(2, "sci")
    ax.bar(x - w/2, bin_vals, w, label="二分类微博", color=colors[0], edgecolor="white", linewidth=1.5)
    ax.bar(x + w/2, emo_vals, w, label="六情绪微博", color=colors[1], edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_yscale("log")
    ax.set_ylabel("数值（对数刻度）", fontsize=13)
    ax.set_title("数据集规模与文本统计", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for i, (b, e) in enumerate(zip(bin_vals, emo_vals)):
        ax.text(i - w/2, b * 1.15, f"{b:.0f}", ha="center", fontsize=10, color="#1A202C")
        ax.text(i + w/2, e * 1.15, f"{e:.0f}", ha="center", fontsize=10, color="#1A202C")

    save_fig(fig, "eda_01_dataset_size", FIGURES_DIR)


def viz_data_quality(df_bin: pd.DataFrame, df_emo: pd.DataFrame) -> None:
    """数据质量指标（独立图）"""
    fig, ax = plt.subplots(figsize=(7, 6))
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
    w = 0.35
    colors = palette(2, "sci")

    ax.bar(x - w/2, vals_bin, w, label="二分类微博", color=colors[0], edgecolor="white", linewidth=1.5)
    ax.bar(x + w/2, vals_emo, w, label="六情绪微博", color=colors[1], edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比例", fontsize=13)
    ax.set_title("数据质量指标", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    for i, (b, e) in enumerate(zip(vals_bin, vals_emo)):
        ax.text(i - w/2, b + 0.03, f"{b*100:.1f}%", ha="center", fontsize=10, color="#1A202C")
        ax.text(i + w/2, e + 0.03, f"{e*100:.1f}%", ha="center", fontsize=10, color="#1A202C")

    save_fig(fig, "eda_02_data_quality", FIGURES_DIR)


def viz_binary_length_dist(df_bin: pd.DataFrame) -> None:
    """二分类文本长度分布（独立图）"""
    lens_bin = df_bin["text"].str.len()
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = palette(2, "sci")
    ax.hist(
        [lens_bin[df_bin["label"] == 0].clip(upper=200),
         lens_bin[df_bin["label"] == 1].clip(upper=200)],
        bins=50, stacked=False, alpha=0.7,
        label=list(CFG.binary_label_names_zh),
        color=colors, edgecolor="white", linewidth=0.8,
    )
    ax.set_xlabel("字符长度", fontsize=13)
    ax.set_ylabel("样本数", fontsize=13)
    ax.set_title("二分类·文本长度分布", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    save_fig(fig, "eda_03_binary_length", FIGURES_DIR)


def viz_emotion_length_box(df_emo: pd.DataFrame) -> None:
    """六情绪文本长度箱型图（独立图）"""
    lens_emo = df_emo["text"].str.len()
    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [lens_emo[df_emo["label"] == i].clip(upper=300)
                for i in sorted(df_emo["label"].unique())]
    box_names = [CFG.emotion_label_names_zh[i]
                 for i in sorted(df_emo["label"].unique())]

    colors = palette(len(box_data), "sci")
    bp = ax.boxplot(box_data, tick_labels=box_names, patch_artist=True,
                    widths=0.6, medianprops={"color": "#1A202C", "linewidth": 2})

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#2D3748")
        patch.set_linewidth(1.2)

    ax.set_ylabel("字符长度", fontsize=13)
    ax.set_title("六情绪·各类别文本长度分布", fontsize=16, fontweight="600", pad=15)
    ax.grid(axis="y", alpha=0.3)

    save_fig(fig, "eda_04_emotion_length", FIGURES_DIR)


def viz_token_length_dist(df_bin: pd.DataFrame, sample_n: int = 5000) -> None:
    """分词 Token 数分布（独立图）"""
    sample = df_bin.sample(min(sample_n, len(df_bin)), random_state=42)
    pos_lens, neg_lens = [], []

    for _, row in sample.iterrows():
        toks = tokenize(row["text"])
        if row["label"] == 1:
            neg_lens.append(len(toks))
        else:
            pos_lens.append(len(toks))

    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.arange(0, 50, 2)
    colors = palette(2, "sci")

    ax.hist(
        [np.clip(pos_lens, 0, 48), np.clip(neg_lens, 0, 48)],
        bins=bins, label=list(CFG.binary_label_names_zh),
        color=colors, edgecolor="white", alpha=0.7, linewidth=0.8,
    )
    ax.set_xlabel("分词后 Token 数", fontsize=13)
    ax.set_ylabel("样本数", fontsize=13)
    ax.set_title("分词长度分布（jieba）", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    save_fig(fig, "eda_05_token_length", FIGURES_DIR)


def viz_token_freq_compare(df_bin: pd.DataFrame, sample_n: int = 5000) -> None:
    """高频词对比（独立图）"""
    sample = df_bin.sample(min(sample_n, len(df_bin)), random_state=42)
    pos_tokens, neg_tokens = [], []

    for _, row in sample.iterrows():
        toks = tokenize(row["text"])
        if row["label"] == 1:
            neg_tokens.extend(toks)
        else:
            pos_tokens.extend(toks)

    fig, ax = plt.subplots(figsize=(11, 7))
    stop = set("的了是我也都很就不和在还有也但被把没没有都也已也我们你们他们对啊吗哦呢吧"
               "一个这个那个这些那些这样那样真的就是".split())
    counter_pos = Counter(w for w in pos_tokens if w not in stop and len(w) > 1)
    counter_neg = Counter(w for w in neg_tokens if w not in stop and len(w) > 1)
    common_pos = [w for w, _ in counter_pos.most_common(12)]
    common_neg = [w for w, _ in counter_neg.most_common(12)]
    n = min(len(common_pos), len(common_neg))
    common_pos, common_neg = common_pos[:n], common_neg[:n]

    y_pos = np.arange(n)
    counts_pos = [counter_pos[w] for w in common_pos]
    counts_neg = [counter_neg[w] for w in common_neg]
    width = 0.4
    colors = palette(2, "sci")

    ax.barh(y_pos - width / 2, counts_pos, width, label=CFG.binary_label_names_zh[0],
            color=colors[0], edgecolor="white", linewidth=1)
    ax.barh(y_pos + width / 2, counts_neg, width, label=CFG.binary_label_names_zh[1],
            color=colors[1], edgecolor="white", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{p}   |   {n}" for p, n in zip(common_pos, common_neg)],
                       fontsize=11)
    ax.set_xlabel("出现次数", fontsize=13)
    ax.set_title("二分类·Top 高频词对比（左：非抑郁  ｜  右：抑郁）",
                 fontsize=15, fontweight="600", pad=15)
    ax.legend(loc="lower right", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    save_fig(fig, "eda_06_token_freq_compare", FIGURES_DIR)


def viz_wordcloud_single(rows: pd.DataFrame, font_path: str,
                          title: str, cmap: str, fname: str) -> None:
    """单类别词云（独立图）"""
    stop = set("的了是我也都很就不和在还有但是这个那个真的就是".split())
    toks = []
    for t in rows["text"].sample(min(len(rows), 3000), random_state=42):
        for tok in tokenize(t):
            if len(tok) > 1 and tok not in stop and not tok.isdigit():
                toks.append(tok)
    text = " ".join(toks)
    if not text.strip():
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    wc = WordCloud(
        font_path=font_path, width=1000, height=700,
        background_color="white", colormap=cmap,
        max_words=150, prefer_horizontal=0.9,
        relative_scaling=0.5,
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="600", pad=15)
    save_fig(fig, fname, FIGURES_DIR)


def viz_wordclouds(df_bin: pd.DataFrame, df_emo: pd.DataFrame) -> None:
    """所有词云（每个类别独立图）"""
    font_path = _find_cn_font_path()
    if font_path is None:
        log.warning("找不到中文字体，跳过词云")
        return

    # 二分类词云
    viz_wordcloud_single(df_bin[df_bin["label"] == 0], font_path,
                          f"{CFG.binary_label_names_zh[0]}·高频词云",
                          "Blues", "eda_07_wordcloud_normal")
    viz_wordcloud_single(df_bin[df_bin["label"] == 1], font_path,
                          f"{CFG.binary_label_names_zh[1]}·高频词云",
                          "Reds", "eda_08_wordcloud_depressive")

    # 六情绪词云（每个情绪单独一张图）
    cmaps = ["YlOrBr", "Greens", "Reds", "Blues", "Purples", "PuBu"]
    for i, lab in enumerate(sorted(df_emo["label"].unique())):
        viz_wordcloud_single(
            df_emo[df_emo["label"] == lab], font_path,
            f"{CFG.emotion_label_names_zh[lab]}·高频词云",
            cmaps[i % len(cmaps)], f"eda_09_wordcloud_emo_{i}",
        )


def viz_label_bar(df: pd.DataFrame, names: list[str], title: str, fname: str) -> None:
    """标签分布柱状图（独立图）"""
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = palette(len(counts), "sci")

    bars = ax.bar(
        [names[i] for i in counts.index], counts.values,
        color=colors, edgecolor="white", linewidth=1.5, width=0.6,
    )
    ax.set_ylabel("样本数", fontsize=13)
    ax.set_title(title, fontsize=16, fontweight="600", pad=15)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(counts) * 0.01,
                f"{int(h)}", ha="center", fontsize=11, color="#1A202C")

    save_fig(fig, fname, FIGURES_DIR)


def viz_label_pie(df: pd.DataFrame, names: list[str], title: str, fname: str) -> None:
    """标签分布饼图（独立图）"""
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = palette(len(counts), "sci")

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=[names[i] for i in counts.index],
        autopct="%1.1f%%",
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
        startangle=90,
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(13)

    ax.set_title(title, fontsize=16, fontweight="600", pad=20)
    save_fig(fig, fname, FIGURES_DIR)


def main() -> None:
    apply_style()
    df_bin = load_binary()
    em = load_emotion()
    df_emo_full = pd.concat([em["train"], em["val"], em["test"]], ignore_index=True)
    log.info(f"二分类：{len(df_bin)}，六情绪：{len(df_emo_full)}")

    with timed("数据集规模"):
        viz_dataset_size(df_bin, df_emo_full)
    with timed("数据质量"):
        viz_data_quality(df_bin, df_emo_full)
    with timed("二分类文本长度"):
        viz_binary_length_dist(df_bin)
    with timed("六情绪文本长度"):
        viz_emotion_length_box(df_emo_full)
    with timed("分词长度分布"):
        viz_token_length_dist(df_bin)
    with timed("高频词对比"):
        viz_token_freq_compare(df_bin)
    with timed("词云"):
        viz_wordclouds(df_bin, df_emo_full)
    with timed("二分类标签分布（柱状图）"):
        viz_label_bar(
            df_bin, list(CFG.binary_label_names_zh),
            "二分类·标签分布", "eda_15_binary_label_bar",
        )
    with timed("二分类标签分布（饼图）"):
        viz_label_pie(
            df_bin, list(CFG.binary_label_names_zh),
            "二分类·标签占比", "eda_16_binary_label_pie",
        )
    with timed("六情绪标签分布（柱状图）"):
        viz_label_bar(
            df_emo_full, list(CFG.emotion_label_names_zh),
            "六情绪·标签分布", "eda_17_emotion_label_bar",
        )
    with timed("六情绪标签分布（饼图）"):
        viz_label_pie(
            df_emo_full, list(CFG.emotion_label_names_zh),
            "六情绪·标签占比", "eda_18_emotion_label_pie",
        )

    log.info(f"✓ EDA 完成，图存于：{FIGURES_DIR}")


if __name__ == "__main__":
    main()

