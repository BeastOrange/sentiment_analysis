"""脚本 08：六情绪分析。

复用 BiLSTM-Attention 在六情绪数据集上训练（快），分析：
1. 各情绪的预测性能
2. 二分类模型预测为"抑郁倾向"的样本，在六情绪上的细粒度分布
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.config import CFG, FIGURES_DIR, LOGS_DIR, MODELS_DIR, PROCESSED_DIR, SEED, get_device
from src.data.datasets import df_to_indexed
from src.data.preprocess import build_vocab
from src.models.bilstm import BiLSTMAttention
from src.training.trainer import evaluate, train_loop
from src.training.viz_training import (
    plot_confusion_matrix,
    plot_history,
    plot_roc_pr,
    save_classification_report,
)
from src.utils import dump_json, get_logger, load_json, set_seed, timed
from src.viz import apply_style, palette, save_fig

log = get_logger("08_emotion")

# 各情绪与"抑郁倾向"的关联强度（基于心理学认知）
DEPRESSION_RISK = {
    0: 0.2,  # 中性 - 低
    1: 0.0,  # 高兴 - 无
    2: 0.5,  # 愤怒 - 中
    3: 1.0,  # 悲伤 - 高
    4: 0.7,  # 恐惧 - 较高
    5: 0.1,  # 惊奇 - 低
}


def viz_class_performance(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], fname: str,
) -> None:
    from sklearn.metrics import classification_report
    rep = classification_report(y_true, y_pred, target_names=class_names,
                                 output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T.loc[class_names, ["precision", "recall", "f1-score", "support"]]
    df.columns = ["精确率", "召回率", "F1", "样本数"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    x = np.arange(len(class_names))
    width = 0.25
    colors = palette(3, "primary")
    for i, col in enumerate(["精确率", "召回率", "F1"]):
        ax.bar(x + (i - 1) * width, df[col].values, width, label=col,
               color=colors[i], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylabel("分数"); ax.set_ylim(0, 1.05)
    ax.set_title("各情绪类别预测性能")
    ax.legend()

    ax = axes[1]
    sns.heatmap(df[["精确率", "召回率", "F1"]], annot=True, fmt=".2f", cmap="YlGn",
                yticklabels=class_names, ax=ax, vmin=0, vmax=1,
                cbar_kws={"label": "分数"})
    ax.set_title("分类性能热力图")
    fig.suptitle("六情绪分类性能", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def viz_depression_risk(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
    class_names: list[str], fname: str,
) -> None:
    """情绪 → 抑郁风险分布。"""
    risk_scores = np.array([DEPRESSION_RISK[i] for i in range(len(class_names))])
    sample_risk = (probs * risk_scores[None, :]).sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    ax = axes[0]
    df_dist = pd.DataFrame({
        "情绪": [class_names[i] for i in y_true],
        "抑郁风险": sample_risk,
    })
    order = class_names
    sns.boxplot(data=df_dist, x="情绪", y="抑郁风险", ax=ax, order=order,
                hue="情绪", legend=False, palette=palette(len(class_names), "primary"))
    ax.set_title("各真实情绪类别的抑郁风险分布")
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    avg_risk = []
    counts = []
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            avg_risk.append(sample_risk[mask].mean())
            counts.append(int(mask.sum()))
        else:
            avg_risk.append(0)
            counts.append(0)
    colors = palette(len(class_names), "primary")
    bars = ax.bar(class_names, avg_risk, color=colors, edgecolor="white")
    for b, v, c in zip(bars, avg_risk, counts):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                f"{v:.2f}\n(n={c})", ha="center", fontsize=9)
    ax.set_ylabel("平均抑郁风险分数")
    ax.set_ylim(0, 1.1)
    ax.set_title("各情绪类别·平均抑郁风险（基于心理学映射）")
    fig.suptitle("情绪 → 抑郁风险映射分析", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def viz_emotion_correlations(
    test_df: pd.DataFrame, probs: np.ndarray, class_names: list[str], fname: str,
) -> None:
    """对每个样本：六情绪 softmax 概率 → 累计抑郁风险加权 → 与真实标签对应关系。"""
    risk_scores = np.array([DEPRESSION_RISK[i] for i in range(len(class_names))])
    weighted = probs * risk_scores[None, :]
    sample_risk = weighted.sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    ax = axes[0]
    high_risk_idx = np.argsort(-sample_risk)[:200]
    contrib = weighted[high_risk_idx].mean(axis=0)
    bars = ax.bar(class_names, contrib, color=palette(len(class_names), "primary"),
                   edgecolor="white")
    for b, v in zip(bars, contrib):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylabel("平均贡献")
    ax.set_title("Top 200 高抑郁风险样本的情绪贡献分解")

    ax = axes[1]
    hist_bins = np.linspace(0, 1, 21)
    counts_by_emo = []
    for i in range(len(class_names)):
        mask = test_df["label"].to_numpy() == i
        h, _ = np.histogram(sample_risk[mask], bins=hist_bins)
        counts_by_emo.append(h)
    counts_arr = np.array(counts_by_emo)
    norm = counts_arr / counts_arr.sum(axis=0, keepdims=True).clip(min=1)
    sns.heatmap(norm, ax=ax, cmap="YlOrRd", cbar_kws={"label": "占比"},
                xticklabels=[f"{b:.1f}" for b in hist_bins[:-1]],
                yticklabels=class_names)
    ax.set_xlabel("抑郁风险分数（分箱）")
    ax.set_ylabel("真实情绪类别")
    ax.set_title("不同情绪类别在抑郁风险分箱中的分布")
    fig.suptitle("情绪与抑郁风险关联可视化", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def main() -> None:
    set_seed(SEED)
    apply_style()
    device = get_device()
    log.info(f"设备：{device}")

    train_df = pd.read_parquet(PROCESSED_DIR / "emotion_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "emotion_val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "emotion_test.parquet")
    num_classes = len(CFG.emotion_label_names_zh)
    class_names = list(CFG.emotion_label_names_zh)
    log.info(f"训练 {len(train_df)} | 验证 {len(val_df)} | 测试 {len(test_df)} | "
             f"类别 {num_classes}")

    # 类别失衡：构建权重
    class_counts = train_df["label"].value_counts().sort_index().values
    weights = torch.tensor(
        (class_counts.sum() / (len(class_counts) * class_counts)).astype("float32")
    ).to(device)
    log.info(f"类别权重：{weights.cpu().numpy()}")

    vocab = load_json(PROCESSED_DIR / "vocab_emotion.json")

    with timed("构建 Dataset / DataLoader"):
        train_ds = df_to_indexed(train_df, vocab, CFG.data.max_len)
        val_ds = df_to_indexed(val_df, vocab, CFG.data.max_len)
        test_ds = df_to_indexed(test_df, vocab, CFG.data.max_len)
        train_loader = DataLoader(train_ds, batch_size=CFG.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CFG.train.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=CFG.train.batch_size, shuffle=False)

    model = BiLSTMAttention(
        vocab_size=len(vocab),
        embed_dim=CFG.model.embed_dim,
        hidden_dim=CFG.model.lstm_hidden,
        num_layers=CFG.model.lstm_layers,
        num_classes=num_classes,
        dropout=CFG.model.dropout,
    ).to(device)
    log.info(f"参数量：{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    save_path = MODELS_DIR / "bilstm_emotion.pt"
    history = train_loop(
        model, train_loader, val_loader, device,
        num_classes=num_classes,
        epochs=CFG.train.epochs_baseline + 2,
        lr=CFG.train.lr_baseline,
        weight_decay=CFG.train.weight_decay,
        grad_clip=CFG.train.grad_clip,
        class_weights=weights,
        model_label="BiLSTM-Emo",
        save_path=save_path,
        early_stop=CFG.train.early_stop_patience,
    )
    dump_json(history.to_dict(), LOGS_DIR / "bilstm_emotion_history.json")

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    metrics, probs, labels = evaluate(model, test_loader, device, num_classes,
                                       return_probs=True)
    log.info(f"测试集：{metrics}")
    dump_json(metrics, LOGS_DIR / "bilstm_emotion_test.json")
    np.savez(LOGS_DIR / "bilstm_emotion_probs.npz", probs=probs, labels=labels)

    plot_history(history.to_dict(), "BiLSTM-Attention（六情绪）",
                 FIGURES_DIR, "emotion_01_history")
    plot_confusion_matrix(
        labels, probs.argmax(-1), class_names,
        "六情绪 BiLSTM-Attention", FIGURES_DIR, "emotion_02_cm",
    )
    plot_roc_pr(
        labels, probs, class_names,
        "六情绪 BiLSTM-Attention", FIGURES_DIR, "emotion_03_roc",
    )
    save_classification_report(
        labels, probs.argmax(-1), class_names,
        LOGS_DIR / "bilstm_emotion_report.csv",
    )
    viz_class_performance(labels, probs.argmax(-1), class_names, "emotion_04_class_perf")
    viz_depression_risk(labels, probs.argmax(-1), probs, class_names,
                         "emotion_05_depression_risk")
    viz_emotion_correlations(test_df, probs, class_names, "emotion_06_correlations")

    log.info("✓ 六情绪分析完成")


if __name__ == "__main__":
    main()
