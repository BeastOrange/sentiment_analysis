"""脚本 11：从已有日志重新生成融合对比图表（独立、科研风格）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc as sk_auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CFG, FIGURES_DIR, LOGS_DIR
from src.training.viz_training_v2 import plot_confusion_matrix
from src.utils import dump_json, get_logger, load_json, timed
from src.viz import apply_style, palette, save_fig

log = get_logger("11_regen_compare")

MODEL_FILES = {
    "TextCNN": "textcnn_binary_probs.npz",
    "BiLSTM-Attention": "bilstm_binary_probs.npz",
    "BERT": "bert_binary_probs.npz",
}


def load_probs() -> dict[str, dict]:
    out = {}
    for name, fname in MODEL_FILES.items():
        p = LOGS_DIR / fname
        if not p.exists():
            log.warning(f"缺失：{p}，跳过 {name}")
            continue
        d = np.load(p)
        out[name] = {"probs": d["probs"], "labels": d["labels"]}
    return out


def metric_pack(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob[:, 1])),
    }


def viz_compare_metrics_bar(results: dict[str, dict], fname: str) -> None:
    """各模型指标对比（柱状图，独立）"""
    metrics_to_show = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_zh = ["准确率", "精确率", "召回率", "F1 分数", "AUC"]

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(metrics_to_show))
    width = 0.8 / len(results)
    colors = palette(len(results), "sci")

    for i, (name, m) in enumerate(results.items()):
        vals = [m[k] for k in metrics_to_show]
        offset = (i - (len(results)-1)/2) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=colors[i], edgecolor="white", linewidth=1.2)
        for j, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                    f"{v:.3f}", ha="center", fontsize=9, rotation=0, color="#1A202C")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_zh, fontsize=12)
    ax.set_ylim(0.92, 1.005)
    ax.set_ylabel("分数", fontsize=13)
    ax.set_title("各模型测试集指标对比", fontsize=16, fontweight="600", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def viz_compare_metrics_heatmap(results: dict[str, dict], fname: str) -> None:
    """各模型指标热力图（独立）"""
    metrics_to_show = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_zh = ["准确率", "精确率", "召回率", "F1", "AUC"]

    df_m = pd.DataFrame(results).T[metrics_to_show]
    df_m.columns = metric_zh

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_m, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.92, vmax=1.0, ax=ax, cbar_kws={"label": "分数"},
                linewidths=1.5, linecolor="white", annot_kws={"fontsize": 11})
    ax.set_xlabel("", fontsize=13)
    ax.set_ylabel("", fontsize=13)
    ax.set_title("模型性能热力图", fontsize=16, fontweight="600", pad=15)
    save_fig(fig, fname, FIGURES_DIR)


def viz_roc_overlay(probs_dict: dict[str, dict], fused_probs: np.ndarray,
                    labels: np.ndarray, fname: str) -> None:
    """ROC 曲线叠加对比（独立图）"""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = palette(len(probs_dict) + 1, "sci")
    sources = {**{n: d["probs"] for n, d in probs_dict.items()}, "Ensemble (融合)": fused_probs}

    for (name, probs), c in zip(sources.items(), colors):
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        auc_v = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5, label=f"{name} (AUC={auc_v:.3f})", color=c)
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("假正率 FPR", fontsize=13)
    ax.set_ylabel("真正率 TPR", fontsize=13)
    ax.set_title("ROC 曲线对比", fontsize=16, fontweight="600", pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def viz_pr_overlay(probs_dict: dict[str, dict], fused_probs: np.ndarray,
                   labels: np.ndarray, fname: str) -> None:
    """PR 曲线叠加对比（独立图）"""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = palette(len(probs_dict) + 1, "sci")
    sources = {**{n: d["probs"] for n, d in probs_dict.items()}, "Ensemble (融合)": fused_probs}

    for (name, probs), c in zip(sources.items(), colors):
        prec, rec, _ = precision_recall_curve(labels, probs[:, 1])
        auc_v = sk_auc(rec, prec)
        ax.plot(rec, prec, linewidth=2.5, label=f"{name} (AUC={auc_v:.3f})", color=c)
    ax.set_xlabel("召回率 Recall", fontsize=13)
    ax.set_ylabel("精确率 Precision", fontsize=13)
    ax.set_title("PR 曲线对比", fontsize=16, fontweight="600", pad=15)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def viz_agreement_matrix(probs_dict: dict[str, dict], labels: np.ndarray, fname: str) -> None:
    """模型一致性矩阵（独立图）"""
    names = list(probs_dict.keys())
    preds = {n: d["probs"].argmax(-1) for n, d in probs_dict.items()}

    n = len(names)
    agree_mat = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            agree_mat[i, j] = (preds[a] == preds[b]).mean()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(agree_mat, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax,
                vmin=0.9, vmax=1.0, cbar_kws={"label": "一致比例"},
                linewidths=1.5, linecolor="white", annot_kws={"fontsize": 11})
    ax.set_title("模型间预测一致比例", fontsize=16, fontweight="600", pad=15)
    save_fig(fig, fname, FIGURES_DIR)


def viz_agreement_stats(probs_dict: dict[str, dict], labels: np.ndarray, fname: str) -> None:
    """模型预测正误分布（独立图）"""
    names = list(probs_dict.keys())
    preds = {n: d["probs"].argmax(-1) for n, d in probs_dict.items()}

    fig, ax = plt.subplots(figsize=(10, 7))
    err_counts = {n: (preds[n] != labels).sum() for n in names}
    all_correct = np.ones_like(labels, dtype=bool)
    for n in names:
        all_correct &= (preds[n] == labels)
    any_correct = np.zeros_like(labels, dtype=bool)
    for n in names:
        any_correct |= (preds[n] == labels)

    bars = ["三者全对", "至少一对", "三者全错"] + [f"{n} 错误" for n in names]
    values = [
        all_correct.sum(),
        any_correct.sum(),
        (~any_correct).sum(),
    ] + [err_counts[n] for n in names]
    colors_b = palette(len(bars), "sci")
    ax.barh(bars, values, color=colors_b, edgecolor="white", linewidth=1.2)
    for i, v in enumerate(values):
        ax.text(v + max(values)*0.01, i, f"{v} ({v/len(labels)*100:.1f}%)",
                va="center", fontsize=10, color="#1A202C")
    ax.set_xlabel("测试样本数", fontsize=13)
    ax.set_title("模型预测正误分布", fontsize=16, fontweight="600", pad=15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def viz_weights(weights: dict, fname: str) -> None:
    """最优融合权重（独立图）"""
    fig, ax = plt.subplots(figsize=(8, 6))
    names = list(weights.keys())
    vals = list(weights.values())
    colors_b = palette(len(names), "sci")
    bars = ax.bar(names, vals, color=colors_b, edgecolor="white", linewidth=1.5, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=12, color="#1A202C")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("融合权重", fontsize=13)
    ax.set_title("最优融合权重（网格搜索，F1 最大化）", fontsize=16, fontweight="600", pad=15)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, fname, FIGURES_DIR)


def main() -> None:
    apply_style()
    probs_dict = load_probs()
    if len(probs_dict) < 2:
        log.error("可用模型不足 2 个")
        return

    labels = list(probs_dict.values())[0]["labels"]

    # 单模型指标
    results = {}
    for name, d in probs_dict.items():
        preds = d["probs"].argmax(-1)
        results[name] = metric_pack(labels, preds, d["probs"])

    # 加载融合结果
    ensemble_data = load_json(LOGS_DIR / "ensemble_summary.json")
    weights = ensemble_data["ensemble"]["weights"]
    fused_data = np.load(LOGS_DIR / "ensemble_probs.npz")
    fused_probs = fused_data["probs"]
    results["Ensemble (融合)"] = ensemble_data["ensemble"]["metrics"]

    with timed("指标对比柱状图"):
        viz_compare_metrics_bar(results, "compare_metrics_bar")
    with timed("指标热力图"):
        viz_compare_metrics_heatmap(results, "compare_metrics_heatmap")
    with timed("ROC 曲线叠加"):
        viz_roc_overlay(probs_dict, fused_probs, labels, "compare_roc_overlay")
    with timed("PR 曲线叠加"):
        viz_pr_overlay(probs_dict, fused_probs, labels, "compare_pr_overlay")
    with timed("一致性矩阵"):
        viz_agreement_matrix(probs_dict, labels, "compare_agreement_matrix")
    with timed("预测正误分布"):
        viz_agreement_stats(probs_dict, labels, "compare_agreement_stats")
    with timed("融合权重"):
        viz_weights(weights, "compare_fusion_weights")
    with timed("融合模型混淆矩阵"):
        plot_confusion_matrix(
            labels, fused_probs.argmax(-1), list(CFG.binary_label_names_zh),
            "Ensemble", FIGURES_DIR, "compare_ensemble",
        )

    log.info("✓ 融合对比可视化完成")


if __name__ == "__main__":
    main()
