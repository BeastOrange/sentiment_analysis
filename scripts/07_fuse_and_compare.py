"""脚本 07：模型融合 + 全面对比。

读取三个模型在测试集上的 probs，做加权融合（搜索最优权重），并产出对比图。
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc as sk_auc,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CFG, FIGURES_DIR, LOGS_DIR
from src.utils import dump_json, get_logger, load_json
from src.viz import apply_style, palette, save_fig
from src.training.viz_training import plot_confusion_matrix, plot_roc_pr

log = get_logger("07_fuse")

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


def search_best_weights(probs_dict: dict[str, dict], step: float = 0.1) -> tuple[dict, dict]:
    """网格搜索三模型加权融合的最优权重（按 F1 最大）。"""
    names = list(probs_dict.keys())
    labels = probs_dict[names[0]]["labels"]
    n = len(names)

    grid = np.arange(0.0, 1.0 + step, step)
    best = {"f1": -1.0, "weights": None, "metrics": None}
    for combo in product(grid, repeat=n):
        s = sum(combo)
        if s < 1e-6:
            continue
        w = np.array(combo) / s
        fused = sum(wi * probs_dict[ni]["probs"] for wi, ni in zip(w, names))
        preds = fused.argmax(-1)
        m = metric_pack(labels, preds, fused)
        if m["f1"] > best["f1"]:
            best.update({"f1": m["f1"], "weights": dict(zip(names, w.tolist())),
                          "metrics": m, "fused_probs": fused})
    return best["weights"], best


def viz_compare_metrics(results: dict[str, dict], fname: str) -> None:
    metrics_to_show = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_zh = ["准确率", "精确率", "召回率", "F1 分数", "AUC"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    ax = axes[0]
    x = np.arange(len(metrics_to_show))
    width = 0.8 / len(results)
    colors = palette(len(results), "primary")
    for i, (name, m) in enumerate(results.items()):
        vals = [m[k] for k in metrics_to_show]
        ax.bar(x + (i - (len(results)-1)/2) * width, vals, width, label=name,
               color=colors[i], edgecolor="white")
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - (len(results)-1)/2) * width, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8, rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(metric_zh)
    ax.set_ylim(0.92, 1.005)
    ax.set_ylabel("分数")
    ax.set_title("各模型测试集指标对比")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    df_m = pd.DataFrame(results).T[metrics_to_show]
    df_m.columns = metric_zh
    sns.heatmap(df_m, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.92, vmax=1.0, ax=ax, cbar_kws={"label": "分数"})
    ax.set_title("指标热力图")
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("模型综合性能对比", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def viz_roc_overlay(probs_dict: dict[str, dict], fused_probs: np.ndarray,
                    labels: np.ndarray, fname: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = palette(len(probs_dict) + 1, "primary")
    sources = {**{n: d["probs"] for n, d in probs_dict.items()}, "Ensemble (融合)": fused_probs}

    for (name, probs), c in zip(sources.items(), colors):
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        auc_v = sk_auc(fpr, tpr)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_v:.3f})", color=c)
    axes[0].plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1)
    axes[0].set_xlabel("假正率 FPR"); axes[0].set_ylabel("真正率 TPR")
    axes[0].set_title("ROC 曲线对比"); axes[0].legend(loc="lower right", fontsize=9)

    for (name, probs), c in zip(sources.items(), colors):
        prec, rec, _ = precision_recall_curve(labels, probs[:, 1])
        auc_v = sk_auc(rec, prec)
        axes[1].plot(rec, prec, linewidth=2, label=f"{name} (AUC={auc_v:.3f})", color=c)
    axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
    axes[1].set_title("PR 曲线对比"); axes[1].legend(loc="lower left", fontsize=9)
    fig.suptitle("三模型与融合模型 ROC / PR 对比", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def viz_agreement(probs_dict: dict[str, dict], labels: np.ndarray, fname: str) -> None:
    """模型一致性矩阵：预测一致比例 + 各组合下的正确率。"""
    names = list(probs_dict.keys())
    preds = {n: d["probs"].argmax(-1) for n, d in probs_dict.items()}

    n = len(names)
    agree_mat = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            agree_mat[i, j] = (preds[a] == preds[b]).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.heatmap(agree_mat, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=axes[0],
                vmin=0.9, vmax=1.0, cbar_kws={"label": "一致比例"})
    axes[0].set_title("模型间预测一致比例")

    ax = axes[1]
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
    colors_b = palette(len(bars), "primary")
    ax.barh(bars, values, color=colors_b, edgecolor="white")
    for i, v in enumerate(values):
        ax.text(v + max(values)*0.005, i, f"{v} ({v/len(labels)*100:.1f}%)",
                va="center", fontsize=9)
    ax.set_xlabel("测试样本数")
    ax.set_title("模型预测正误分布")
    ax.invert_yaxis()
    fig.suptitle("模型间一致性分析", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def viz_weights_search(weights: dict, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(weights.keys())
    vals = list(weights.values())
    colors_b = palette(len(names), "primary")
    bars = ax.bar(names, vals, color=colors_b, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.2 + 0.05)
    ax.set_ylabel("融合权重")
    ax.set_title("最优融合权重（网格搜索，按验证 F1 最大化）")
    save_fig(fig, fname, FIGURES_DIR)


def main() -> None:
    apply_style()
    probs_dict = load_probs()
    if len(probs_dict) < 2:
        log.error("可用模型不足 2 个，无法做融合")
        return

    labels = list(probs_dict.values())[0]["labels"]

    # 单模型指标
    results = {}
    for name, d in probs_dict.items():
        preds = d["probs"].argmax(-1)
        results[name] = metric_pack(labels, preds, d["probs"])
        log.info(f"{name}：{results[name]}")

    # 加权融合
    log.info("⏱  搜索最优融合权重...")
    weights, best = search_best_weights(probs_dict, step=0.1)
    log.info(f"最优权重：{weights}")
    log.info(f"融合指标：{best['metrics']}")
    results["Ensemble (加权融合)"] = best["metrics"]
    fused_probs = best["fused_probs"]

    dump_json({
        "single": {k: v for k, v in results.items() if "Ensemble" not in k},
        "ensemble": {"weights": weights, "metrics": best["metrics"]},
    }, LOGS_DIR / "ensemble_summary.json")
    np.savez(LOGS_DIR / "ensemble_probs.npz", probs=fused_probs, labels=labels)

    # 可视化
    viz_compare_metrics(results, "compare_01_metrics")
    viz_roc_overlay(probs_dict, fused_probs, labels, "compare_02_roc_pr")
    viz_agreement(probs_dict, labels, "compare_03_agreement")
    viz_weights_search(weights, "compare_04_weights")

    # 融合模型的混淆矩阵
    plot_confusion_matrix(
        labels, fused_probs.argmax(-1), list(CFG.binary_label_names_zh),
        "Ensemble", FIGURES_DIR, "compare_05_ensemble_cm",
    )

    log.info("✓ 模型融合与对比完成")


if __name__ == "__main__":
    main()
