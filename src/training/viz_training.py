"""训练过程可视化：历史曲线、混淆矩阵、ROC、PR、分类报告。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc as sk_auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.viz import palette, save_fig


def plot_history(history: dict, model_label: str, out_dir: Path, fname: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "-o", color=palette(2, "primary")[0],
            label="训练 Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "-s", color=palette(2, "primary")[1],
            label="验证 Loss", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("损失曲线"); ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "-o", color=palette(2, "primary")[0],
            label="训练 Accuracy", linewidth=2)
    ax.plot(epochs, history["val_acc"], "-s", color=palette(2, "primary")[1],
            label="验证 Accuracy", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("准确率曲线"); ax.legend()

    ax = axes[2]
    ax.plot(epochs, history["val_f1"], "-o", color=palette(3, "primary")[2],
            label="验证 F1", linewidth=2)
    if history.get("val_auc") and any(history["val_auc"]):
        ax.plot(epochs, history["val_auc"], "-s", color=palette(4, "primary")[3],
                label="验证 AUC", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("指标值")
    ax.set_title("F1 / AUC 曲线"); ax.legend()

    fig.suptitle(f"{model_label}·训练历史", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, fname, out_dir)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
    model_label: str, out_dir: Path, fname: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar=False, annot_kws={"fontsize": 10})
    axes[0].set_title("混淆矩阵（样本数）")
    axes[0].set_xlabel("预测标签"); axes[0].set_ylabel("真实标签")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar=False, annot_kws={"fontsize": 10})
    axes[1].set_title("混淆矩阵（按行归一化）")
    axes[1].set_xlabel("预测标签"); axes[1].set_ylabel("真实标签")

    fig.suptitle(f"{model_label}·测试集混淆矩阵", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, fname, out_dir)


def plot_roc_pr(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str],
    model_label: str, out_dir: Path, fname: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = palette(len(class_names), "primary")
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = sk_auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=colors[1], linewidth=2,
                     label=f"ROC (AUC={roc_auc:.3f})")
        axes[0].plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1)
        axes[0].set_xlabel("假正率 FPR"); axes[0].set_ylabel("真正率 TPR")
        axes[0].set_title("ROC 曲线"); axes[0].legend(loc="lower right")

        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        pr_auc = sk_auc(recall, precision)
        axes[1].plot(recall, precision, color=colors[1], linewidth=2,
                     label=f"PR (AUC={pr_auc:.3f})")
        axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
        axes[1].set_title("PR 曲线"); axes[1].legend(loc="lower left")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = sk_auc(fpr, tpr)
            axes[0].plot(fpr, tpr, color=colors[i % len(colors)], linewidth=1.8,
                         label=f"{name} (AUC={roc_auc:.2f})")
        axes[0].plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1)
        axes[0].set_xlabel("假正率 FPR"); axes[0].set_ylabel("真正率 TPR")
        axes[0].set_title("ROC 曲线（各类 One-vs-Rest）"); axes[0].legend(loc="lower right", fontsize=9)

        for i, name in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            pr_auc = sk_auc(recall, precision)
            axes[1].plot(recall, precision, color=colors[i % len(colors)], linewidth=1.8,
                         label=f"{name} (AUC={pr_auc:.2f})")
        axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
        axes[1].set_title("PR 曲线（各类 One-vs-Rest）"); axes[1].legend(loc="lower left", fontsize=9)

    fig.suptitle(f"{model_label}·ROC 与 PR", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, fname, out_dir)


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], path: Path,
) -> None:
    rep = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rep).T.to_csv(path, encoding="utf-8-sig")
