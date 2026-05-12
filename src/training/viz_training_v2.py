"""训练过程可视化：独立图表、科研风格。

每个指标独立一张图，便于论文使用。
"""
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


def plot_train_loss(history: dict, model_label: str, out_dir: Path, fname: str) -> None:
    """训练损失曲线（独立图）"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = palette(2, "sci")

    ax.plot(epochs, history["train_loss"], "-o", color=colors[0],
            label="训练 Loss", linewidth=2.5, markersize=6)
    ax.plot(epochs, history["val_loss"], "-s", color=colors[1],
            label="验证 Loss", linewidth=2.5, markersize=6)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(f"{model_label}·损失曲线", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(alpha=0.3)
    save_fig(fig, fname, out_dir)


def plot_train_acc(history: dict, model_label: str, out_dir: Path, fname: str) -> None:
    """训练准确率曲线（独立图）"""
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = palette(2, "sci")

    ax.plot(epochs, history["train_acc"], "-o", color=colors[0],
            label="训练 Accuracy", linewidth=2.5, markersize=6)
    ax.plot(epochs, history["val_acc"], "-s", color=colors[1],
            label="验证 Accuracy", linewidth=2.5, markersize=6)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(f"{model_label}·准确率曲线", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(alpha=0.3)
    save_fig(fig, fname, out_dir)


def plot_val_metrics(history: dict, model_label: str, out_dir: Path, fname: str) -> None:
    """验证集 F1 / AUC 曲线（独立图）"""
    epochs = range(1, len(history["val_f1"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = palette(2, "sci")

    ax.plot(epochs, history["val_f1"], "-o", color=colors[0],
            label="验证 F1", linewidth=2.5, markersize=6)
    if history.get("val_auc") and any(history["val_auc"]):
        ax.plot(epochs, history["val_auc"], "-s", color=colors[1],
                label="验证 AUC", linewidth=2.5, markersize=6)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("指标值", fontsize=13)
    ax.set_title(f"{model_label}·F1 与 AUC 曲线", fontsize=16, fontweight="600", pad=15)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(alpha=0.3)
    save_fig(fig, fname, out_dir)


def plot_confusion_matrix_count(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
    model_label: str, out_dir: Path, fname: str,
) -> None:
    """混淆矩阵（样本数，独立图）"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={"label": "样本数"}, annot_kws={"fontsize": 12},
                linewidths=1, linecolor="white")
    ax.set_xlabel("预测标签", fontsize=13)
    ax.set_ylabel("真实标签", fontsize=13)
    ax.set_title(f"{model_label}·混淆矩阵（样本数）", fontsize=16, fontweight="600", pad=15)
    save_fig(fig, fname, out_dir)


def plot_confusion_matrix_norm(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
    model_label: str, out_dir: Path, fname: str,
) -> None:
    """混淆矩阵（归一化，独立图）"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={"label": "比例"}, annot_kws={"fontsize": 12},
                linewidths=1, linecolor="white")
    ax.set_xlabel("预测标签", fontsize=13)
    ax.set_ylabel("真实标签", fontsize=13)
    ax.set_title(f"{model_label}·混淆矩阵（归一化）", fontsize=16, fontweight="600", pad=15)
    save_fig(fig, fname, out_dir)


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str],
    model_label: str, out_dir: Path, fname: str,
) -> None:
    """ROC 曲线（独立图）"""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = palette(len(class_names), "sci")

    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[1], linewidth=3,
                label=f"ROC (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1.5, alpha=0.7)
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2.5,
                    label=f"{name} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("假正率 FPR", fontsize=13)
    ax.set_ylabel("真正率 TPR", fontsize=13)
    ax.set_title(f"{model_label}·ROC 曲线", fontsize=16, fontweight="600", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    save_fig(fig, fname, out_dir)


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str],
    model_label: str, out_dir: Path, fname: str,
) -> None:
    """PR 曲线（独立图）"""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = palette(len(class_names), "sci")

    if len(class_names) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        pr_auc = sk_auc(recall, precision)
        ax.plot(recall, precision, color=colors[1], linewidth=3,
                label=f"PR (AUC={pr_auc:.3f})")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        for i, name in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            pr_auc = sk_auc(recall, precision)
            ax.plot(recall, precision, color=colors[i % len(colors)], linewidth=2.5,
                    label=f"{name} (AUC={pr_auc:.2f})")

    ax.set_xlabel("召回率 Recall", fontsize=13)
    ax.set_ylabel("精确率 Precision", fontsize=13)
    ax.set_title(f"{model_label}·PR 曲线", fontsize=16, fontweight="600", pad=15)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)
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


# 兼容旧接口
def plot_history(history: dict, model_label: str, out_dir: Path, fname_prefix: str) -> None:
    """生成所有训练历史图（拆分为独立文件）"""
    plot_train_loss(history, model_label, out_dir, f"{fname_prefix}_loss")
    plot_train_acc(history, model_label, out_dir, f"{fname_prefix}_acc")
    plot_val_metrics(history, model_label, out_dir, f"{fname_prefix}_metrics")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
    model_label: str, out_dir: Path, fname_prefix: str,
) -> None:
    """生成混淆矩阵图（拆分为独立文件）"""
    plot_confusion_matrix_count(y_true, y_pred, class_names, model_label, out_dir,
                                 f"{fname_prefix}_cm_count")
    plot_confusion_matrix_norm(y_true, y_pred, class_names, model_label, out_dir,
                                f"{fname_prefix}_cm_norm")


def plot_roc_pr(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str],
    model_label: str, out_dir: Path, fname_prefix: str,
) -> None:
    """生成 ROC 和 PR 曲线（拆分为独立文件）"""
    plot_roc_curve(y_true, y_prob, class_names, model_label, out_dir,
                   f"{fname_prefix}_roc")
    plot_pr_curve(y_true, y_prob, class_names, model_label, out_dir,
                  f"{fname_prefix}_pr")
