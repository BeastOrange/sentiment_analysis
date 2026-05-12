"""脚本 10：从已有日志重新生成训练可视化图表（独立、科研风格）。

不重新训练模型，直接从 outputs/logs/ 读取历史和概率，重新生成图表。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import CFG, FIGURES_DIR, LOGS_DIR
from src.training.viz_training_v2 import (
    plot_confusion_matrix,
    plot_history,
    plot_roc_pr,
    save_classification_report,
)
from src.utils import get_logger, load_json, timed
from src.viz import apply_style

log = get_logger("10_regen_train_viz")


def regen_textcnn() -> None:
    """重新生成 TextCNN 图表"""
    log.info("TextCNN...")
    history = load_json(LOGS_DIR / "textcnn_binary_history.json")
    data = np.load(LOGS_DIR / "textcnn_binary_probs.npz")
    probs, labels = data["probs"], data["labels"]
    preds = probs.argmax(-1)

    plot_history(history, "TextCNN", FIGURES_DIR, "train_textcnn")
    plot_confusion_matrix(labels, preds, list(CFG.binary_label_names_zh),
                          "TextCNN", FIGURES_DIR, "train_textcnn")
    plot_roc_pr(labels, probs, list(CFG.binary_label_names_zh),
                "TextCNN", FIGURES_DIR, "train_textcnn")
    save_classification_report(labels, preds, list(CFG.binary_label_names_zh),
                                LOGS_DIR / "textcnn_binary_report.csv")


def regen_bilstm() -> None:
    """重新生成 BiLSTM 图表"""
    log.info("BiLSTM-Attention...")
    history = load_json(LOGS_DIR / "bilstm_binary_history.json")
    data = np.load(LOGS_DIR / "bilstm_binary_probs.npz")
    probs, labels = data["probs"], data["labels"]
    preds = probs.argmax(-1)

    plot_history(history, "BiLSTM-Attention", FIGURES_DIR, "train_bilstm")
    plot_confusion_matrix(labels, preds, list(CFG.binary_label_names_zh),
                          "BiLSTM-Attention", FIGURES_DIR, "train_bilstm")
    plot_roc_pr(labels, probs, list(CFG.binary_label_names_zh),
                "BiLSTM-Attention", FIGURES_DIR, "train_bilstm")
    save_classification_report(labels, preds, list(CFG.binary_label_names_zh),
                                LOGS_DIR / "bilstm_binary_report.csv")


def regen_bert() -> None:
    """重新生成 BERT 图表"""
    log.info("BERT...")
    history = load_json(LOGS_DIR / "bert_binary_history.json")
    data = np.load(LOGS_DIR / "bert_binary_probs.npz")
    probs, labels = data["probs"], data["labels"]
    preds = probs.argmax(-1)

    plot_history(history, "BERT (bert-base-chinese)", FIGURES_DIR, "train_bert")
    plot_confusion_matrix(labels, preds, list(CFG.binary_label_names_zh),
                          "BERT", FIGURES_DIR, "train_bert")
    plot_roc_pr(labels, probs, list(CFG.binary_label_names_zh),
                "BERT", FIGURES_DIR, "train_bert")
    save_classification_report(labels, preds, list(CFG.binary_label_names_zh),
                                LOGS_DIR / "bert_binary_report.csv")


def regen_emotion() -> None:
    """重新生成六情绪图表"""
    log.info("六情绪 BiLSTM-Attention...")
    history = load_json(LOGS_DIR / "bilstm_emotion_history.json")
    data = np.load(LOGS_DIR / "bilstm_emotion_probs.npz")
    probs, labels = data["probs"], data["labels"]
    preds = probs.argmax(-1)

    plot_history(history, "BiLSTM-Attention（六情绪）", FIGURES_DIR, "train_emotion")
    plot_confusion_matrix(labels, preds, list(CFG.emotion_label_names_zh),
                          "六情绪 BiLSTM-Attention", FIGURES_DIR, "train_emotion")
    plot_roc_pr(labels, probs, list(CFG.emotion_label_names_zh),
                "六情绪 BiLSTM-Attention", FIGURES_DIR, "train_emotion")
    save_classification_report(labels, preds, list(CFG.emotion_label_names_zh),
                                LOGS_DIR / "bilstm_emotion_report.csv")


def main() -> None:
    apply_style()

    with timed("TextCNN 可视化"):
        regen_textcnn()
    with timed("BiLSTM 可视化"):
        regen_bilstm()
    with timed("BERT 可视化"):
        regen_bert()
    with timed("六情绪可视化"):
        regen_emotion()

    log.info("✓ 训练可视化重新生成完成")


if __name__ == "__main__":
    main()
