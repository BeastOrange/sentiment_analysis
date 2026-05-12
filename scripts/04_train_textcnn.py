"""脚本 04：训练 TextCNN（二分类）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CFG, FIGURES_DIR, LOGS_DIR, MODELS_DIR, PROCESSED_DIR, SEED, get_device
from src.data.datasets import df_to_indexed
from src.models.textcnn import TextCNN
from src.training.trainer import evaluate, train_loop
from src.training.viz_training import (
    plot_confusion_matrix,
    plot_history,
    plot_roc_pr,
    save_classification_report,
)
from src.utils import dump_json, get_logger, load_json, set_seed, timed
from src.viz import apply_style

log = get_logger("04_textcnn")


def main() -> None:
    set_seed(SEED)
    apply_style()
    device = get_device()
    log.info(f"设备：{device}")

    train_df = pd.read_parquet(PROCESSED_DIR / "binary_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "binary_val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "binary_test.parquet")
    vocab = load_json(PROCESSED_DIR / "vocab_binary.json")
    log.info(f"训练 {len(train_df)} | 验证 {len(val_df)} | 测试 {len(test_df)} | 词表 {len(vocab)}")

    with timed("构建 Dataset / DataLoader"):
        train_ds = df_to_indexed(train_df, vocab, CFG.data.max_len)
        val_ds = df_to_indexed(val_df, vocab, CFG.data.max_len)
        test_ds = df_to_indexed(test_df, vocab, CFG.data.max_len)
        train_loader = DataLoader(train_ds, batch_size=CFG.train.batch_size,
                                  shuffle=True, num_workers=CFG.train.num_workers)
        val_loader = DataLoader(val_ds, batch_size=CFG.train.batch_size,
                                shuffle=False, num_workers=CFG.train.num_workers)
        test_loader = DataLoader(test_ds, batch_size=CFG.train.batch_size,
                                 shuffle=False, num_workers=CFG.train.num_workers)

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=CFG.model.embed_dim,
        num_classes=2,
        kernel_sizes=CFG.model.cnn_kernel_sizes,
        num_filters=CFG.model.cnn_num_filters,
        dropout=CFG.model.dropout,
    ).to(device)
    log.info(f"参数量：{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    save_path = MODELS_DIR / "textcnn_binary.pt"
    history = train_loop(
        model, train_loader, val_loader, device,
        num_classes=2,
        epochs=CFG.train.epochs_baseline,
        lr=CFG.train.lr_baseline,
        weight_decay=CFG.train.weight_decay,
        grad_clip=CFG.train.grad_clip,
        model_label="TextCNN",
        save_path=save_path,
        early_stop=CFG.train.early_stop_patience,
    )
    dump_json(history.to_dict(), LOGS_DIR / "textcnn_binary_history.json")

    # 测试集评估（加载 best checkpoint）
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    metrics, probs, labels = evaluate(model, test_loader, device, 2, return_probs=True)
    log.info(f"测试集：{metrics}")
    dump_json(metrics, LOGS_DIR / "textcnn_binary_test.json")
    np.savez(LOGS_DIR / "textcnn_binary_probs.npz", probs=probs, labels=labels)

    plot_history(history.to_dict(), "TextCNN", FIGURES_DIR, "train_01_textcnn_history")
    plot_confusion_matrix(
        labels, probs.argmax(-1), list(CFG.binary_label_names_zh),
        "TextCNN", FIGURES_DIR, "train_02_textcnn_cm",
    )
    plot_roc_pr(
        labels, probs, list(CFG.binary_label_names_zh),
        "TextCNN", FIGURES_DIR, "train_03_textcnn_roc",
    )
    save_classification_report(
        labels, probs.argmax(-1), list(CFG.binary_label_names_zh),
        LOGS_DIR / "textcnn_binary_report.csv",
    )

    log.info("✓ TextCNN 完成")


if __name__ == "__main__":
    main()
