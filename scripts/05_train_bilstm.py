"""脚本 05：训练 BiLSTM + Attention（二分类）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CFG, FIGURES_DIR, LOGS_DIR, MODELS_DIR, PROCESSED_DIR, SEED, get_device
from src.data.datasets import df_to_indexed
from src.data.preprocess import tokenize
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

log = get_logger("05_bilstm")


def viz_attention(
    model: BiLSTMAttention, vocab: dict[str, int], device: torch.device,
    samples: list[tuple[str, int]], fname: str,
) -> None:
    """选取若干样本展示注意力权重热力图。"""
    from src.data.preprocess import encode
    model.eval()

    fig, axes = plt.subplots(len(samples), 1, figsize=(13, 1.2 + 0.8 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for ax, (text, true_label) in zip(axes, samples):
        toks = tokenize(text)[:CFG.data.max_len]
        ids, _ = encode(text, vocab, CFG.data.max_len)
        ids_t = torch.from_numpy(ids).long().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, attn = model.forward_with_attn(ids_t)
        attn = attn.squeeze(0).cpu().numpy()[:len(toks)]
        prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(prob.argmax())

        attn_norm = attn / (attn.max() + 1e-8)
        ax.imshow(attn_norm[None, :], aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, fontsize=9, rotation=0)
        ax.set_yticks([])
        ax.set_title(
            f"真实：{CFG.binary_label_names_zh[true_label]} | "
            f"预测：{CFG.binary_label_names_zh[pred]}（置信度 {prob[pred]:.2f}）",
            fontsize=10, loc="left",
        )
    fig.suptitle("BiLSTM-Attention·注意力权重可视化", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, fname, FIGURES_DIR)


def main() -> None:
    set_seed(SEED)
    apply_style()
    device = get_device()
    log.info(f"设备：{device}")

    train_df = pd.read_parquet(PROCESSED_DIR / "binary_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "binary_val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "binary_test.parquet")
    vocab = load_json(PROCESSED_DIR / "vocab_binary.json")

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
        num_classes=2,
        dropout=CFG.model.dropout,
    ).to(device)
    log.info(f"参数量：{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    save_path = MODELS_DIR / "bilstm_binary.pt"
    history = train_loop(
        model, train_loader, val_loader, device,
        num_classes=2,
        epochs=CFG.train.epochs_baseline,
        lr=CFG.train.lr_baseline,
        weight_decay=CFG.train.weight_decay,
        grad_clip=CFG.train.grad_clip,
        model_label="BiLSTM-Attn",
        save_path=save_path,
        early_stop=CFG.train.early_stop_patience,
    )
    dump_json(history.to_dict(), LOGS_DIR / "bilstm_binary_history.json")

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    metrics, probs, labels = evaluate(model, test_loader, device, 2, return_probs=True)
    log.info(f"测试集：{metrics}")
    dump_json(metrics, LOGS_DIR / "bilstm_binary_test.json")
    np.savez(LOGS_DIR / "bilstm_binary_probs.npz", probs=probs, labels=labels)

    plot_history(history.to_dict(), "BiLSTM-Attention", FIGURES_DIR, "train_04_bilstm_history")
    plot_confusion_matrix(
        labels, probs.argmax(-1), list(CFG.binary_label_names_zh),
        "BiLSTM-Attention", FIGURES_DIR, "train_05_bilstm_cm",
    )
    plot_roc_pr(
        labels, probs, list(CFG.binary_label_names_zh),
        "BiLSTM-Attention", FIGURES_DIR, "train_06_bilstm_roc",
    )
    save_classification_report(
        labels, probs.argmax(-1), list(CFG.binary_label_names_zh),
        LOGS_DIR / "bilstm_binary_report.csv",
    )

    # 注意力权重可视化（取一些代表性测试样本）
    test_pos = test_df[test_df["label"] == 0].sample(2, random_state=SEED).to_dict("records")
    test_neg = test_df[test_df["label"] == 1].sample(2, random_state=SEED).to_dict("records")
    samples = [(r["text"][:60], r["label"]) for r in (test_neg + test_pos)]
    viz_attention(model, vocab, device, samples, "train_07_bilstm_attention")

    log.info("✓ BiLSTM-Attention 完成")


if __name__ == "__main__":
    main()
