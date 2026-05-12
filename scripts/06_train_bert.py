"""脚本 06：微调 BERT（bert-base-chinese，二分类）。

注意：在 M4 MPS 上 BERT 训练较慢，使用 small batch 和 epochs_bert=2。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CFG, FIGURES_DIR, LOGS_DIR, MODELS_DIR, PROCESSED_DIR, SEED, get_device
from src.data.datasets import BertTextDataset, make_bert_collate_fn
from src.models.bert_clf import BertClassifier, build_bert
from src.training.trainer import evaluate, train_loop
from src.training.viz_training import (
    plot_confusion_matrix,
    plot_history,
    plot_roc_pr,
    save_classification_report,
)
from src.utils import dump_json, get_logger, set_seed, timed
from src.viz import apply_style

log = get_logger("06_bert")


def main() -> None:
    set_seed(SEED)
    apply_style()
    device = get_device()
    log.info(f"设备：{device}")

    train_df = pd.read_parquet(PROCESSED_DIR / "binary_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "binary_val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "binary_test.parquet")

    # M4 MPS 上的微调子采样（保持类别均衡），避免单次跑过久
    bert_train_subset = 8000
    if len(train_df) > bert_train_subset:
        per_class = bert_train_subset // train_df["label"].nunique()
        train_df = (
            train_df.groupby("label", group_keys=False)
            .sample(n=per_class, random_state=SEED)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )
    log.info(f"BERT 子采样训练集：{len(train_df)}（验证 {len(val_df)} / 测试 {len(test_df)}）")

    with timed(f"加载预训练模型 {CFG.model.bert_name}"):
        tokenizer, hf_model = build_bert(CFG.model.bert_name, num_classes=2)
        model = BertClassifier(hf_model).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"参数量：{n_params:.1f}M")

    collate = make_bert_collate_fn(tokenizer, CFG.data.max_len)
    train_ds = BertTextDataset(train_df["text"].tolist(), train_df["label"].to_numpy())
    val_ds = BertTextDataset(val_df["text"].tolist(), val_df["label"].to_numpy())
    test_ds = BertTextDataset(test_df["text"].tolist(), test_df["label"].to_numpy())

    train_loader = DataLoader(train_ds, batch_size=CFG.train.bert_batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=CFG.train.bert_batch_size,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=CFG.train.bert_batch_size,
                             shuffle=False, collate_fn=collate)

    total_steps = len(train_loader) * CFG.train.epochs_bert
    warmup_steps = int(total_steps * CFG.train.warmup_ratio)

    def opt_factory(m):
        no_decay = ("bias", "LayerNorm.weight")
        params = [
            {
                "params": [p for n, p in m.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in m.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(params, lr=CFG.train.lr_bert)

    def sched_factory(opt):
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

    save_path = MODELS_DIR / "bert_binary.pt"
    history = train_loop(
        model, train_loader, val_loader, device,
        num_classes=2,
        epochs=CFG.train.epochs_bert,
        lr=CFG.train.lr_bert,
        weight_decay=0.01,
        grad_clip=CFG.train.grad_clip,
        optimizer_factory=opt_factory,
        scheduler_factory=sched_factory,
        model_label="BERT",
        save_path=save_path,
        early_stop=None,
        log_every=25,
    )
    dump_json(history.to_dict(), LOGS_DIR / "bert_binary_history.json")

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    metrics, probs, labels = evaluate(model, test_loader, device, 2, return_probs=True)
    log.info(f"测试集：{metrics}")
    dump_json(metrics, LOGS_DIR / "bert_binary_test.json")
    np.savez(LOGS_DIR / "bert_binary_probs.npz", probs=probs, labels=labels)

    plot_history(history.to_dict(), "BERT (bert-base-chinese)",
                 FIGURES_DIR, "train_08_bert_history")
    plot_confusion_matrix(
        labels, probs.argmax(-1), list(CFG.binary_label_names_zh),
        "BERT", FIGURES_DIR, "train_09_bert_cm",
    )
    plot_roc_pr(
        labels, probs, list(CFG.binary_label_names_zh),
        "BERT", FIGURES_DIR, "train_10_bert_roc",
    )
    save_classification_report(
        labels, probs.argmax(-1), list(CFG.binary_label_names_zh),
        LOGS_DIR / "bert_binary_report.csv",
    )

    log.info("✓ BERT 完成")


if __name__ == "__main__":
    main()
