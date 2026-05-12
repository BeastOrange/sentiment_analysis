"""通用训练/评估循环。支持索引 Dataset 与 BERT Dataset。"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import get_logger

log = get_logger("trainer")

# 仅在 TTY 中显示进度条；否则禁用以保持日志整洁
_TQDM_DISABLE = not sys.stderr.isatty() or os.getenv("DISABLE_TQDM") == "1"


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None = None


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)
    epoch_time: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, num_classes: int
) -> dict:
    average = "binary" if num_classes == 2 else "macro"
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    auc = None
    if y_prob is not None:
        try:
            if num_classes == 2:
                auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            auc = None
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
    return_probs: bool = False,
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    model.eval()
    crit = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            labels = batch.get("labels", batch.get("label"))
            logits = model(**batch)
            loss = crit(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = probs.argmax(axis=1)
    metrics = _compute_metrics(labels, preds, probs, num_classes)
    metrics["loss"] = total_loss / len(labels)
    if return_probs:
        return metrics, probs, labels
    return metrics, None, None


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    scheduler=None,
    grad_clip: float | None = 1.0,
    desc: str = "",
    log_every: int = 0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    pbar = tqdm(loader, desc=desc, leave=False, disable=_TQDM_DISABLE)
    n_steps = len(loader)
    for step, batch in enumerate(pbar, start=1):
        batch = _move_batch(batch, device)
        labels = batch.get("labels", batch.get("label"))
        optimizer.zero_grad()
        logits = model(**batch)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total += batch_size
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                          acc=f"{total_correct/total:.3f}")
        if log_every and (step % log_every == 0 or step == n_steps):
            log.info(
                f"{desc} step {step}/{n_steps} | "
                f"loss {total_loss/total:.4f} acc {total_correct/total:.3f}"
            )
    return total_loss / total, total_correct / total


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    grad_clip: float | None = 1.0,
    optimizer_factory: Callable | None = None,
    scheduler_factory: Callable | None = None,
    class_weights: torch.Tensor | None = None,
    model_label: str = "model",
    save_path: Path | None = None,
    early_stop: int | None = None,
    log_every: int = 0,
) -> TrainHistory:
    import time

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if optimizer_factory is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )
    else:
        optimizer = optimizer_factory(model)
    scheduler = scheduler_factory(optimizer) if scheduler_factory else None

    history = TrainHistory()
    best_f1 = -1.0
    patience = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, criterion,
            scheduler=scheduler, grad_clip=grad_clip,
            desc=f"[{model_label}] Epoch {epoch}/{epochs}",
            log_every=log_every,
        )
        val_metrics, _, _ = evaluate(model, val_loader, device, num_classes, criterion)
        elapsed = time.time() - t0

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_metrics["loss"])
        history.val_acc.append(val_metrics["accuracy"])
        history.val_f1.append(val_metrics["f1"])
        history.val_auc.append(val_metrics.get("auc") or 0.0)
        history.epoch_time.append(elapsed)

        log.info(
            f"[{model_label}] Ep{epoch:02d} | "
            f"训练 loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"验证 loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.3f} "
            f"f1 {val_metrics['f1']:.3f} auc {val_metrics.get('auc') or 0.0:.3f} | "
            f"用时 {elapsed:.1f}s"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience = 0
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(),
                            "epoch": epoch, "val_f1": best_f1}, save_path)
        else:
            patience += 1
            if early_stop and patience >= early_stop:
                log.info(f"[{model_label}] 早停（连续 {patience} 个 epoch 无提升）")
                break

    return history
