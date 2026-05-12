"""全局配置：路径、超参、设备。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"

for _p in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DataConfig:
    binary_train_size: int = 20000
    binary_val_size: int = 2500
    binary_test_size: int = 2500
    emotion_train_size: int = 12000
    emotion_val_size: int = 1500
    emotion_test_size: int = 1500
    max_len: int = 80
    min_freq: int = 2
    vocab_size_cap: int = 50000


@dataclass
class TrainConfig:
    batch_size: int = 64
    bert_batch_size: int = 32
    lr_baseline: float = 1e-3
    lr_bert: float = 2e-5
    epochs_baseline: int = 6
    epochs_bert: int = 2
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1
    early_stop_patience: int = 3
    num_workers: int = 0


@dataclass
class ModelConfig:
    embed_dim: int = 200
    cnn_kernel_sizes: tuple[int, ...] = (2, 3, 4, 5)
    cnn_num_filters: int = 96
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.4
    bert_name: str = "bert-base-chinese"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    binary_label_names_zh: tuple[str, ...] = ("非抑郁倾向", "抑郁倾向")
    binary_label_names_en: tuple[str, ...] = ("Normal", "Depressive")

    emotion_label_names_zh: tuple[str, ...] = (
        "中性",
        "高兴",
        "愤怒",
        "悲伤",
        "恐惧",
        "惊奇",
    )


CFG = Config()
