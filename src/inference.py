"""统一推理模块：加载所有训练好的模型，提供单文本/批量预测接口。"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, MODELS_DIR, PROCESSED_DIR, get_device
from src.data.preprocess import clean_text, encode
from src.models.bert_clf import BertClassifier, build_bert
from src.models.bilstm import BiLSTMAttention
from src.models.textcnn import TextCNN
from src.utils import load_json

# 心理学经验：六情绪 → 抑郁风险分数
DEPRESSION_RISK = {
    0: 0.2,  # 中性
    1: 0.0,  # 高兴
    2: 0.5,  # 愤怒
    3: 1.0,  # 悲伤
    4: 0.7,  # 恐惧
    5: 0.1,  # 惊奇
}

# 最优融合权重（来自 ensemble_summary.json）
ENSEMBLE_WEIGHTS = {"TextCNN": 0.0, "BiLSTM-Attention": 0.8, "BERT": 0.2}


@lru_cache(maxsize=1)
def _device():
    return get_device()


@lru_cache(maxsize=1)
def _load_vocab_binary() -> dict[str, int]:
    return load_json(PROCESSED_DIR / "vocab_binary.json")


@lru_cache(maxsize=1)
def _load_vocab_emotion() -> dict[str, int]:
    return load_json(PROCESSED_DIR / "vocab_emotion.json")


@lru_cache(maxsize=1)
def load_textcnn() -> TextCNN:
    vocab = _load_vocab_binary()
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=CFG.model.embed_dim,
        num_classes=2,
        kernel_sizes=CFG.model.cnn_kernel_sizes,
        num_filters=CFG.model.cnn_num_filters,
        dropout=CFG.model.dropout,
    )
    ckpt = torch.load(MODELS_DIR / "textcnn_binary.pt",
                      map_location=_device(), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(_device()).eval()
    return model


@lru_cache(maxsize=1)
def load_bilstm_binary() -> BiLSTMAttention:
    vocab = _load_vocab_binary()
    model = BiLSTMAttention(
        vocab_size=len(vocab),
        embed_dim=CFG.model.embed_dim,
        hidden_dim=CFG.model.lstm_hidden,
        num_layers=CFG.model.lstm_layers,
        num_classes=2,
        dropout=CFG.model.dropout,
    )
    ckpt = torch.load(MODELS_DIR / "bilstm_binary.pt",
                      map_location=_device(), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(_device()).eval()
    return model


@lru_cache(maxsize=1)
def load_bilstm_emotion() -> BiLSTMAttention:
    vocab = _load_vocab_emotion()
    model = BiLSTMAttention(
        vocab_size=len(vocab),
        embed_dim=CFG.model.embed_dim,
        hidden_dim=CFG.model.lstm_hidden,
        num_layers=CFG.model.lstm_layers,
        num_classes=len(CFG.emotion_label_names_zh),
        dropout=CFG.model.dropout,
    )
    ckpt = torch.load(MODELS_DIR / "bilstm_emotion.pt",
                      map_location=_device(), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(_device()).eval()
    return model


@lru_cache(maxsize=1)
def load_bert():
    tokenizer, hf_model = build_bert(CFG.model.bert_name, num_classes=2)
    model = BertClassifier(hf_model)
    ckpt = torch.load(MODELS_DIR / "bert_binary.pt",
                      map_location=_device(), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(_device()).eval()
    return tokenizer, model


def _predict_indexed(model, text: str, vocab: dict[str, int]) -> np.ndarray:
    """对索引模型做单文本预测，返回 softmax 概率向量。"""
    cleaned = clean_text(text)
    ids, _ = encode(cleaned, vocab, CFG.data.max_len)
    ids_t = torch.from_numpy(ids).long().unsqueeze(0).to(_device())
    with torch.no_grad():
        logits = model(input_ids=ids_t)
    return torch.softmax(logits, dim=-1).cpu().numpy()[0]


def _predict_bert(text: str) -> np.ndarray:
    tokenizer, model = load_bert()
    cleaned = clean_text(text)
    enc = tokenizer(cleaned, padding=True, truncation=True,
                    max_length=CFG.data.max_len, return_tensors="pt")
    enc = {k: v.to(_device()) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc)
    return torch.softmax(logits, dim=-1).cpu().numpy()[0]


def predict_binary(text: str, use_models: list[str] | None = None) -> dict:
    """二分类抑郁倾向预测（含融合）。

    返回：
        {
            "TextCNN": {"prob": [p0, p1], "label": "...", "confidence": 0.xx},
            "BiLSTM-Attention": {...},
            "BERT": {...},
            "Ensemble": {...},  # 加权融合
        }
    """
    use_models = use_models or ["TextCNN", "BiLSTM-Attention", "BERT"]
    vocab = _load_vocab_binary()
    results: dict[str, dict] = {}
    probs_dict: dict[str, np.ndarray] = {}

    if "TextCNN" in use_models:
        probs = _predict_indexed(load_textcnn(), text, vocab)
        probs_dict["TextCNN"] = probs
    if "BiLSTM-Attention" in use_models:
        probs = _predict_indexed(load_bilstm_binary(), text, vocab)
        probs_dict["BiLSTM-Attention"] = probs
    if "BERT" in use_models:
        probs = _predict_bert(text)
        probs_dict["BERT"] = probs

    for name, probs in probs_dict.items():
        label_idx = int(probs.argmax())
        results[name] = {
            "prob": probs.tolist(),
            "label": CFG.binary_label_names_zh[label_idx],
            "label_idx": label_idx,
            "confidence": float(probs[label_idx]),
        }

    # 加权融合
    if len(probs_dict) >= 2:
        fused = np.zeros(2)
        total_w = 0.0
        for name, probs in probs_dict.items():
            w = ENSEMBLE_WEIGHTS.get(name, 0.0)
            fused += w * probs
            total_w += w
        if total_w > 0:
            fused /= total_w
            label_idx = int(fused.argmax())
            results["Ensemble (融合)"] = {
                "prob": fused.tolist(),
                "label": CFG.binary_label_names_zh[label_idx],
                "label_idx": label_idx,
                "confidence": float(fused[label_idx]),
            }

    return results


def predict_emotion(text: str) -> dict:
    """六情绪预测 + 抑郁风险分数。

    返回：
        {
            "label": "悲伤",
            "label_idx": 3,
            "confidence": 0.92,
            "probs": [p0, p1, p2, p3, p4, p5],
            "depression_risk": 0.92,  # 加权后的抑郁风险分数
            "class_names": [...],
        }
    """
    vocab = _load_vocab_emotion()
    model = load_bilstm_emotion()
    probs = _predict_indexed(model, text, vocab)

    label_idx = int(probs.argmax())
    risk_scores = np.array([DEPRESSION_RISK[i] for i in range(len(probs))])
    risk = float((probs * risk_scores).sum())

    return {
        "label": CFG.emotion_label_names_zh[label_idx],
        "label_idx": label_idx,
        "confidence": float(probs[label_idx]),
        "probs": probs.tolist(),
        "depression_risk": risk,
        "class_names": list(CFG.emotion_label_names_zh),
    }


def predict_batch_binary(texts: list[str],
                          use_models: list[str] | None = None) -> list[dict]:
    """批量二分类预测。"""
    return [predict_binary(t, use_models) for t in texts]


def predict_batch_emotion(texts: list[str]) -> list[dict]:
    """批量六情绪预测。"""
    return [predict_emotion(t) for t in texts]


def predict_all(text: str) -> dict:
    """同时预测二分类 + 六情绪 + 抑郁风险。"""
    return {
        "binary": predict_binary(text),
        "emotion": predict_emotion(text),
        "cleaned_text": clean_text(text),
    }
