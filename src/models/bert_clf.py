"""BERT 分类器（wrapper），自动选择 transformers 的 ForSequenceClassification。"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_bert(model_name: str, num_classes: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    return tokenizer, model


class BertClassifier(nn.Module):
    """对 transformers 模型的轻量封装，使训练循环签名统一。"""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor, attention_mask=None,
                token_type_ids=None, labels=None, **_) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out.logits
