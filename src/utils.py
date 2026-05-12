"""通用工具：日志、随机种子、计时。"""
from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from rich.logging import RichHandler


def get_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    log = logger or get_logger()
    start = time.time()
    log.info(f"⏱  开始：{label}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        log.info(f"✓  完成：{label}（用时 {elapsed:.1f}s）")


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
