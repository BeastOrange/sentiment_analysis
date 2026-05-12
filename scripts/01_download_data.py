"""脚本 01：下载所有数据集。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import ensure_binary_dataset, ensure_emotion_dataset, load_binary, load_emotion
from src.utils import get_logger, timed

log = get_logger("01_download")


def main() -> None:
    with timed("下载微博情感二分类数据集"):
        ensure_binary_dataset()
    with timed("下载微博六情绪数据集"):
        ensure_emotion_dataset()

    df_bin = load_binary()
    em = load_emotion()
    log.info(f"二分类：{len(df_bin)} 条 | 标签：{df_bin['label'].value_counts().to_dict()}")
    for split, df in em.items():
        log.info(f"六情绪[{split}]：{len(df)} 条 | 标签：{df['label'].value_counts().to_dict()}")
    log.info("✓ 数据下载完成")


if __name__ == "__main__":
    main()
