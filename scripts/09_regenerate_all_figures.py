"""脚本 09：重新生成所有图表（独立、大尺寸、科研风格）。

将原本拼接的子图全部拆分为独立文件，提升可读性与论文质量。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, timed

log = get_logger("09_regen")


def main() -> None:
    log.info("开始重新生成所有图表...")

    # 按顺序调用各个脚本
    scripts = [
        "02_explore_data.py",
        "03_preprocess.py",
        "04_train_textcnn.py",
        "05_train_bilstm.py",
        "06_train_bert.py",
        "07_fuse_and_compare.py",
        "08_emotion_analysis.py",
    ]

    for script in scripts:
        script_path = Path(__file__).parent / script
        log.info(f"⏱  执行：{script}")
        with timed(script):
            exec(open(script_path).read(), {"__name__": "__main__"})

    log.info("✓ 所有图表重新生成完成")


if __name__ == "__main__":
    main()
