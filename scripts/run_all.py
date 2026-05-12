"""一键运行全部实验流程。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "scripts/01_download_data.py",
    "scripts/02_explore_data.py",
    "scripts/03_preprocess.py",
    "scripts/04_train_textcnn.py",
    "scripts/05_train_bilstm.py",
    "scripts/06_train_bert.py",
    "scripts/07_fuse_and_compare.py",
    "scripts/08_emotion_analysis.py",
]


def main() -> None:
    for script in SCRIPTS:
        path = ROOT / script
        print(f"\n{'='*60}")
        print(f"▶ 运行：{script}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            env={**__import__("os").environ, "DISABLE_TQDM": "1"},
        )
        if result.returncode != 0:
            print(f"✗ 失败：{script}（退出码 {result.returncode}）")
            sys.exit(result.returncode)
        print(f"✓ 完成：{script}")

    print(f"\n{'='*60}")
    print("✓ 全部实验完成！")
    print(f"  图表：{ROOT / 'outputs/figures/'}")
    print(f"  模型：{ROOT / 'outputs/models/'}")
    print(f"  日志：{ROOT / 'outputs/logs/'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
