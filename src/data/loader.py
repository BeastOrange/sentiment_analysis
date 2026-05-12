"""数据加载：多源下载 + 回退。

二分类：weibo_senti_100k（10 万条微博正负情感）
六情绪：使用 SMP2020-EWECT 风格的 ChnSentiCorp 多类版本或合成

由于 HuggingFace 在某些网络下不稳定，本模块提供多源回退。
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from src.config import RAW_DIR
from src.utils import get_logger

log = get_logger("data_loader")

BINARY_FILENAME = "weibo_senti_100k.csv"
EMOTION_FILES = ("train.csv", "validation.csv", "test.csv")
EMOTION_REPO = "souljoy/COVID-19_weibo_emotion"
# 真实标签映射：happy=1, neutral=0, angry=2, sad=3, fear=4, surprise=5
EMOTION_LABEL_NAMES_ZH = {
    0: "中性",
    1: "高兴",
    2: "愤怒",
    3: "悲伤",
    4: "恐惧",
    5: "惊奇",
}


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        log.info(f"⬇  尝试下载：{url}")
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 14):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        log.info(f"   下载完成：{dest.name}（{downloaded/1024:.1f} KB）")
        return True
    except Exception as e:
        log.warning(f"   下载失败：{e}")
        if dest.exists():
            dest.unlink()
        return False


def _try_hf_hub(repo_id: str, filename: str, dest: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
        log.info(f"⬇  HuggingFace Hub：{repo_id}/{filename}")
        p = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset",
                            local_dir=dest.parent)
        src = Path(p)
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
        return True
    except Exception as e:
        log.warning(f"   HF Hub 失败：{e}")
        return False


def ensure_binary_dataset() -> Path:
    """获取微博情感二分类数据集。返回 CSV 路径，列：[label, review]。"""
    dest = RAW_DIR / BINARY_FILENAME
    if dest.exists() and dest.stat().st_size > 100_000:
        log.info(f"✓ 已存在：{dest}")
        return dest

    sources = [
        ("https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/"
         "weibo_senti_100k/intro.ipynb", None),
        ("https://github.com/dengxiuqi/weibo_sentiment_analysis/raw/master/data/"
         "weibo_senti_100k.csv", dest),
        ("https://gitee.com/dogecheng/python/raw/master/data/weibo_senti_100k.csv", dest),
    ]
    for url, target in sources:
        if target is None:
            continue
        if _download(url, target):
            return target

    if _try_hf_hub("dirtycomputer/weibo_senti_100k", "weibo_senti_100k.csv", dest):
        return dest
    if _try_hf_hub("seamew/WeiboSenti", "data.csv", dest):
        return dest

    log.warning("⚠  公开源不可用，构建内置示例数据集...")
    return _build_synthetic_binary(dest)


def ensure_emotion_dataset() -> dict[str, Path]:
    """获取真实多情绪微博数据集（6 类：中性/高兴/愤怒/悲伤/恐惧/惊奇）。

    来源：HuggingFace `souljoy/COVID-19_weibo_emotion`，共约 1.36 万条真实微博。
    返回 {split: path} 字典。
    """
    paths: dict[str, Path] = {}
    all_present = True
    for f in EMOTION_FILES:
        p = RAW_DIR / f
        paths[f.replace(".csv", "")] = p
        if not (p.exists() and p.stat().st_size > 10_000):
            all_present = False

    if all_present:
        log.info(f"✓ 已存在六情绪数据集：{list(paths.values())}")
        return paths

    try:
        from huggingface_hub import hf_hub_download
        for f in EMOTION_FILES:
            log.info(f"⬇  HuggingFace：{EMOTION_REPO}/{f}")
            p = hf_hub_download(
                repo_id=EMOTION_REPO, filename=f,
                repo_type="dataset", local_dir=str(RAW_DIR),
            )
            paths[f.replace(".csv", "")] = Path(p)
        return paths
    except Exception as e:
        log.error(f"   HF 下载失败：{e}")
        log.warning("⚠  公开源不可用，构建内置示例多情绪数据集...")
        synth = _build_synthetic_emotion(RAW_DIR / "train.csv")
        return {"train": synth}


def _build_synthetic_binary(dest: Path) -> Path:
    """合成兜底数据集（仅当公开源全部失败）。"""
    import random
    random.seed(42)
    pos_words = ["开心", "快乐", "美好", "喜欢", "棒", "幸福", "温暖", "好看", "完美", "赞"]
    neg_words = ["难过", "痛苦", "累", "想死", "绝望", "孤独", "失眠", "崩溃", "压抑", "黑暗"]
    pos_phrases = ["今天天气真好", "刚吃完一顿大餐", "终于完成了", "和朋友一起玩"]
    neg_phrases = ["一个人待着", "什么都不想做", "睡不着觉", "感觉活着没意义"]
    rows = []
    for _ in range(8000):
        n = random.randint(1, 3)
        text = random.choice(pos_phrases) + "，" + "、".join(random.choices(pos_words, k=n))
        rows.append({"label": 1, "review": text})
        text = random.choice(neg_phrases) + "，" + "、".join(random.choices(neg_words, k=n))
        rows.append({"label": 0, "review": text})
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(dest, index=False, encoding="utf-8")
    log.info(f"   合成完成：{len(df)} 条")
    return dest


def _build_synthetic_emotion(dest: Path) -> Path:
    import random
    random.seed(43)
    bank = {
        0: ("喜悦", ["开心", "幸福", "笑", "美好", "棒"], ["今天好开心", "终于做到了"]),
        1: ("愤怒", ["生气", "气死", "讨厌", "受不了"], ["真的太过分了", "气炸了"]),
        2: ("厌恶", ["恶心", "无聊", "烦", "嫌弃"], ["这种事真让人厌恶", "看着就烦"]),
        3: ("低落", ["难过", "绝望", "孤独", "累", "压抑"], ["一个人好孤独", "什么都不想做"]),
    }
    rows = []
    for lab, (_n, words, ph) in bank.items():
        for _ in range(3500):
            t = random.choice(ph) + "，" + "、".join(random.choices(words, k=random.randint(1, 3)))
            rows.append({"label": lab, "review": t})
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(dest, index=False, encoding="utf-8")
    log.info(f"   合成完成：{len(df)} 条")
    return dest


def load_binary(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or RAW_DIR / BINARY_FILENAME
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    label_col = cols.get("label") or cols.get("情感") or list(df.columns)[0]
    text_col = cols.get("review") or cols.get("text") or cols.get("评论") or list(df.columns)[1]
    df = df.rename(columns={label_col: "label", text_col: "text"})[["label", "text"]]
    df = df.dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    # 二分类：原数据 1=正向、0=负向 → 抑郁倾向取反：1=抑郁倾向（负向），0=非抑郁
    df["label"] = 1 - df["label"]
    return df


def load_emotion() -> dict[str, pd.DataFrame]:
    """加载六情绪数据集的 train/val/test 三个 split。"""
    out: dict[str, pd.DataFrame] = {}
    for split, fname in [("train", "train.csv"), ("val", "validation.csv"), ("test", "test.csv")]:
        p = RAW_DIR / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        # 原列：text, label_name, label
        df = df[["text", "label"]].dropna().reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        df["text"] = df["text"].astype(str)
        out[split] = df
    return out
