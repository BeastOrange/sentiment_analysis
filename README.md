# 基于社交媒体文本的中文情感分析与抑郁倾向识别

通过 NLP 与机器学习方法，构建能够自动识别用户抑郁情绪的智能分析系统。
分析社交媒体短文本内容（如微博、小红书等公开评论），及早发现潜在的抑郁
倾向，为心理健康干预提供技术支持。

## 环境

- macOS / Linux，Python 3.11
- 依赖管理：[uv](https://github.com/astral-sh/uv)
- 训练加速：Apple M 系列芯片使用 PyTorch MPS 后端

## 快速开始

```bash
uv sync                          # 安装依赖
uv run python scripts/01_download_data.py
uv run python scripts/02_explore_data.py
uv run python scripts/03_preprocess.py
uv run python scripts/04_train_textcnn.py
uv run python scripts/05_train_bilstm.py
uv run python scripts/06_train_bert.py
uv run python scripts/07_fuse_and_compare.py
uv run python scripts/08_emotion_analysis.py
```

或一键：

```bash
uv run python scripts/run_all.py
```

## 项目结构

```text
sentiment_analysis/
├── pyproject.toml
├── src/
│   ├── config.py            # 全局配置
│   ├── viz.py               # 可视化（中文字体 + 现代风格）
│   ├── data/                # 数据加载与预处理
│   ├── models/              # TextCNN / BiLSTM-Attn / BERT
│   └── training/            # 训练循环、评估、融合
├── scripts/                 # 顺序执行的实验脚本
├── data/                    # 数据（gitignore）
└── outputs/
    ├── figures/             # 所有可视化图
    ├── models/              # 模型权重
    └── logs/                # 训练日志
```

## 模型对比

| 模型 | 类型 | 测试 Accuracy | F1 | AUC |
| --- | --- | ---: | ---: | ---: |
| TextCNN | 卷积，Kim 2014 多尺度核 | 0.9760 | 0.9761 | 0.9959 |
| BiLSTM-Attention | 双向 LSTM + 加性注意力 | 0.9808 | 0.9810 | 0.9976 |
| BERT (bert-base-chinese) | 预训练 Transformer 微调 | 0.9668 | 0.9671 | 0.9902 |
| **加权融合（Ensemble）** | TextCNN 0.0 / BiLSTM 0.8 / BERT 0.2 | **0.9824** | **0.9826** | **0.9974** |

> 二分类任务：将公开微博情感语料（120k 条）映射为「非抑郁倾向 / 抑郁倾向」二分类问题，
> 训练 20k / 验证 2.5k / 测试 2.5k；BERT 由于 M4 MPS 训练吞吐受限，使用 8k 子集 × 2 epoch。

## 六情绪迁移分析

使用 HuggingFace `souljoy/COVID-19_weibo_emotion`（13.6k 条真实微博，6 类：中性 / 高兴 /
愤怒 / 悲伤 / 恐惧 / 惊奇）训练 BiLSTM-Attention，测试 **Accuracy 85.4%**，并基于心理学
经验将六情绪映射为「抑郁风险分数」（悲伤=1.0，恐惧=0.7，愤怒=0.5，中性=0.2，惊奇=0.1，
高兴=0.0），生成各情绪类别在抑郁风险分箱中的分布热力图。

## 可视化产出（34 张图，均为中文）

| 主题 | 文件前缀 | 数量 |
| --- | --- | ---: |
| 数据集探索（EDA） | `eda_*.png` | 7 |
| 预处理 + 词表分析 | `preprocess_*.png` | 6 |
| TextCNN 训练 | `train_01..03_textcnn_*.png` | 3 |
| BiLSTM-Attention 训练（含注意力热力图） | `train_04..07_bilstm_*.png` | 4 |
| BERT 训练 | `train_08..10_bert_*.png` | 3 |
| 模型融合与对比 | `compare_*.png` | 5 |
| 六情绪分析 | `emotion_*.png` | 6 |
