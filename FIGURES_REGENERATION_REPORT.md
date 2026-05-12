# 图表重新生成完成报告

## 概述

所有图表已重新生成为**独立文件**，采用**现代科研风格**：
- **DPI 300**（高分辨率，适合论文发表）
- **更大尺寸**（8-11 英寸宽，便于查看细节）
- **科研配色**（蓝色系为主，专业简约）
- **独立图表**（每个指标一张图，不再拼接）

## 图表统计

| 类别 | 数量 | 说明 |
| --- | ---: | --- |
| EDA 探索性分析 | 25 | 数据集规模、长度分布、词云、标签分布等 |
| 预处理效果 | 20 | 清洗效果、词表分析、数据切分等 |
| 训练历史与评估 | 38 | Loss/Acc/F1 曲线、混淆矩阵、ROC/PR 等 |
| 模型融合对比 | 14 | 指标对比、ROC/PR 叠加、一致性分析等 |
| 六情绪分析 | 6 | 情绪分类性能、抑郁风险映射等 |
| **总计** | **103** | 全部独立、高分辨率、科研风格 |

## 新旧对比

### 旧版（原 34 张拼接图）
- 多个子图拼接在一张图中
- 尺寸较小（figsize 11-13 英寸，DPI 160）
- 文字可能重叠
- 不便于单独引用

### 新版（103 张独立图）
- 每个指标独立一张图
- 尺寸更大（8-11 英寸，DPI 300）
- 文字清晰，间距合理
- 便于论文单独引用

## 主要改进

### 1. 视觉风格
- **配色**：从鲜艳的 Tailwind 色系改为沉稳的科研蓝色系
- **字体**：标题 16pt 粗体，轴标签 13pt，图例 11-12pt
- **网格**：半透明网格线（alpha 0.3），辅助阅读
- **边框**：移除顶部和右侧边框，保留左侧和底部

### 2. 图表拆分示例

#### 训练历史（原 1 张 → 现 3 张）
- `train_textcnn_loss.png` - 训练与验证 Loss 曲线
- `train_textcnn_acc.png` - 训练与验证 Accuracy 曲线
- `train_textcnn_metrics.png` - 验证 F1 与 AUC 曲线

#### 混淆矩阵（原 1 张 → 现 2 张）
- `train_textcnn_cm_count.png` - 样本数混淆矩阵
- `train_textcnn_cm_norm.png` - 归一化混淆矩阵

#### ROC/PR（原 1 张 → 现 2 张）
- `train_textcnn_roc.png` - ROC 曲线
- `train_textcnn_pr.png` - PR 曲线

### 3. 新增图表

#### EDA 阶段
- 数据质量指标（重复率、清洗保留率）
- 二分类与六情绪的独立长度分布图
- 每个情绪类别的独立词云（6 张）
- 标签分布的柱状图和饼图（各 2 张）

#### 预处理阶段
- 清洗前后长度对比（独立）
- 被清洗掉的字符数分布（独立）
- Token 数分布（独立）
- Zipf 分布（独立）
- 词表覆盖率曲线（独立）
- 数据切分样本量（独立）
- 各 Split 内类别分布（独立）

#### 模型对比阶段
- 指标对比柱状图（独立）
- 指标热力图（独立）
- ROC 曲线叠加（独立）
- PR 曲线叠加（独立）
- 模型一致性矩阵（独立）
- 预测正误分布（独立）
- 融合权重柱状图（独立）
- 融合模型混淆矩阵（2 张）

## 文件命名规范

### EDA
- `eda_01_dataset_size.png` - 数据集规模
- `eda_02_data_quality.png` - 数据质量
- `eda_03_binary_length.png` - 二分类长度分布
- `eda_04_emotion_length.png` - 六情绪长度分布
- `eda_05_token_length.png` - Token 数分布
- `eda_06_token_freq_compare.png` - 高频词对比
- `eda_07_wordcloud_normal.png` - 非抑郁词云
- `eda_08_wordcloud_depressive.png` - 抑郁词云
- `eda_09_wordcloud_emo_0.png` ~ `eda_09_wordcloud_emo_5.png` - 六情绪词云
- `eda_15_binary_label_bar.png` - 二分类标签柱状图
- `eda_16_binary_label_pie.png` - 二分类标签饼图
- `eda_17_emotion_label_bar.png` - 六情绪标签柱状图
- `eda_18_emotion_label_pie.png` - 六情绪标签饼图

### 预处理
- `preprocess_01_binary_clean_length.png` - 二分类清洗长度对比
- `preprocess_02_binary_clean_diff.png` - 二分类清洗效果
- `preprocess_03_binary_token_count.png` - 二分类 Token 数
- `preprocess_04_binary_vocab_zipf.png` - 二分类 Zipf 分布
- `preprocess_05_binary_vocab_coverage.png` - 二分类词表覆盖率
- `preprocess_06_binary_split_sizes.png` - 二分类切分样本量
- `preprocess_07_binary_split_dist.png` - 二分类切分类别分布
- `preprocess_08_emotion_clean_length.png` ~ `preprocess_14_emotion_split_dist.png` - 六情绪对应图表

### 训练
- `train_textcnn_loss.png` - TextCNN Loss 曲线
- `train_textcnn_acc.png` - TextCNN Accuracy 曲线
- `train_textcnn_metrics.png` - TextCNN F1/AUC 曲线
- `train_textcnn_cm_count.png` - TextCNN 混淆矩阵（样本数）
- `train_textcnn_cm_norm.png` - TextCNN 混淆矩阵（归一化）
- `train_textcnn_roc.png` - TextCNN ROC 曲线
- `train_textcnn_pr.png` - TextCNN PR 曲线
- `train_bilstm_*.png` - BiLSTM 对应图表（7 张）
- `train_bert_*.png` - BERT 对应图表（7 张）
- `train_emotion_*.png` - 六情绪对应图表（7 张）

### 模型对比
- `compare_metrics_bar.png` - 指标对比柱状图
- `compare_metrics_heatmap.png` - 指标热力图
- `compare_roc_overlay.png` - ROC 曲线叠加
- `compare_pr_overlay.png` - PR 曲线叠加
- `compare_agreement_matrix.png` - 一致性矩阵
- `compare_agreement_stats.png` - 预测正误分布
- `compare_fusion_weights.png` - 融合权重
- `compare_ensemble_cm_count.png` - 融合模型混淆矩阵（样本数）
- `compare_ensemble_cm_norm.png` - 融合模型混淆矩阵（归一化）

## 使用建议

### 论文引用
每个图表都是独立文件，可以直接在 LaTeX/Word 中单独引用：

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{outputs/figures/train_textcnn_loss.png}
  \caption{TextCNN 训练与验证 Loss 曲线}
  \label{fig:textcnn_loss}
\end{figure}
```

### Markdown 引用
```markdown
![TextCNN Loss 曲线](outputs/figures/train_textcnn_loss.png)
```

### 幻灯片使用
- 每张图都足够大，可以直接插入 PPT/Keynote
- 高分辨率（DPI 300）保证投影清晰
- 科研配色适合学术汇报

## 复现说明

如需重新生成所有图表：

```bash
# 方式 1：单独运行各脚本
uv run python scripts/02_explore_data_v2.py      # EDA 图表
uv run python scripts/03_preprocess_v2.py        # 预处理图表
uv run python scripts/10_regen_train_viz.py     # 训练图表
uv run python scripts/11_regen_compare_viz.py   # 对比图表

# 方式 2：一键重新生成（需要先有训练日志）
uv run python scripts/09_regenerate_all_figures.py
```

## 技术细节

### 样式配置（src/viz.py）
```python
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,          # 高分辨率输出
    "axes.titlesize": 16,        # 标题字号
    "axes.titleweight": "600",   # 标题粗细
    "axes.labelsize": 13,        # 轴标签字号
    "xtick.labelsize": 11,       # 刻度字号
    "ytick.labelsize": 11,
    "legend.fontsize": 11,       # 图例字号
    "grid.alpha": 0.6,           # 网格透明度
    "axes.facecolor": "#FAFAFA", # 背景色
})
```

### 科研配色
```python
PALETTE_SCI = [
    "#0C5DA5",  # 深蓝
    "#E34A33",  # 橙红
    "#00B945",  # 绿色
    "#FF9500",  # 橙色
    "#845B97",  # 紫色
    "#474747",  # 灰色
]
```

---

**生成时间**：2026-05-12  
**工具版本**：matplotlib 3.8+, seaborn 0.13+  
**Python 版本**：3.12+
