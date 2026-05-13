"""关于项目页面：数据集、模型、性能说明。"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown("# ℹ️ 关于项目")
    st.caption("基于社交媒体文本的中文情感分析与抑郁倾向识别系统")
    st.divider()

    # 项目概述
    st.markdown("## 🎯 项目目标")
    st.markdown("""
    通过 NLP 与机器学习方法，构建能够自动识别用户抑郁情绪的智能分析系统。
    分析社交媒体短文本内容（如微博、小红书等公开评论），及早发现潜在的抑郁倾向，
    为心理健康干预提供技术支持。
    """)

    st.divider()

    # 数据集
    st.markdown("## 📚 数据集")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("### 二分类·抑郁倾向")
            st.markdown("""
            - **来源**：`weibo_senti_100k` (SophonPlus)
            - **规模**：约 12 万条真实微博
            - **原标签**：正向 / 负向
            - **处理**：取反映射为「抑郁倾向」代理标签
            - **切分**：训练 20k / 验证 2.5k / 测试 2.5k
            """)

    with col2:
        with st.container(border=True):
            st.markdown("### 六情绪·细粒度")
            st.markdown("""
            - **来源**：HuggingFace `souljoy/COVID-19_weibo_emotion`
            - **规模**：约 1.36 万条真实微博
            - **类别**：中性 / 高兴 / 愤怒 / 悲伤 / 恐惧 / 惊奇
            - **切分**：训练 10k / 验证 1.5k / 测试 1.5k
            """)

    st.divider()

    # 模型架构
    st.markdown("## 🏗️ 模型架构")

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("### TextCNN")
            st.markdown("""
            **Kim 2014 多尺度卷积**
            - 词嵌入 200 维
            - 卷积核 (2, 3, 4, 5)
            - 每尺度 96 个 filter
            - 参数量 ~4.7M
            - 训练时间：~5 min
            """)
    with col2:
        with st.container(border=True):
            st.markdown("### BiLSTM-Attention")
            st.markdown("""
            **双向 LSTM + 加性注意力**
            - 词嵌入 200 维
            - 2 层双向 LSTM
            - 每方向隐藏 128
            - 加性 Attention
            - 参数量 ~5.6M
            - 训练时间：~10 min
            """)
    with col3:
        with st.container(border=True):
            st.markdown("### BERT")
            st.markdown("""
            **bert-base-chinese 微调**
            - 12 层 Transformer
            - 隐藏 768 / 12 头
            - 学习率 2e-5
            - Warmup 10%
            - 参数量 ~102M
            - 训练时间：~7 min (8k 子集)
            """)

    st.divider()

    # 性能指标
    st.markdown("## 📊 性能指标（测试集）")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 二分类（抑郁倾向）")
        import pandas as pd
        df_bin = pd.DataFrame([
            {"模型": "TextCNN", "Accuracy": 0.9760, "F1": 0.9761, "AUC": 0.9959},
            {"模型": "BiLSTM-Attention", "Accuracy": 0.9808, "F1": 0.9810, "AUC": 0.9976},
            {"模型": "BERT", "Accuracy": 0.9668, "F1": 0.9671, "AUC": 0.9902},
            {"模型": "Ensemble (融合)", "Accuracy": 0.9824, "F1": 0.9826, "AUC": 0.9974},
        ])
        st.dataframe(
            df_bin.style.format({"Accuracy": "{:.4f}", "F1": "{:.4f}", "AUC": "{:.4f}"})
                       .highlight_max(subset=["Accuracy", "F1", "AUC"], color="#FEF3C7"),
            use_container_width=True, hide_index=True,
        )

    with col2:
        st.markdown("### 六情绪分类")
        df_emo = pd.DataFrame([
            {"情绪": "中性", "Precision": 0.773, "Recall": 0.736, "F1": 0.754},
            {"情绪": "高兴", "Precision": 0.867, "Recall": 0.756, "F1": 0.808},
            {"情绪": "愤怒", "Precision": 0.784, "Recall": 0.844, "F1": 0.813},
            {"情绪": "悲伤", "Precision": 0.821, "Recall": 0.920, "F1": 0.868},
            {"情绪": "恐惧", "Precision": 0.914, "Recall": 0.892, "F1": 0.903},
            {"情绪": "惊奇", "Precision": 0.972, "Recall": 0.976, "F1": 0.974},
        ])
        st.dataframe(
            df_emo.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"})
                       .highlight_max(subset=["F1"], color="#FEF3C7"),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # 抑郁风险映射
    st.markdown("## 🧭 抑郁风险映射")
    st.markdown("""
    基于**心理学经验**，将六情绪加权映射为 0-1 的抑郁风险分数：
    """)

    import pandas as pd
    risk_df = pd.DataFrame([
        {"情绪": "悲伤 😢", "权重": 1.0, "说明": "直接关联抑郁核心症状"},
        {"情绪": "恐惧 😨", "权重": 0.7, "说明": "焦虑障碍常与抑郁共病"},
        {"情绪": "愤怒 😡", "权重": 0.5, "说明": "压抑的愤怒可能转化为抑郁"},
        {"情绪": "中性 😐", "权重": 0.2, "说明": "情感麻木是抑郁的早期信号"},
        {"情绪": "惊奇 😮", "权重": 0.1, "说明": "几乎与抑郁无关"},
        {"情绪": "高兴 😊", "权重": 0.0, "说明": "与抑郁相反的情绪状态"},
    ])
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    st.info("""
    **风险分数计算公式**：
    ```
    risk(x) = Σ (P_i(x) × w_i)    其中 w_i 为上表权重
    ```
    """)

    st.divider()

    # 风险等级
    st.markdown("## 🚦 风险等级划分")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="padding: 1rem; background: #D1FAE5;
                    border-left: 4px solid #10B981; border-radius: 4px;">
            <div style="font-weight: 700; color: #059669;">🟢 低风险</div>
            <div style="font-size: 0.9rem; color: #6B7280;">0.00 ~ 0.25</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="padding: 1rem; background: #FEF3C7;
                    border-left: 4px solid #F59E0B; border-radius: 4px;">
            <div style="font-weight: 700; color: #D97706;">🟡 中低风险</div>
            <div style="font-size: 0.9rem; color: #6B7280;">0.25 ~ 0.50</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="padding: 1rem; background: #FED7AA;
                    border-left: 4px solid #F97316; border-radius: 4px;">
            <div style="font-weight: 700; color: #EA580C;">🟠 中高风险</div>
            <div style="font-size: 0.9rem; color: #6B7280;">0.50 ~ 0.75</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="padding: 1rem; background: #FEE2E2;
                    border-left: 4px solid #DC2626; border-radius: 4px;">
            <div style="font-weight: 700; color: #B91C1C;">🔴 高风险</div>
            <div style="font-size: 0.9rem; color: #6B7280;">0.75 ~ 1.00</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 使用说明
    st.markdown("## 📖 使用说明")
    with st.expander("🔧 本地启动"):
        st.code("""
# 1. 克隆仓库
git clone https://github.com/BeastOrange/sentiment_analysis.git
cd sentiment_analysis

# 2. 安装依赖
uv sync

# 3. 训练模型（如需）
uv run python scripts/run_all.py

# 4. 启动 Streamlit
uv run streamlit run app.py
        """, language="bash")

    with st.expander("📂 批量文件格式"):
        st.markdown("""
        **CSV 格式示例**：
        ```csv
        text,date
        今天心情很好,2026-01-01
        感觉好累啊,2026-01-02
        ```

        **TXT 格式示例**（每行一条）：
        ```
        今天心情很好
        感觉好累啊
        想一个人静静
        ```

        **列名识别规则**：
        - 优先使用 `text` 列
        - 次选 `content` / `review` / `评论` / `内容` / `文本`
        - 都不存在则取第一列
        """)

    st.divider()

    # 免责声明
    st.markdown("## ⚠️ 免责声明")
    st.warning("""
    本系统仅供**学术研究**与**教学演示**使用，不可用于临床诊断。

    - 抑郁倾向的判断应以**专业心理量表**（如 PHQ-9、CES-D）与临床医师诊断为准；
    - 模型使用的标签为公开微博情感语料的**弱代理标签**，未经临床验证；
    - 六情绪到抑郁风险的加权映射基于**心理学常识**，未经临床校准；
    - 若检测到高风险样本，建议用户寻求专业心理咨询或医疗帮助。

    **心理援助热线**：
    - 全国心理援助热线：**400-161-9995**
    - 北京心理危机研究与干预中心：**010-82951332**
    - 上海市心理援助热线：**021-12320-5**
    """)

    st.divider()

    # 引用与链接
    st.markdown("## 🔗 相关链接")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **GitHub 仓库**
        [https://github.com/BeastOrange/sentiment_analysis](https://github.com/BeastOrange/sentiment_analysis)

        **数据集**
        - [SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
        - [souljoy/COVID-19_weibo_emotion](https://huggingface.co/datasets/souljoy/COVID-19_weibo_emotion)
        """)
    with col2:
        st.markdown("""
        **技术栈**
        - PyTorch 2.3+ (MPS 后端)
        - HuggingFace Transformers
        - scikit-learn, pandas, jieba
        - Streamlit, Plotly

        **模型论文**
        - TextCNN: Kim 2014
        - BiLSTM-Attention: Yang et al. 2016
        - BERT: Devlin et al. 2019
        """)
