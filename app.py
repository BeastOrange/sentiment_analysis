"""Streamlit 主应用：中文情感分析与抑郁倾向识别系统。

启动方式：
    uv run streamlit run app.py

功能：
1. 单文本分析：粘贴一段文本，输出抑郁倾向 + 六情绪 + 风险分数
2. 批量文件分析：上传 CSV/TXT 文件，批量分析并下载结果
3. 模型对比：对比 TextCNN / BiLSTM / BERT / Ensemble 在同一文本上的差异
4. 关于项目：数据集、模型架构、性能指标说明
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from app_pages import single_text, batch_file, model_compare, about

# 页面配置
st.set_page_config(
    page_title="抑郁倾向识别系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    # 侧边栏
    with st.sidebar:
        st.markdown("# 🧠 抑郁倾向识别")
        st.caption("基于社交媒体中文文本的情感分析与抑郁倾向识别系统")
        st.divider()

        page = st.radio(
            "功能选择",
            options=[
                "📝 单文本分析",
                "📂 批量文件分析",
                "⚖️ 模型对比",
                "ℹ️ 关于项目",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### 📊 模型性能")
        st.markdown(
            """
            **二分类（测试集）**
            - 融合模型：F1 **0.983**
            - BiLSTM-Attn：F1 0.981
            - TextCNN：F1 0.976
            - BERT：F1 0.967

            **六情绪（测试集）**
            - BiLSTM-Attn：F1 **0.853**
            """
        )

        st.divider()
        st.caption("⚠️ 仅供学术研究，不用于临床诊断")

    # 路由
    if page == "📝 单文本分析":
        single_text.render()
    elif page == "📂 批量文件分析":
        batch_file.render()
    elif page == "⚖️ 模型对比":
        model_compare.render()
    elif page == "ℹ️ 关于项目":
        about.render()


if __name__ == "__main__":
    main()
