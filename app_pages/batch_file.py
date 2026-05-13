"""批量文件分析页面：上传 CSV/TXT，批量分析并下载结果。"""
from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference import predict_batch_binary, predict_batch_emotion, DEPRESSION_RISK


def _parse_uploaded(file, text_col: str | None) -> pd.DataFrame:
    """解析上传的文件，返回带 `text` 列的 DataFrame。"""
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    elif name.endswith(".txt"):
        content = file.read().decode("utf-8")
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        df = pd.DataFrame({"text": lines})
    else:
        raise ValueError(f"不支持的文件格式：{name}")

    if text_col and text_col in df.columns:
        df = df.rename(columns={text_col: "text"})
    elif "text" not in df.columns:
        candidates = [c for c in df.columns if c.lower() in ("text", "content", "review", "评论", "内容", "文本")]
        if candidates:
            df = df.rename(columns={candidates[0]: "text"})
        else:
            df = df.rename(columns={df.columns[0]: "text"})

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    return df


def render() -> None:
    st.markdown("# 📂 批量文件分析")
    st.caption("上传文件，批量识别抑郁倾向与情绪分布")
    st.divider()

    # 文件格式说明
    with st.container(border=True):
        st.markdown("### 📋 支持的文件格式")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **CSV 文件 (.csv)**
            - UTF-8 编码
            - 必须包含文本列
            - 列名可以是：`text` / `content` / `评论` / `内容` 等
            - 若无对应列名，默认取第一列
            """)
        with col2:
            st.markdown("""
            **Excel 文件 (.xlsx/.xls)**
            - 第一个工作表
            - 必须包含文本列
            - 列名规则同 CSV
            """)
        with col3:
            st.markdown("""
            **纯文本 (.txt)**
            - UTF-8 编码
            - 每行一条文本
            - 自动过滤空行
            """)

        st.info("💡 **示例 CSV**：`text,date\\n今天心情很好,2026-01-01\\n感觉好累,2026-01-02`")

    st.divider()

    # 上传 + 配置
    col_up, col_cfg = st.columns([2, 1])
    with col_up:
        file = st.file_uploader(
            "📤 上传文件",
            type=["csv", "xlsx", "xls", "txt"],
            help="最大 200MB，建议单文件不超过 5000 条记录",
        )

    with col_cfg:
        analysis_type = st.radio(
            "分析类型",
            options=["二分类（抑郁倾向）", "六情绪 + 抑郁风险", "完整分析"],
            help="完整分析包含两者，耗时最长",
        )

    text_col_input = None
    if file is not None and not file.name.lower().endswith(".txt"):
        # 预览列名供用户选择
        try:
            file.seek(0)
            if file.name.lower().endswith(".csv"):
                _df = pd.read_csv(file, nrows=5)
            else:
                _df = pd.read_excel(file, nrows=5)
            file.seek(0)

            text_col_input = st.selectbox(
                "📍 选择文本列",
                options=list(_df.columns),
                index=0,
            )

            with st.expander("👀 预览前 5 行"):
                st.dataframe(_df, use_container_width=True)
        except Exception as e:
            st.error(f"解析失败：{e}")
            return

    if file is None:
        st.info("⬆️ 请先上传文件")
        return

    run = st.button("🚀 开始批量分析", type="primary")
    if not run:
        return

    try:
        df = _parse_uploaded(file, text_col_input)
    except Exception as e:
        st.error(f"❌ 解析失败：{e}")
        return

    if len(df) == 0:
        st.warning("⚠️ 文件中没有有效文本")
        return

    n = len(df)
    st.success(f"✅ 已加载 {n} 条文本，开始分析...")

    progress = st.progress(0)
    status = st.empty()
    rows = []

    for i, row in df.iterrows():
        text = row["text"]
        out = {"text": text[:200] + ("..." if len(text) > 200 else "")}

        if "二分类" in analysis_type or "完整" in analysis_type:
            with st.spinner(""):
                bin_res = predict_batch_binary([text])[0]
            ens = bin_res["Ensemble (融合)"]
            out["抑郁倾向"] = ens["label"]
            out["抑郁置信度"] = round(ens["confidence"], 4)
            out["P(抑郁)"] = round(ens["prob"][1], 4)

        if "六情绪" in analysis_type or "完整" in analysis_type:
            emo_res = predict_batch_emotion([text])[0]
            out["主导情绪"] = emo_res["label"]
            out["情绪置信度"] = round(emo_res["confidence"], 4)
            out["抑郁风险分数"] = round(emo_res["depression_risk"], 4)
            for j, name in enumerate(emo_res["class_names"]):
                out[f"P({name})"] = round(emo_res["probs"][j], 4)

        rows.append(out)
        progress.progress((i + 1) / n)
        status.caption(f"进度：{i+1}/{n}")

    progress.empty()
    status.empty()

    result_df = pd.DataFrame(rows)

    st.divider()
    st.markdown(f"## 📊 分析结果（共 {n} 条）")

    # 统计概览
    if "抑郁倾向" in result_df.columns:
        depress_count = int((result_df["抑郁倾向"] == "抑郁倾向").sum())
        normal_count = n - depress_count
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总样本数", n)
        with col2:
            st.metric("抑郁倾向", depress_count, delta=f"{depress_count/n*100:.1f}%",
                      delta_color="inverse")
        with col3:
            st.metric("非抑郁倾向", normal_count, delta=f"{normal_count/n*100:.1f}%")

    # 风险分数分布
    if "抑郁风险分数" in result_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=result_df["抑郁风险分数"],
            nbinsx=20,
            marker_color="#EF4444",
            marker_line_color="white",
            marker_line_width=1,
        ))
        fig.update_layout(
            title="抑郁风险分数分布",
            xaxis_title="抑郁风险分数 (0~1)",
            yaxis_title="样本数",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 情绪分布饼图
    if "主导情绪" in result_df.columns:
        emo_counts = result_df["主导情绪"].value_counts()
        fig = go.Figure(go.Pie(
            labels=emo_counts.index,
            values=emo_counts.values,
            hole=0.4,
            marker=dict(line=dict(color="white", width=2)),
        ))
        fig.update_layout(
            title="主导情绪分布",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 结果表格
    st.markdown("### 📋 详细结果")
    st.dataframe(result_df, use_container_width=True, height=400)

    # 下载
    csv_buf = io.StringIO()
    result_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 下载结果 (CSV)",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name=f"analysis_result_{n}_rows.csv",
        mime="text/csv",
        type="primary",
    )
