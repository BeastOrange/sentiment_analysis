"""模型对比页面：同一文本在 4 个模型上的预测差异。"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.inference import predict_binary


def render() -> None:
    st.markdown("# ⚖️ 模型对比")
    st.caption("对比 TextCNN、BiLSTM-Attention、BERT 与融合模型在同一文本上的预测差异")
    st.divider()

    examples = {
        "边界样本（含混合情绪）": "今天又去看了医生，虽然医生说没什么大问题，但是我心里还是很担心，不知道该怎么办",
        "明显抑郁文本": "一个人在房间里好几天了，不想见任何人，感觉活着没意思",
        "明显正常文本": "刚刚跟朋友吃了一顿超级棒的火锅，真开心！",
        "反讽/讽刺文本": "哈哈哈真是太棒了呢，今天又是加班到凌晨的一天，开心死了",
    }

    selected = st.selectbox("💡 示例文本", options=["自定义输入"] + list(examples.keys()))
    default = examples.get(selected, "")
    text = st.text_area("输入文本", value=default, height=100, max_chars=500)

    if not st.button("🔍 对比分析", type="primary"):
        return

    if not text.strip():
        st.warning("⚠️ 请输入有效文本")
        return

    with st.spinner("🤖 4 个模型依次推理中..."):
        results = predict_binary(text)

    st.divider()

    # 概览卡片
    st.markdown("### 📊 各模型预测概览")
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        is_depress = res["label_idx"] == 1
        bg_color = "#FEE2E2" if is_depress else "#D1FAE5"
        fg_color = "#DC2626" if is_depress else "#059669"
        with col:
            st.markdown(
                f"""
                <div style="padding: 1rem; background: {bg_color};
                            border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.85rem; color: #6B7280;">{name}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: {fg_color};
                                margin: 0.3rem 0;">{res['label']}</div>
                    <div style="font-size: 0.85rem; color: #6B7280;">
                        置信度 {res['confidence']*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # 概率分布对比
    st.markdown("### 📈 各模型概率分布")
    model_names = list(results.keys())
    p_normal = [results[n]["prob"][0] for n in model_names]
    p_depress = [results[n]["prob"][1] for n in model_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="非抑郁倾向",
        x=model_names,
        y=p_normal,
        marker_color="#10B981",
        text=[f"{v*100:.2f}%" for v in p_normal],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="抑郁倾向",
        x=model_names,
        y=p_depress,
        marker_color="#EF4444",
        text=[f"{v*100:.2f}%" for v in p_depress],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        height=450,
        yaxis=dict(range=[0, 1.15], tickformat=".0%", title="概率"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 详细表格
    st.markdown("### 📋 详细数据")
    import pandas as pd
    table_data = []
    for name, res in results.items():
        table_data.append({
            "模型": name,
            "预测标签": res["label"],
            "置信度": f"{res['confidence']*100:.2f}%",
            "P(非抑郁)": f"{res['prob'][0]*100:.2f}%",
            "P(抑郁)": f"{res['prob'][1]*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # 一致性分析
    st.divider()
    st.markdown("### 🎯 一致性分析")
    preds = [r["label_idx"] for r in results.values()]
    if len(set(preds)) == 1:
        st.success("✅ **完全一致**：所有模型都判断为同一类别，预测可信度高")
    else:
        agree_count = sum(1 for p in preds if p == preds[-1])
        st.warning(
            f"⚠️ **预测分歧**：{len(preds)} 个模型中有 {agree_count} 个支持融合模型的结论，"
            f"建议结合抑郁风险分数综合判断"
        )

    # 融合权重说明
    with st.expander("ℹ️ 融合权重说明"):
        st.markdown("""
        **融合权重** (基于验证集 F1 网格搜索)：
        - TextCNN: **0.0**
        - BiLSTM-Attention: **0.8**
        - BERT: **0.2**

        **为什么 TextCNN 权重为 0？**
        BiLSTM-Attention 在所有指标（准确率、召回率、F1、AUC）上都严格优于 TextCNN，
        加入 TextCNN 会稀释 BiLSTM 的判别面。这并不意味着 TextCNN 没价值——
        在其他随机种子或小数据集下，TextCNN 仍是稳健的 baseline。

        **融合公式**：
        ```
        P_ensemble = 0.8 × P_BiLSTM + 0.2 × P_BERT
        ```
        """)
