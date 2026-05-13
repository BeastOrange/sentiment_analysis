"""单文本分析页面：输入一段文本，输出抑郁倾向 + 六情绪 + 风险分数。"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.inference import predict_all

EMOTION_COLORS = {
    "中性": "#9CA3AF",
    "高兴": "#10B981",
    "愤怒": "#EF4444",
    "悲伤": "#3B82F6",
    "恐惧": "#8B5CF6",
    "惊奇": "#F59E0B",
}

RISK_LEVEL = [
    (0.0, 0.25, "低风险", "#10B981"),
    (0.25, 0.5, "中低风险", "#F59E0B"),
    (0.5, 0.75, "中高风险", "#F97316"),
    (0.75, 1.01, "高风险", "#DC2626"),
]


def get_risk_level(score: float) -> tuple[str, str]:
    for lo, hi, name, color in RISK_LEVEL:
        if lo <= score < hi:
            return name, color
    return "高风险", "#DC2626"


def render() -> None:
    st.markdown("# 📝 单文本分析")
    st.caption("输入一段中文社交媒体文本，系统将分析其情绪特征与抑郁倾向")
    st.divider()

    # 示例
    examples = {
        "🟢 正常情绪": "今天天气真好，和朋友一起去公园散步，心情特别舒畅！",
        "🔴 疑似抑郁": "最近感觉好累啊，什么都不想做，一个人躲在房间里，感觉活着没什么意思",
        "😡 愤怒情绪": "真是气死我了，这种事情怎么会发生，太过分了！",
        "😢 悲伤情绪": "想到那些失去的回忆，眼泪就止不住地流下来，心里空落落的",
        "😨 恐惧情绪": "疫情期间每天都很担心，害怕自己和家人被感染，夜里睡不着觉",
    }

    col_ex, _ = st.columns([3, 1])
    with col_ex:
        selected = st.selectbox(
            "💡 选择示例文本（可选）",
            options=["自定义输入"] + list(examples.keys()),
        )

    default_text = examples.get(selected, "")
    text = st.text_area(
        "输入文本",
        value=default_text,
        height=120,
        placeholder="请粘贴一段中文微博/小红书评论...",
        max_chars=500,
    )

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        run = st.button("🔍 开始分析", type="primary", use_container_width=True)

    if not run:
        st.info("⬆️ 请输入文本并点击「开始分析」")
        return

    if not text.strip():
        st.warning("⚠️ 请输入有效文本")
        return

    with st.spinner("🤖 模型推理中（首次加载约需 20-30 秒）..."):
        result = predict_all(text)

    st.divider()

    # 顶层结论卡片：综合判断（二分类融合 + 六情绪风险 + BERT 双校验）
    ens = result["binary"]["Ensemble (融合)"]
    bert_res = result["binary"]["BERT"]
    emo = result["emotion"]
    risk = emo["depression_risk"]
    risk_name, risk_color = get_risk_level(risk)

    # 综合判断逻辑：避免单模型误判
    # 规则：
    #   - 若融合预测 = 抑郁，但六情绪风险 < 0.25 且 BERT 也判非抑郁，
    #     则采纳"非抑郁"结论并提示模型分歧
    #   - 反之亦然
    ens_pred_depress = ens["label_idx"] == 1
    bert_pred_depress = bert_res["label_idx"] == 1
    risk_says_depress = risk >= 0.5

    # 多数投票（融合 / BERT / 六情绪风险）
    votes = [ens_pred_depress, bert_pred_depress, risk_says_depress]
    depress_count = sum(votes)
    final_is_depress = depress_count >= 2

    # 综合置信度：三者一致时较高，分歧时较低
    if depress_count == 0:
        final_conf = (bert_res["prob"][0] + ens["prob"][0] + (1 - risk)) / 3
        final_label = "非抑郁倾向"
    elif depress_count == 3:
        final_conf = (bert_res["prob"][1] + ens["prob"][1] + risk) / 3
        final_label = "抑郁倾向"
    else:
        # 分歧时取较保守的结论
        final_label = "抑郁倾向" if final_is_depress else "非抑郁倾向"
        final_conf = max(votes.count(final_is_depress) / 3, 0.5)

    has_disagreement = depress_count not in (0, 3)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="综合判断",
            value=final_label,
            delta=f"置信度 {final_conf*100:.1f}%",
            delta_color="inverse" if final_is_depress else "normal",
        )
    with col2:
        st.metric(
            label="主导情绪",
            value=emo["label"],
            delta=f"置信度 {emo['confidence']*100:.1f}%",
        )
    with col3:
        st.metric(
            label="抑郁风险分数",
            value=f"{risk:.3f}",
            delta="范围 0 ~ 1",
        )
    with col4:
        st.markdown(
            f"""
            <div style="padding: 1rem; background: {risk_color}22;
                        border-left: 4px solid {risk_color}; border-radius: 4px;">
                <div style="font-size: 0.9rem; color: #6B7280;">风险等级</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {risk_color};">
                    {risk_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 分歧提示
    if has_disagreement:
        st.warning(
            f"⚠️ **模型存在分歧**：二分类融合预测「{ens['label']}」，"
            f"BERT 预测「{bert_res['label']}」，六情绪风险分数 {risk:.3f}。"
            f"系统采用**多数投票 + 六情绪交叉验证**给出综合判断「{final_label}」。"
            f"建议查看下方各模型详细概率，结合上下文人工判断。"
        )

    # 综合判断逻辑说明（可折叠）
    with st.expander("🔬 综合判断逻辑说明"):
        st.markdown("""
        **为什么需要综合判断？**

        单个模型（即使是融合模型）在某些短文本上可能因训练数据偏差而误判。
        例如：BiLSTM 模型对某些简短的正面表达可能过度敏感，给出较高的抑郁概率。

        **综合判断采用三方投票**：
        1. **二分类融合模型**（BiLSTM 0.8 + BERT 0.2）的预测
        2. **BERT 模型**的独立预测（最稳健的预训练模型）
        3. **六情绪风险分数** ≥ 0.5 视为高风险

        三者中至少两个支持"抑郁倾向"才最终判定为抑郁，否则判为非抑郁。
        这种**多数投票 + 跨任务交叉验证**机制能显著降低单模型误判风险。
        """)

    st.divider()

    # 详细结果：Tab
    tab1, tab2, tab3 = st.tabs(["📊 模型预测", "🎭 情绪分布", "🔍 文本预处理"])

    with tab1:
        st.markdown("#### 各模型二分类预测结果")

        model_names = []
        prob_normal = []
        prob_depress = []
        conf_labels = []

        for name, res in result["binary"].items():
            model_names.append(name)
            prob_normal.append(res["prob"][0])
            prob_depress.append(res["prob"][1])
            conf_labels.append(res["label"])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="非抑郁倾向",
            x=model_names,
            y=prob_normal,
            marker_color="#10B981",
            text=[f"{v*100:.1f}%" for v in prob_normal],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="抑郁倾向",
            x=model_names,
            y=prob_depress,
            marker_color="#EF4444",
            text=[f"{v*100:.1f}%" for v in prob_depress],
            textposition="outside",
        ))
        fig.update_layout(
            barmode="group",
            height=400,
            yaxis=dict(range=[0, 1.15], tickformat=".0%", title="概率"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 模型一致性提示
        preds = {n: r["label_idx"] for n, r in result["binary"].items()}
        unique_preds = set(preds.values())
        if len(unique_preds) == 1:
            st.success(f"✅ 所有模型一致判断为「{list(result['binary'].values())[0]['label']}」")
        else:
            # 列出每个模型的判断
            details = " · ".join(
                f"{n}: {r['label']}({r['confidence']*100:.0f}%)"
                for n, r in result["binary"].items()
            )
            st.warning(f"⚠️ 模型预测存在分歧：{details}")
            st.caption(
                "💡 由于训练数据中的偏差，单一模型在某些短文本上可能出现误判。"
                "建议参考顶部的「综合判断」和六情绪风险分数，进行综合考量。"
            )

    with tab2:
        st.markdown("#### 六情绪分布")
        probs = emo["probs"]
        class_names = emo["class_names"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=class_names,
            y=probs,
            marker_color=[EMOTION_COLORS[n] for n in class_names],
            text=[f"{v*100:.1f}%" for v in probs],
            textposition="outside",
        ))
        fig.update_layout(
            height=400,
            yaxis=dict(range=[0, max(probs) * 1.2], tickformat=".0%", title="概率"),
            xaxis=dict(title=""),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 风险贡献分解
        st.markdown("#### 抑郁风险贡献分解")
        st.caption("每种情绪对抑郁风险的贡献 = 情绪概率 × 心理学权重")

        from src.inference import DEPRESSION_RISK
        contrib = [probs[i] * DEPRESSION_RISK[i] for i in range(6)]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=class_names,
            y=contrib,
            marker_color=[EMOTION_COLORS[n] for n in class_names],
            text=[f"{v:.3f}" for v in contrib],
            textposition="outside",
        ))
        fig2.update_layout(
            height=350,
            yaxis=dict(title="风险贡献"),
            xaxis=dict(title=""),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.info(
            "💡 **心理学权重**：悲伤 1.0 · 恐惧 0.7 · 愤怒 0.5 · 中性 0.2 · 惊奇 0.1 · 高兴 0.0"
        )

    with tab3:
        st.markdown("#### 原始文本")
        st.code(text, language=None)

        st.markdown("#### 清洗后文本")
        st.code(result["cleaned_text"], language=None)

        with st.expander("🔧 清洗规则说明"):
            st.markdown("""
            - 移除 URL（`https?://...`）
            - 移除 @用户名
            - 移除 #话题#
            - 压缩过度重复字符（如 "哈哈哈哈哈" → "哈哈哈"）
            - 清理中文方括号、书名号
            - 合并多余空白
            """)
