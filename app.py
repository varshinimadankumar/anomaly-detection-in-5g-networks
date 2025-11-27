import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
st.set_option("client.showErrorDetails", False)
st.set_page_config(
    page_title="5G Anomaly Detection Dashboard",
    layout="wide",
    page_icon="📡"
)

API_URL = "http://localhost:8000/predict"

st.title("📡 5G Network Anomaly Detection — Full Dashboard")
st.markdown("---")

uploaded = st.file_uploader("Upload your features_clean/labeled_with_metadata parquet file:", type=["parquet"])

if uploaded:
    df = pd.read_parquet(uploaded)
    st.subheader("Select number of rows for anomaly detection")

    max_rows = len(df)
    n_rows = st.slider("Rows to analyze", 1, max_rows, 50)

    df_selected = df.head(n_rows)
    st.write(f"Analyzing first **{n_rows}** rows")

    # Remove old score columns to avoid duplicates
    score_cols = ["ae_score", "ocsvm_score", "rf_score", "gnn_score", "fused_score"]
    df_selected = df_selected.drop(columns=[c for c in score_cols if c in df.columns], errors="ignore")

    st.success(f"Loaded {len(df_selected)} windows")

    st.markdown("### 🔎 Preview of Input Data")
    st.dataframe(df_selected.head())

    st.markdown("### 🚀 Run Inference")

    if st.button("Run Detection"):
        with st.spinner("Contacting API and scoring windows..."):
            data_json = df_selected.to_dict(orient="records")
            response = requests.post(API_URL, json={"data": data_json})
            st.write("RAW RESPONSE:", response.text)
            result = response.json()
            scores = pd.DataFrame(result["scores"])

        st.success("Detection completed.")

        # Merge scores with metadata
        scored = pd.concat([df_selected.reset_index(drop=True), scores], axis=1)

        # 1. FUSED SCORE TREND
        st.markdown("## 📈 Fused Anomaly Score Over Time")
        if "time" in scored.columns:
            fig = px.line(
                scored,
                x="time",
                y="fused_score",
                title="Fused Score Timeline",
                markers=True
            )
        else:
            scored["index"] = scored.index
            fig = px.line(
                scored,
                x="index",
                y="fused_score",
                title="Fused Score Timeline",
                markers=True
            )

        st.plotly_chart(fig, use_container_width=True)

        # 2. SCORE PANEL
        st.markdown("## 🧠 Model Score Comparison")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=scored["ae_score"], mode="lines", name="AE"))
        fig2.add_trace(go.Scatter(y=scored["ocsvm_score"], mode="lines", name="OCSVM"))
        fig2.add_trace(go.Scatter(y=scored["rf_score"], mode="lines", name="RF"))
        fig2.add_trace(go.Scatter(y=scored.get("gnn_score", np.zeros(len(scored))), mode="lines", name="GNN"))
        fig2.add_trace(go.Scatter(y=scored["fused_score"], mode="lines", name="Fused", line=dict(width=4)))

        fig2.update_layout(title="Model Score Trends", xaxis_title="Index", yaxis_title="Score")
        st.plotly_chart(fig2, use_container_width=True)

        # 3. TOP ANOMALY WINDOWS TABLE
        st.markdown("## 🔥 Top 50 Most Suspicious Windows")
        top50 = scored.sort_values("fused_score", ascending=False).head(50)
        st.dataframe(top50)

        # 4. ANOMALY BY SOURCE IP
        if "source" in scored.columns:
            st.markdown("## 🌐 Anomaly Breakdown by Source IP")
            src_scores = scored.groupby("source")["fused_score"].mean().sort_values(ascending=False)
            fig3 = px.bar(
                src_scores.head(20),
                title="Top Suspicious Source Hosts",
                labels={"value": "Avg Fused Score", "source": "Source"}
            )
            st.plotly_chart(fig3, use_container_width=True)

        # 5. ANOMALY BY PROTOCOL
        if "protocol" in scored.columns:
            st.markdown("## 📡 Protocol Anomaly Distribution")
            proto_scores = scored.groupby("protocol")["fused_score"].mean()
            fig4 = px.bar(
                proto_scores.sort_values(ascending=False),
                title="Protocols vs Avg Anomaly Score",
                labels={"value": "Avg Fused Score", "protocol": "Protocol"}
            )
            st.plotly_chart(fig4, use_container_width=True)

        # 6. DRILL-DOWN ANALYSIS
        st.markdown("## 🕵️ Drill Down Into a Single Window")
        idx = st.number_input("Enter window index", min_value=0, max_value=len(scored) - 1, value=0)
        st.json(scored.iloc[idx].to_dict())

else:
    st.info("Upload a parquet file to begin.")
