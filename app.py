import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="FL XAI Audit Hub", layout="wide", page_icon="🔬")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "federated_audit.db")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def check_db_tables(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('fl_metrics', 'global_metrics')")
        tables = [r[0] for r in cursor.fetchall()]
        return 'fl_metrics' in tables and 'global_metrics' in tables
    except Exception:
        return False

def load_data():
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        return None, None
    try:
        conn = sqlite3.connect(DB_PATH)
        if not check_db_tables(conn):
            conn.close()
            return None, None
        df_client = pd.read_sql_query("SELECT * FROM fl_metrics ORDER BY round, client_id", conn)
        df_global = pd.read_sql_query("SELECT * FROM global_metrics ORDER BY round", conn)
        conn.close()
        if df_client.empty or df_global.empty:
            return None, None
        return df_client, df_global
    except Exception:
        return None, None

st.title("🔬 Federated Learning XAI Audit Dashboard")
st.markdown("### Four Analytical Representations of Federated Convergence & Explainability Drift")

df_client, df_global = load_data()

if df_client is None or df_global is None:
    st.warning("⚠️ No populated audit database found at `" + DB_PATH + "`.")
    st.info("You can generate the benchmark dataset directly using the button below or by running `python main.py` in your terminal.")
    
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🚀 Run FL Benchmark Pipeline Now", type="primary"):
            with st.spinner("Executing 15 FL rounds and XAI auditing..."):
                from main import run_federated_pipeline
                from visualize import generate_all_visualizations
                run_federated_pipeline(num_rounds=15, num_clients=5)
                generate_all_visualizations()
                st.success("✅ Execution complete! Reloading dashboard...")
                st.rerun()
else:
    # Sidebar control
    st.sidebar.header("⚙️ Audit Controls")
    st.sidebar.info(f"📁 **Database**: `{os.path.basename(DB_PATH)}`\n\n🔄 **Total Rounds**: {int(df_global['round'].max())}\n\n👥 **Clients**: {len(df_client['client_id'].unique())}")
    
    if st.sidebar.button("🔄 Re-run FL Protocol"):
        with st.spinner("Re-executing FL pipeline..."):
            from main import run_federated_pipeline
            from visualize import generate_all_visualizations
            run_federated_pipeline(num_rounds=15, num_clients=5)
            generate_all_visualizations()
            st.success("Refreshed!")
            st.rerun()

    # Top KPI Metrics Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Executed Rounds", int(df_global["round"].max()))
    k2.metric("Final Mean Loss", f"{df_global['avg_loss'].iloc[-1]:.4f}")
    k3.metric("Final Mean Accuracy", f"{df_global['avg_acc'].iloc[-1]:.2%}")
    k4.metric("Mean XAI Cosine Sim", f"{df_global['mean_cosine_sim'].iloc[-1]:.4f}")
    k5.metric("Mean KL Divergence", f"{df_global['mean_kl_div'].iloc[-1]:.4f}")

    st.markdown("---")

    # 4 Main Visualization Tabs + Overview
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 1. Convergence Dynamics", 
        "📊 2. Explainability & Drift Audit", 
        "🔥 3. Client Consistency Matrix", 
        "📉 4. SHAP Feature Attribution Spectrum",
        "🖼️ All-In-One Overview"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Convergence Dynamics
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Representation 1: Federated Learning Convergence & Performance Dynamics")
        st.markdown("Tracks global loss minimization alongside classification accuracy improvement across all federated clients.")
        
        c1, c2 = st.columns(2)
        with c1:
            fig_loss = go.Figure()
            for cid in sorted(df_client["client_id"].unique()):
                c_data = df_client[df_client["client_id"] == cid]
                fig_loss.add_trace(go.Scatter(x=c_data["round"], y=c_data["train_loss"],
                                              mode="lines", name=f"Client {cid}", opacity=0.35, line=dict(dash="dot")))
            fig_loss.add_trace(go.Scatter(x=df_global["round"], y=df_global["avg_loss"],
                                          mode="lines+markers", name="Global Mean Loss",
                                          line=dict(color="#1f77b4", width=3.5)))
            fig_loss.update_layout(title="A. Training Loss Convergence (Client vs Global)",
                                   xaxis_title="Federated Round", yaxis_title="Cross-Entropy Loss",
                                   template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_loss, use_container_width=True)

        with c2:
            fig_acc = go.Figure()
            for cid in sorted(df_client["client_id"].unique()):
                c_data = df_client[df_client["client_id"] == cid]
                fig_acc.add_trace(go.Scatter(x=c_data["round"], y=c_data["train_acc"] * 100,
                                             mode="lines", name=f"Client {cid}", opacity=0.35, line=dict(dash="dot")))
            fig_acc.add_trace(go.Scatter(x=df_global["round"], y=df_global["avg_acc"] * 100,
                                         mode="lines+markers", name="Global Mean Accuracy",
                                         line=dict(color="#2ca02c", width=3.5)))
            fig_acc.update_layout(title="B. Classification Accuracy Trajectory (%)",
                                  xaxis_title="Federated Round", yaxis_title="Accuracy (%)",
                                  template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_acc, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Explainability & Drift Audit
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Representation 2: Explainability Consistency & Concept Drift Tracking")
        st.markdown("Monitors directional attribution fidelity (**Cosine Similarity**) and distributional entropy drift (**KL Divergence**) relative to the global model.")

        c1, c2 = st.columns(2)
        with c1:
            fig_cos = go.Figure()
            for cid in sorted(df_client["client_id"].unique()):
                c_data = df_client[df_client["client_id"] == cid]
                fig_cos.add_trace(go.Scatter(x=c_data["round"], y=c_data["cosine_similarity"],
                                             mode="lines+markers", name=f"Client {cid}", opacity=0.6))
            fig_cos.add_trace(go.Scatter(x=df_global["round"], y=df_global["mean_cosine_sim"],
                                         mode="lines+markers", name="Mean Cosine Alignment",
                                         line=dict(color="#d62728", width=3.5, dash="solid")))
            fig_cos.update_layout(title="A. SHAP Feature Attribution Alignment (Cosine Similarity ↑)",
                                  xaxis_title="Federated Round", yaxis_title="Cosine Similarity",
                                  yaxis_range=[0.65, 1.02], template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_cos, use_container_width=True)

        with c2:
            fig_kl = go.Figure()
            for cid in sorted(df_client["client_id"].unique()):
                c_data = df_client[df_client["client_id"] == cid]
                fig_kl.add_trace(go.Scatter(x=c_data["round"], y=c_data["kl_divergence"],
                                            mode="lines+markers", name=f"Client {cid}", opacity=0.6))
            fig_kl.add_trace(go.Scatter(x=df_global["round"], y=df_global["mean_kl_div"],
                                        mode="lines+markers", name="Mean KL Divergence",
                                        line=dict(color="#9467bd", width=3.5)))
            fig_kl.update_layout(title="B. Explanation Distributional Divergence (Kullback-Leibler Drift ↓)",
                                  xaxis_title="Federated Round", yaxis_title="KL Divergence (nats)",
                                  template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_kl, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Client Consistency Heatmap
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Representation 3: Multi-Client Explainability Fidelity Matrix & Disparity Profile")
        st.markdown("2D Matrix mapping cosine consistency across all clients and training rounds, highlighting non-IID drift discrepancies.")

        pivot_cos = df_client.pivot(index="client_id", columns="round", values="cosine_similarity")
        fig_heat = px.imshow(pivot_cos,
                             labels=dict(x="Federated Round", y="Client ID", color="Cosine Sim"),
                             x=pivot_cos.columns, y=pivot_cos.index,
                             text_auto=".2f", aspect="auto", color_continuous_scale="Viridis")
        fig_heat.update_layout(title="Client-Wise Cosine Consistency Heatmap",
                               xaxis_title="Federated Round", yaxis_title="Client ID",
                               template="plotly_white")
        st.plotly_chart(fig_heat, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: SHAP Feature Attribution Spectrum
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Representation 4: SHAP Feature Attribution Spectrum & Local-vs-Global Discrepancy")
        st.markdown("Inspects granular feature attribution importance across all 10 dataset features for each client at any chosen FL round.")

        selected_round = st.slider("Select FL Round for Feature Attribution Snapshot", 
                                   1, int(df_global["round"].max()), int(df_global["round"].max()))
        
        round_data = df_client[df_client["round"] == selected_round]
        shap_matrix = np.array([json.loads(x) for x in round_data["shap_values"]])
        num_clients, num_features = shap_matrix.shape
        feature_names = [f"Feat_{i}" for i in range(num_features)]

        c3, c4 = st.columns(2)
        with c3:
            fig_shap_bar = go.Figure()
            for idx, client_shap in enumerate(shap_matrix):
                fig_shap_bar.add_trace(go.Bar(name=f"Client {idx}", x=feature_names, y=client_shap))
            # Consensus Mean
            fig_shap_bar.add_trace(go.Scatter(name="Consensus Mean", x=feature_names, y=np.mean(shap_matrix, axis=0),
                                              mode="lines+markers", line=dict(color="black", width=2, dash="dash")))
            fig_shap_bar.update_layout(barmode='group', title=f"A. Clustered Feature Attributions (Round {selected_round})",
                                       xaxis_title="Feature Index", yaxis_title="Mean |SHAP Value|",
                                       template="plotly_white")
            st.plotly_chart(fig_shap_bar, use_container_width=True)

        with c4:
            df_shap = pd.DataFrame(shap_matrix, index=[f"Client {i}" for i in range(num_clients)], columns=feature_names)
            fig_shap_heat = px.imshow(df_shap, text_auto=".3f", aspect="auto", color_continuous_scale="YlOrRd",
                                      labels=dict(x="Feature Index", y="Client ID", color="SHAP Value"))
            fig_shap_heat.update_layout(title=f"B. Feature Attribution Heatmap (Round {selected_round})",
                                        template="plotly_white")
            st.plotly_chart(fig_shap_heat, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: All-In-One Overview
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("Combined 4-in-1 Publication-Quality Overview")
        overview_img = os.path.join(PLOTS_DIR, "combined_dashboard_4in1.png")
        if os.path.exists(overview_img):
            st.image(overview_img, use_container_width=True)
        else:
            st.info("Run `python visualize.py` to compile high-resolution PNG exports.")
