import warnings
warnings.filterwarnings('ignore')

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ML/Forecasting modules (place these under modules/ as provided)
from modules.etl_sql import load_crm_tables
from modules.features import attach_engagement_and_values
from modules.ml_leads import train_lead_model, score_leads, attach_nba
from modules.ml_calls import daily_call_series, forecast_calls, optimal_call_windows
from modules.ml_tasks import train_sla_model, score_sla
from modules.ml_agents import availability_heatmap
from modules.kpi_conversion import propensity_weighted_pipeline
from modules.geo_priority import market_priority

# ---------------- Page config & CSS ----------------
st.set_page_config(
    page_title="Executive CRM Dashboard - Advanced Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1419; color: #ffffff; }
    .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%); }
    .metric-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        padding: 16px; border-radius: 12px; border: 1px solid #4a5568;
        margin: 6px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .metric-label { font-size: 0.85rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .metric-value { font-size: 2.0rem; font-weight: 700; color: #f59e0b; }
    .dashboard-header {
        background: linear-gradient(90deg, #1a202c 0%, #2d3748 100%);
        padding: 20px; border-radius: 12px; margin-bottom: 18px; border: 1px solid #4a5568;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def metric_card(title, value, fmt="number"):
    if fmt == "currency" and isinstance(value, (int, float)):
        display_value = f"${value:,.0f}"
    elif fmt == "percentage" and isinstance(value, (int, float)):
        display_value = f"{value:.1f}%"
    elif isinstance(value, (int, float)):
        display_value = f"{value:,.0f}" if value >= 1000 else f"{value:.1f}"
    else:
        display_value = str(value)
    return f"""
    <div class="metric-container">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{display_value}</div>
    </div>
    """

@st.cache_data(show_spinner=False)
def load_data(conn_str: str):
    d = load_crm_tables(conn_str)
    leads = attach_engagement_and_values(d['leads'], d['calls'])
    # LeadVelocity: days since CreatedOn (for cycle-time signals)
    if 'CreatedOn' in leads.columns:
        leads['CreatedOn'] = pd.to_datetime(leads['CreatedOn'], errors='coerce')
        leads['LeadVelocity'] = (pd.Timestamp.now().normalize() - leads['CreatedOn']).dt.days.clip(lower=0)
    return leads, d['calls'], d['tasks']

@st.cache_resource(show_spinner=False)
def train_models(leads_labeled: pd.DataFrame, tasks_labeled: pd.DataFrame):
    lead_model, auc = train_lead_model(leads_labeled, label_col='Converted')
    sla_model = train_sla_model(tasks_labeled, label_col='IsBreach')
    return lead_model, auc, sla_model

@st.cache_data(show_spinner=False)
def infer_all(lead_model, sla_model, leads_current, calls_current, tasks_current):
    # Lead scoring + NBA
    scored = attach_nba(score_leads(lead_model, leads_current))
    # Calls forecasting + optimal hours
    ds = daily_call_series(calls_current, date_col='CallDateTime')
    fc = forecast_calls(ds, periods=30)
    best_hours = optimal_call_windows(calls_current)
    # SLA breach risk
    tasks_scored = score_sla(sla_model, tasks_current)
    # Conversion KPIs
    conv_kpi, top10 = propensity_weighted_pipeline(scored)
    # Geo priority
    geo = market_priority(scored)
    return scored, fc, best_hours, tasks_scored, conv_kpi, top10, geo

# ---------------- Main ----------------
def main():
    st.markdown("""
    <div class="dashboard-header">
        <h2 style="color: #f59e0b; margin: 0;">🚀 Executive CRM Dashboard</h2>
        <p style="color: #a0aec0; margin: 6px 0 0 0;">AI‑Powered Real Estate Analytics with Predictive Insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("🎛️ Controls")
    conn_str = st.secrets.get("mssql", {}).get("conn_str", os.getenv("MSSQL_CONN_STR", ""))
    if not conn_str:
        st.sidebar.warning("No SQL connection string found in secrets. Set .streamlit/secrets.toml or MSSQL_CONN_STR.")
    horizon = st.sidebar.slider("📈 Forecast Horizon (days)", 7, 60, 30)
    agents_slider = st.sidebar.slider("👥 Agents staffed", 5, 60, 15)

    # Load, train, infer
    leads_df, calls_df, schedule_df = load_data(conn_str) if conn_str else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if leads_df.empty or calls_df.empty or schedule_df.empty:
        st.error("Required tables not loaded. Check SQL connection and table names.")
        st.stop()

    lead_model, auc, sla_model = train_models(leads_df, schedule_df)
    scored, fc, best_hours, tasks_scored, conv_kpi, top10, geo = infer_all(
        lead_model, sla_model, leads_df, calls_df, schedule_df
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Executive Overview",
        "Lead Status",
        "AI Call Activity",
        "Follow-up & Tasks",
        "Agent Availability",
        "Conversion",
        "Geographic",
        "AI Command Center"
    ])

    # -------- Executive Overview --------
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Lead Model AUC", float(auc), "number"), unsafe_allow_html=True)
        with c2:
            total_pipeline = float(pd.to_numeric(scored.get('RevenuePotential', 0), errors='coerce').sum())
            st.markdown(metric_card("Pipeline Value", total_pipeline, "currency"), unsafe_allow_html=True)
        with c3:
            pw = float(conv_kpi['ExpectedRevenuePW'].iloc)
            st.markdown(metric_card("30‑Day PW Revenue", pw, "currency"), unsafe_allow_html=True)
        with c4:
            eff = float(conv_kpi['PipelineEfficiencyPct'].iloc)
            st.markdown(metric_card("Pipeline Efficiency", eff, "percentage"), unsafe_allow_html=True)

        st.subheader("Top Opportunities (AI‑Ranked)")
        cols = ['FullName','RevenuePotential','PropensityToConvert','NBA_Action','NBA_Priority','NBA_Confidence','NBA_Reason']
        show_cols = [c for c in cols if c in scored.columns]
        st.dataframe(scored[show_cols].sort_values('PropensityToConvert', ascending=False).head(15), use_container_width=True)

    # -------- Lead Status --------
    with tab2:
        st.subheader("Lead Status Breakdown (Pie)")
        # Prefer LeadStatus if available, else map LeadStage
        if 'StatusName_E' in scored.columns:
            status_counts = scored['StatusName_E'].fillna('Unknown').value_counts()
            fig = go.Figure(data=[go.Pie(labels=status_counts.index, values=status_counts.values, hole=0.35)])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        elif 'LeadStageId' in scored.columns:
            stage_map = {1:'New', 2:'In Progress', 3:'Interested', 4:'Closed'}
            stage_counts = scored['LeadStageId'].map(stage_map).fillna('Unknown').value_counts()
            fig = go.Figure(data=[go.Pie(labels=stage_counts.index, values=stage_counts.values, hole=0.35)])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No status/stage columns found to render pie.")

        st.subheader("AI Propensity & Next Best Actions")
        cols = ['FullName','LeadStageId','PropensityToConvert','NBA_Action','NBA_Priority','NBA_Confidence','NBA_Reason']
        show_cols = [c for c in cols if c in scored.columns]
        st.dataframe(scored[show_cols].sort_values('PropensityToConvert', ascending=False).head(25), use_container_width=True)

    # -------- AI Call Activity --------
    with tab3:
        st.subheader(f"{horizon}-Day Call Forecast")
        # Recompute to chosen horizon quickly
        ds = daily_call_series(calls_df, date_col='CallDateTime')
        fc_h = forecast_calls(ds, periods=horizon)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc_h['date'], y=fc_h['forecast'], mode='lines', name='Forecast', line=dict(color='#3b82f6', width=3)))
        fig.add_trace(go.Scatter(x=fc_h['date'], y=fc_h['lo'], mode='lines', name='Lo', line=dict(color='rgba(59,130,246,0.3)', dash='dash')))
        fig.add_trace(go.Scatter(x=fc_h['date'], y=fc_h['hi'], mode='lines', name='Hi', line=dict(color='rgba(59,130,246,0.3)', dash='dash'), fill='tonexty', fillcolor='rgba(59,130,246,0.08)'))
        fig.update_layout(title="Forecast with Uncertainty Bands", paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Optimal Call Windows (by Success Rate)")
        st.dataframe(best_hours.head(6), use_container_width=True)

    # -------- Follow-up & Tasks --------
    with tab4:
        st.subheader("SLA Breach Risk Queue")
        st.dataframe(tasks_scored.sort_values('SLA_BreachRisk', ascending=False).head(30), use_container_width=True)

        st.subheader("Upcoming 7 Days")
        sched = schedule_df.copy()
        if 'ScheduledDate' in sched.columns:
            sched['ScheduledDate'] = pd.to_datetime(sched['ScheduledDate'], errors='coerce')
            upcoming = sched[(sched['ScheduledDate'] >= pd.Timestamp.now()) & (sched['ScheduledDate'] <= pd.Timestamp.now() + pd.Timedelta(days=7))]
            st.dataframe(upcoming.sort_values('ScheduledDate').head(30), use_container_width=True)
        else:
            st.info("ScheduledDate not available to compute upcoming tasks.")

    # -------- Agent Availability --------
    with tab5:
        st.subheader("Free/Busy Heatmap (Next 7 Days)")
        fc7 = fc.iloc[:7].copy()
        heat = availability_heatmap(fc7, aht_sec=300, agents=agents_slider, interval_minutes=60)
        pivot = heat.pivot(index='hour', columns='date', values='utilization').fillna(0)
        heatfig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns.astype(str), y=pivot.index,
            colorscale='RdYlGn_r', zmin=0, zmax=1
        ))
        heatfig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis_title='Hour', xaxis_title='Date')
        st.plotly_chart(heatfig, use_container_width=True)

    # -------- Conversion --------
    with tab6:
        st.subheader("Conversion Overview")
        # Converted vs Dropped (based on StatusName_E)
        if 'StatusName_E' in scored.columns:
            closed = scored['StatusName_E'].str.upper().isin(['WON','LOST'])
            conv = (scored['StatusName_E'].str.upper() == 'WON').sum()
            drop = (scored['StatusName_E'].str.upper() == 'LOST').sum()
            fig = go.Figure(data=[go.Bar(x=['Converted','Dropped'], y=[conv, drop], marker_color=['#10b981','#ef4444'])])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        conv_kpi_row = conv_kpi.to_dict(orient='records')
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(metric_card("Total Pipeline", float(conv_kpi_row['TotalPipeline']), "currency"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("PW Expected Revenue", float(conv_kpi_row['ExpectedRevenuePW']), "currency"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("Pipeline Efficiency", float(conv_kpi_row['PipelineEfficiencyPct']), "percentage"), unsafe_allow_html=True)

        st.subheader("Top 10 Opportunities")
        show_cols = [c for c in ['FullName','ExpectedRevenue','PropensityToConvert','ExpectedValue','Company','LeadStageId'] if c in top10.columns]
        st.dataframe(top10[show_cols].head(10), use_container_width=True)

    # -------- Geographic --------
    with tab7:
        st.subheader("Market Priority Ranking")
        # If you have a Country lookup, join here to display names; using CountryId as per schema for now
        show_cols = [c for c in ['CountryId','PriorityScore','exp_rev','mean_prop','mean_rev','leads'] if c in geo.columns]
        st.dataframe(geo[show_cols].head(15), use_container_width=True)

    # -------- AI Command Center --------
    with tab8:
        st.subheader("AI System Health & Insights")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_card("Lead Model AUC", float(auc)), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("Forecast Days", float(len(fc))), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("Tasks Scored", float(len(tasks_scored))), unsafe_allow_html=True)
        with c4: st.markdown(metric_card("Leads Scored", float(len(scored))), unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    f1, f2, f3 = st.columns(3)
    with f1: st.markdown("**🔄 Last Updated:** just now")
    with f2: st.markdown("**📊 Data Status:** OK")
    with f3: st.markdown("**🤖 AI Models:** Lead + SLA + Forecasting")

if __name__ == "__main__":
    main()
