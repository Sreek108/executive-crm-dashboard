# app_enhanced.py
import warnings
warnings.filterwarnings('ignore')

import os
import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ML/Forecasting modules (ensure modules/ folder is in your repo as provided earlier)
from modules.features import attach_engagement_and_values
from modules.ml_leads import train_lead_model, score_leads, attach_nba, next_best_action
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

# ---------------- GitHub/Upload Data Loading ----------------
def read_csv_public(url: str) -> pd.DataFrame:
    # Public GitHub raw URLs are directly readable by pandas over HTTPS
    return pd.read_csv(url)  # pandas supports http/https read_csv directly [25]

def read_csv_private(url: str, token: str) -> pd.DataFrame:
    # For private repos, send a GitHub token as an Authorization header
    resp = requests.get(url, headers={"Authorization": f"token {token}"})
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

@st.cache_data(show_spinner=False)
def load_from_github(raw_base: str, token: str | None, leads_name="leads_current.csv", calls_name="calls.csv", tasks_name="tasks_current.csv"):
    # raw_base example: https://raw.githubusercontent.com/USER/REPO/BRANCH/data
    base = raw_base.rstrip("/")
    urls = {
        "leads": f"{base}/{leads_name}",
        "calls": f"{base}/{calls_name}",
        "tasks": f"{base}/{tasks_name}"
    }
    reader = (lambda u: read_csv_private(u, token)) if token else read_csv_public
    leads = reader(urls["leads"])
    calls = reader(urls["calls"])
    tasks = reader(urls["tasks"])
    return leads, calls, tasks

def load_from_upload():
    c1, c2, c3 = st.columns(3)
    with c1:
        f_leads = st.file_uploader("Upload leads CSV", type=["csv"], key="leads_up")
    with c2:
        f_calls = st.file_uploader("Upload calls CSV", type=["csv"], key="calls_up")
    with c3:
        f_tasks = st.file_uploader("Upload tasks CSV", type=["csv"], key="tasks_up")
    if not (f_leads and f_calls and f_tasks):
        st.info("Please upload all three CSVs (leads, calls, tasks).")
        st.stop()
    return pd.read_csv(f_leads), pd.read_csv(f_calls), pd.read_csv(f_tasks)

def ensure_minimum_features(leads: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    # Derive EngagementScore and RevenuePotential if missing
    leads2 = attach_engagement_and_values(leads, calls)
    # LeadVelocity as days since CreatedOn if present
    if 'CreatedOn' in leads2.columns:
        leads2['CreatedOn'] = pd.to_datetime(leads2['CreatedOn'], errors='coerce')
        leads2['LeadVelocity'] = (pd.Timestamp.now().normalize() - leads2['CreatedOn']).dt.days.clip(lower=0)
    return leads2

def cold_start_propensity(leads: pd.DataFrame) -> pd.DataFrame:
    df = leads.copy()
    # Normalize EngagementScore 0..1
    eng = pd.to_numeric(df.get('EngagementScore', 0), errors='coerce').fillna(0.0)
    eng_n = (eng - eng.min()) / (eng.max() - eng.min() + 1e-9)
    stage_weight = df.get('LeadStageId', pd.Series(*len(df)))  # default mid-stage
    stage_map = {1:0.35, 2:0.55, 3:0.70, 4:0.90}
    sw = stage_weight.map(stage_map).fillna(0.5)
    # Heuristic propensity blend
    df['PropensityToConvert'] = (0.6*eng_n + 0.4*sw).clip(0,1)
    # Next Best Action via rule-based function
    acts = df.apply(next_best_action, axis=1, result_type='expand')
    df[['NBA_Action','NBA_Priority','NBA_Confidence','NBA_Reason']] = acts
    return df

# ---------------- Main ----------------
def main():
    st.markdown("""
    <div class="dashboard-header">
        <h2 style="color: #f59e0b; margin: 0;">🚀 Executive CRM Dashboard</h2>
        <p style="color: #a0aec0; margin: 6px 0 0 0;">AI‑Powered Real Estate Analytics with Predictive Insights</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("📦 Data source")
    src = st.sidebar.radio("Choose how to load data", ["GitHub (raw URLs)", "Upload CSVs"], index=0)

    if src == "GitHub (raw URLs)":
        st.sidebar.caption("Use raw.githubusercontent.com links via a base path; files must be public or provide a token for private repos.")
        raw_base = st.sidebar.text_input(
            "Raw base URL",
            placeholder="https://raw.githubusercontent.com/USER/REPO/BRANCH/data"
        )
        leads_name = st.sidebar.text_input("Leads file name", value="leads_current.csv")
        calls_name = st.sidebar.text_input("Calls file name", value="calls.csv")
        tasks_name = st.sidebar.text_input("Tasks file name", value="tasks_current.csv")
        gh_token = st.secrets.get("github", {}).get("token", os.getenv("GITHUB_TOKEN", "")) or None
        if not raw_base:
            st.info("Enter the Raw base URL to load CSVs from GitHub.")
            st.stop()
        leads_df, calls_df, schedule_df = load_from_github(raw_base, gh_token, leads_name, calls_name, tasks_name)
    else:
        leads_df, calls_df, schedule_df = load_from_upload()

    # Ensure required features for models or heuristics
    leads_df = ensure_minimum_features(leads_df, calls_df)

    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    horizon = st.sidebar.slider("📈 Forecast Horizon (days)", 7, 60, 30)
    agents_slider = st.sidebar.slider("👥 Agents staffed", 5, 60, 15)

    # Detect labels for training; otherwise cold-start heuristic
    has_lead_labels = 'Converted' in leads_df.columns and leads_df['Converted'].dropna().isin([0,1]).any()
    has_task_labels = 'IsBreach' in schedule_df.columns and schedule_df['IsBreach'].dropna().isin([0,1]).any()

    if has_lead_labels and has_task_labels:
        lead_model, auc, sla_model = train_lead_model(leads_df, 'Converted'), None, None
        # Calibrated classifier returns model + AUC in our earlier API; align here:
        # Re-train using provided signature for consistency:
        from modules.ml_leads import train_lead_model as train_lead_model_full
        lead_model, auc = train_lead_model_full(leads_df, label_col='Converted')
        from modules.ml_tasks import train_sla_model as train_sla_model_full
        sla_model = train_sla_model_full(schedule_df, label_col='IsBreach')
        # Inference
        scored = attach_nba(score_leads(lead_model, leads_df))
        tasks_scored = score_sla(sla_model, schedule_df)
    else:
        auc = float('nan')
        scored = cold_start_propensity(leads_df)
        # Heuristic SLA risk if no labels/model
        t = schedule_df.copy()
        # Derive DaysUntilDue if ScheduledDate exists
        if 'ScheduledDate' in t.columns:
            t['ScheduledDate'] = pd.to_datetime(t['ScheduledDate'], errors='coerce')
            t['DaysUntilDue'] = (t['ScheduledDate'] - pd.Timestamp.now()).dt.days
        days = pd.to_numeric(t.get('DaysUntilDue', 0), errors='coerce').fillna(0)
        # Risk: overdue -> high; else decays with days
        risk = np.where(days < 0, 0.9, np.clip(0.6 - 0.02*days, 0.05, 0.8))
        # If explicit Overdue/TaskStatusId==5 available, override to high risk
        if 'TaskStatusId' in t.columns:
            risk = np.where(t['TaskStatusId'] == 5, 0.95, risk)
        t['SLA_BreachRisk'] = risk
        t['SLA_RiskLevel'] = pd.cut(t['SLA_BreachRisk'], bins=[-0.01,0.33,0.66,1.0], labels=['Low','Medium','High'])
        tasks_scored = t

    # Calls forecasting + optimal hours
    ds = daily_call_series(calls_df, date_col='CallDateTime')
    fc = forecast_calls(ds, periods=horizon)
    best_hours = optimal_call_windows(calls_df)

    # Conversion KPIs and Geo priority
    conv_kpi, top10 = propensity_weighted_pipeline(scored)
    geo = market_priority(scored)

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
            st.markdown(metric_card("Lead Model AUC", float(auc) if auc == auc else 0.0), unsafe_allow_html=True)
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
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc['date'], y=fc['forecast'], mode='lines', name='Forecast', line=dict(color='#3b82f6', width=3)))
        fig.add_trace(go.Scatter(x=fc['date'], y=fc['lo'], mode='lines', name='Lo', line=dict(color='rgba(59,130,246,0.3)', dash='dash')))
        fig.add_trace(go.Scatter(x=fc['date'], y=fc['hi'], mode='lines', name='Hi', line=dict(color='rgba(59,130,246,0.3)', dash='dash'), fill='tonexty', fillcolor='rgba(59,130,246,0.08)')))
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
        if 'StatusName_E' in scored.columns:
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
        show_cols = [c for c in ['CountryId','PriorityScore','exp_rev','mean_prop','mean_rev','leads'] if c in geo.columns]
        st.dataframe(geo[show_cols].head(15), use_container_width=True)

    # -------- AI Command Center --------
    with tab8:
        st.subheader("AI System Health & Insights")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_card("Lead Model AUC", float(auc) if auc == auc else 0.0), unsafe_allow_html=True)
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
