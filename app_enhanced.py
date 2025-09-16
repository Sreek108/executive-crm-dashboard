# app_enhanced.py — Local CSV loader (data/ folder), no HTTP/SQL
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ML/forecasting modules (ensure these files exist under modules/)
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

# ---------------- Local CSV Data Loading ----------------
@st.cache_data(show_spinner=False)
def load_from_repo(
    base_dir: str = "data",
    leads_name: str = "enhanced_leads_advanced.csv",
    calls_name: str = "enhanced_calls_advanced.csv",
    tasks_name: str = "enhanced_schedule_advanced.csv",
    agents_name: str = "agent_performance_advanced.csv",
):
    base = Path(base_dir)
    leads_path = base / leads_name
    calls_path = base / calls_name
    tasks_path = base / tasks_name
    agents_path = base / agents_name

    missing = [p for p in [leads_path, calls_path, tasks_path, agents_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV file(s): {', '.join(str(p) for p in missing)}")  # load local files via pandas.read_csv [13]

    # Parse dates up-front
    leads = pd.read_csv(leads_path, parse_dates=['CreatedOn'], dayfirst=False)  # parse dates on read [13]
    calls = pd.read_csv(calls_path, parse_dates=['CallDateTime'], dayfirst=False)  # parse dates on read [13]
    tasks = pd.read_csv(tasks_path, parse_dates=['ScheduledDate'], dayfirst=False)  # parse dates on read [13]
    agents = pd.read_csv(agents_path)  # no dates expected [13]

    # Derive binary flags if not present
    if 'CallStatusId' in calls.columns and 'IsSuccessful' not in calls.columns:
        calls['IsSuccessful'] = (calls['CallStatusId'] == 1).astype(int)  # Connected=1 per schema [1]
    if 'TaskStatusId' in tasks.columns and 'IsBreach' not in tasks.columns:
        tasks['IsBreach'] = (tasks['TaskStatusId'] == 5).astype(int)  # Overdue=5 per schema [1]

    return leads, calls, tasks, agents  # local-only [13]

def ensure_minimum_features(leads: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    leads2 = attach_engagement_and_values(leads, calls)  # engagement + revenue backfill [13]
    if 'CreatedOn' in leads2.columns:
        leads2['LeadVelocity'] = (pd.Timestamp.now().normalize() - pd.to_datetime(leads2['CreatedOn'], errors='coerce')).dt.days.clip(lower=0)  # cycle time days [13]
    return leads2  # enriched features [13]

def cold_start_propensity(leads: pd.DataFrame) -> pd.DataFrame:
    df = leads.copy()
    eng = pd.to_numeric(df.get('EngagementScore', 0), errors='coerce').fillna(0.0)  # numeric safe [13]
    eng_n = (eng - eng.min()) / (eng.max() - eng.min() + 1e-9)  # 0..1 normalize [13]
    stage_weight = df.get('LeadStageId', pd.Series(*len(df)))
    stage_map = {1:0.35, 2:0.55, 3:0.70, 4:0.90}
    sw = stage_weight.map(stage_map).fillna(0.5)  # heuristic stage prior [13]
    df['PropensityToConvert'] = (0.6*eng_n + 0.4*sw).clip(0,1)  # blend [13]
    acts = df.apply(next_best_action, axis=1, result_type='expand')  # NBA rules [13]
    df[['NBA_Action','NBA_Priority','NBA_Confidence','NBA_Reason']] = acts  # attach [13]
    return df  # scored leads [13]

# ---------------- Main ----------------
def main():
    st.markdown("""
    <div class="dashboard-header">
        <h2 style="color: #f59e0b; margin: 0;">🚀 Executive CRM Dashboard</h2>
        <p style="color: #a0aec0; margin: 6px 0 0 0;">AI‑Powered Real Estate Analytics with Predictive Insights</p>
    </div>
    """, unsafe_allow_html=True)  # header [13]

    # Sidebar: fixed to data/ + advanced filenames
    st.sidebar.header("📦 Local dataset")  # sidebar [13]
    base_dir = st.sidebar.text_input("Folder path", value="data")  # data folder [1]
    leads_name = st.sidebar.text_input("Leads file", value="enhanced_leads_advanced.csv")  # filename [1]
    calls_name = st.sidebar.text_input("Calls file", value="enhanced_calls_advanced.csv")  # filename [1]
    tasks_name = st.sidebar.text_input("Schedule file", value="enhanced_schedule_advanced.csv")  # filename [1]
    agents_name = st.sidebar.text_input("Agent performance file", value="agent_performance_advanced.csv")  # filename [1]

    # Load CSVs locally
    try:
        leads_df, calls_df, schedule_df, agent_perf_df = load_from_repo(base_dir, leads_name, calls_name, tasks_name, agents_name)  # local read [13]
    except FileNotFoundError as e:
        st.error(str(e))  # show missing files [13]
        st.stop()  # halt [13]

    # Feature hygiene
    leads_df = ensure_minimum_features(leads_df, calls_df)  # enrich [13]

    # Controls
    st.sidebar.header("🎛️ Controls")  # UI [13]
    horizon = st.sidebar.slider("📈 Forecast Horizon (days)", 7, 60, 30)  # input [13]
    agents_slider = st.sidebar.slider("👥 Agents staffed", 5, 60, 15)  # input [13]

    # Labels detection
    has_lead_labels = 'Converted' in leads_df.columns and leads_df['Converted'].dropna().isin([0,1]).any()  # check label [13]
    has_task_labels = 'IsBreach' in schedule_df.columns and schedule_df['IsBreach'].dropna().isin([0,1]).any()  # check label [13]

    # Train or cold-start
    if has_lead_labels and has_task_labels:
        lead_model, auc = train_lead_model(leads_df, label_col='Converted')  # fit lead model [13]
        sla_model = train_sla_model(schedule_df, label_col='IsBreach')  # fit SLA model [13]
        scored = attach_nba(score_leads(lead_model, leads_df))  # infer + NBA [13]
        tasks_scored = score_sla(sla_model, schedule_df)  # predict risk [13]
    else:
        auc = float('nan')  # no AUC [13]
        scored = cold_start_propensity(leads_df)  # heuristic [13]
        t = schedule_df.copy()  # tasks [13]
        if 'ScheduledDate' in t.columns:
            t['DaysUntilDue'] = (pd.to_datetime(t['ScheduledDate'], errors='coerce') - pd.Timestamp.now()).dt.days  # delta [13]
        days = pd.to_numeric(t.get('DaysUntilDue', 0), errors='coerce').fillna(0)  # numeric [13]
        risk = np.where(days < 0, 0.9, np.clip(0.6 - 0.02*days, 0.05, 0.8))  # heuristic risk [13]
        if 'TaskStatusId' in t.columns:
            risk = np.where(t['TaskStatusId'] == 5, 0.95, risk)  # overdue high [1]
        t['SLA_BreachRisk'] = risk  # add col [13]
        t['SLA_RiskLevel'] = pd.cut(t['SLA_BreachRisk'], bins=[-0.01,0.33,0.66,1.0], labels=['Low','Medium','High'])  # bucket [13]
        tasks_scored = t  # assign [13]

    # Forecasting + optimal windows
    ds = daily_call_series(calls_df, date_col='CallDateTime')  # daily series [13]
    fc = forecast_calls(ds, periods=horizon)  # Holt‑Winters [13]
    best_hours = optimal_call_windows(calls_df)  # success by hour [13]

    # KPIs & Geo
    conv_kpi, top10 = propensity_weighted_pipeline(scored)  # pipeline KPIs [13]
    geo = market_priority(scored)  # market ranking [13]

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Executive Overview", "Lead Status", "AI Call Activity", "Follow-up & Tasks",
        "Agent Availability", "Conversion", "Geographic", "AI Command Center"
    ])  # UI tabs [13]

    # -------- Executive Overview --------
    with tab1:
        c1, c2, c3, c4 = st.columns(4)  # grid [13]
        with c1: st.markdown(metric_card("Lead Model AUC", float(auc) if auc == auc else 0.0), unsafe_allow_html=True)  # auc [13]
        with c2:
            total_pipeline = float(pd.to_numeric(scored.get('RevenuePotential', 0), errors='coerce').sum())  # sum [13]
            st.markdown(metric_card("Pipeline Value", total_pipeline, "currency"), unsafe_allow_html=True)  # card [13]
        with c3:
            pw = float(conv_kpi['ExpectedRevenuePW'].iloc)  # value [13]
            st.markdown(metric_card("30‑Day PW Revenue", pw, "currency"), unsafe_allow_html=True)  # card [13]
        with c4:
            eff = float(conv_kpi['PipelineEfficiencyPct'].iloc)  # pct [13]
            st.markdown(metric_card("Pipeline Efficiency", eff, "percentage"), unsafe_allow_html=True)  # card [13]
        st.subheader("Top Opportunities (AI‑Ranked)")  # header [13]
        cols = ['FullName','RevenuePotential','PropensityToConvert','NBA_Action','NBA_Priority','NBA_Confidence','NBA_Reason']  # fields [13]
        show_cols = [c for c in cols if c in scored.columns]  # filter [13]
        st.dataframe(scored[show_cols].sort_values('PropensityToConvert', ascending=False).head(15), use_container_width=True)  # table [13]

    # -------- Lead Status --------
    with tab2:
        st.subheader("Lead Status Breakdown (Pie)")  # header [13]
        if 'StatusName_E' in scored.columns:
            status_counts = scored['StatusName_E'].fillna('Unknown').value_counts()  # counts [13]
            fig = go.Figure(data=[go.Pie(labels=status_counts.index, values=status_counts.values, hole=0.35)])  # pie [13]
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))  # theme [13]
            st.plotly_chart(fig, use_container_width=True)  # render [13]
        elif 'LeadStageId' in scored.columns:
            stage_map = {1:'New', 2:'In Progress', 3:'Interested', 4:'Closed'}  # names [1]
            stage_counts = scored['LeadStageId'].map(stage_map).fillna('Unknown').value_counts()  # counts [13]
            fig = go.Figure(data=[go.Pie(labels=stage_counts.index, values=stage_counts.values, hole=0.35)])  # pie [13]
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))  # theme [13]
            st.plotly_chart(fig, use_container_width=True)  # render [13]
        else:
            st.info("No status/stage columns found to render pie.")  # info [13]
        st.subheader("AI Propensity & Next Best Actions")  # header [13]
        cols = ['FullName','LeadStageId','PropensityToConvert','NBA_Action','NBA_Priority','NBA_Confidence','NBA_Reason']  # fields [13]
        show_cols = [c for c in cols if c in scored.columns]  # filter [13]
        st.dataframe(scored[show_cols].sort_values('PropensityToConvert', ascending=False).head(25), use_container_width=True)  # table [13]

    # -------- AI Call Activity --------
    with tab3:
        st.subheader(f"{horizon}-Day Call Forecast")  # header [13]
        fig = go.Figure()  # figure [13]
        fig.add_trace(go.Scatter(x=fc['date'], y=fc['forecast'], mode='lines', name='Forecast', line=dict(color='#3b82f6', width=3)))  # main line [13]
        fig.add_trace(go.Scatter(x=fc['date'], y=fc['lo'], mode='lines', name='Lo', line=dict(color='rgba(59,130,246,0.3)', dash='dash')))  # lo [13]
        fig.add_trace(go.Scatter(x=fc['date'], y=fc['hi'], mode='lines', name='Hi', line=dict(color='rgba(59,130,246,0.3)', dash='dash'), fill='tonexty', fillcolor='rgba(59,130,246,0.08)'))  # hi band [13]
        fig.update_layout(title="Forecast with Uncertainty Bands", paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))  # layout [13]
        st.plotly_chart(fig, use_container_width=True)  # render [13]
        st.subheader("Optimal Call Windows (by Success Rate)")  # header [13]
        st.dataframe(best_hours.head(6), use_container_width=True)  # table [13]

    # -------- Follow-up & Tasks --------
    with tab4:
        st.subheader("SLA Breach Risk Queue")  # header [13]
        st.dataframe(tasks_scored.sort_values('SLA_BreachRisk', ascending=False).head(30), use_container_width=True)  # queue [13]
        st.subheader("Upcoming 7 Days")  # header [13]
        sched = schedule_df.copy()  # copy [13]
        if 'ScheduledDate' in sched.columns:
            upcoming = sched[(pd.to_datetime(sched['ScheduledDate'], errors='coerce') >= pd.Timestamp.now()) &
                             (pd.to_datetime(sched['ScheduledDate'], errors='coerce') <= pd.Timestamp.now() + pd.Timedelta(days=7))]  # filter [13]
            st.dataframe(upcoming.sort_values('ScheduledDate').head(30), use_container_width=True)  # table [13]
        else:
            st.info("ScheduledDate not available to compute upcoming tasks.")  # info [13]

    # -------- Agent Availability --------
    with tab5:
        st.subheader("Free/Busy Heatmap (Next 7 Days)")  # header [13]
        fc7 = fc.iloc[:7].copy()  # slice [13]
        heat = availability_heatmap(fc7, aht_sec=300, agents=agents_slider, interval_minutes=60)  # compute [13]
        pivot = heat.pivot(index='hour', columns='date', values='utilization').fillna(0)  # pivot [13]
        heatfig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.astype(str), y=pivot.index, colorscale='RdYlGn_r', zmin=0, zmax=1))  # heatmap [13]
        heatfig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis_title='Hour', xaxis_title='Date')  # theme [13]
        st.plotly_chart(heatfig, use_container_width=True)  # render [13]

        st.subheader("Agent KPIs (from CSV)")  # header [13]
        if not agent_perf_df.empty:
            kp1, kp2, kp3, kp4 = st.columns(4)  # grid [13]
            if 'PerformanceScore' in agent_perf_df.columns:
                with kp1: st.markdown(metric_card("Avg Performance", float(agent_perf_df['PerformanceScore'].mean())), unsafe_allow_html=True)  # metric [13]
            if 'CallSuccessRate' in agent_perf_df.columns:
                with kp2: st.markdown(metric_card("Avg Call Success", float(agent_perf_df['CallSuccessRate'].mean()*100), "percentage"), unsafe_allow_html=True)  # metric [13]
            if 'ConversionRate' in agent_perf_df.columns:
                with kp3: st.markdown(metric_card("Avg Conversion", float(agent_perf_df['ConversionRate'].mean()*100), "percentage"), unsafe_allow_html=True)  # metric [13]
            if 'TotalRevenue' in agent_perf_df.columns:
                with kp4: st.markdown(metric_card("Total Revenue", float(agent_perf_df['TotalRevenue'].sum()), "currency"), unsafe_allow_html=True)  # metric [13]
            st.dataframe(agent_perf_df, use_container_width=True)  # table [13]
        else:
            st.info("Agent performance CSV is empty.")  # info [13]

    # -------- Conversion --------
    with tab6:
        st.subheader("Conversion Overview")  # header [13]
        if 'StatusName_E' in scored.columns:
            conv = (scored['StatusName_E'].str.upper() == 'WON').sum()  # wins [13]
            drop = (scored['StatusName_E'].str.upper() == 'LOST').sum()  # lost [13]
            fig = go.Figure(data=[go.Bar(x=['Converted','Dropped'], y=[conv, drop], marker_color=['#10b981','#ef4444'])])  # bars [13]
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))  # theme [13]
            st.plotly_chart(fig, use_container_width=True)  # render [13]
        conv_kpi_row = conv_kpi.to_dict(orient='records')  # row [13]
        c1, c2, c3 = st.columns(3)  # grid [13]
        with c1: st.markdown(metric_card("Total Pipeline", float(conv_kpi_row['TotalPipeline']), "currency"), unsafe_allow_html=True)  # metric [13]
        with c2: st.markdown(metric_card("PW Expected Revenue", float(conv_kpi_row['ExpectedRevenuePW']), "currency"), unsafe_allow_html=True)  # metric [13]
        with c3: st.markdown(metric_card("Pipeline Efficiency", float(conv_kpi_row['PipelineEfficiencyPct']), "percentage"), unsafe_allow_html=True)  # metric [13]
        st.subheader("Top 10 Opportunities")  # header [13]
        show_cols = [c for c in ['FullName','ExpectedRevenue','PropensityToConvert','ExpectedValue','Company','LeadStageId'] if c in top10.columns]  # fields [13]
        st.dataframe(top10[show_cols].head(10), use_container_width=True)  # table [13]

    # -------- Geographic --------
    with tab7:
        st.subheader("Market Priority Ranking")  # header [13]
        show_cols = [c for c in ['CountryId','PriorityScore','exp_rev','mean_prop','mean_rev','leads'] if c in geo.columns]  # columns [13]
        st.dataframe(geo[show_cols].head(15), use_container_width=True)  # table [13]

    # -------- AI Command Center --------
    with tab8:
        st.subheader("AI System Health & Insights")  # header [13]
        c1, c2, c3, c4 = st.columns(4)  # grid [13]
        with c1: st.markdown(metric_card("Lead Model AUC", float(auc) if auc == auc else 0.0), unsafe_allow_html=True)  # metric [13]
        with c2: st.markdown(metric_card("Forecast Days", float(len(fc))), unsafe_allow_html=True)  # metric [13]
        with c3: st.markdown(metric_card("Tasks Scored", float(len(schedule_df))), unsafe_allow_html=True)  # metric [13]
        with c4: st.markdown(metric_card("Leads Scored", float(len(scored))), unsafe_allow_html=True)  # metric [13]

    # Footer
    st.markdown("---")  # divider [13]
    f1, f2, f3 = st.columns(3)  # grid [13]
    with f1: st.markdown("**🔄 Last Updated:** just now")  # text [13]
    with f2: st.markdown("**📊 Data Status:** OK")  # text [13]
    with f3: st.markdown("**🤖 AI Models:** Lead + SLA + Forecasting")  # text [13]

if __name__ == "__main__":
    main()
