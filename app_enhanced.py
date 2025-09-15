import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --------------------------------------
# Page config and minimal executive styling
# --------------------------------------
st.set_page_config(
    page_title="Executive CRM Dashboard - Advanced Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .metric-container{background:linear-gradient(135deg,#1a202c 0%,#2d3748 100%);padding:16px;border-radius:10px;border:1px solid #4a5568;margin:8px 0}
      .metric-value{font-size:2.2rem;font-weight:700;color:#f59e0b}
      .metric-label{font-size:.9rem;color:#a0aec0;text-transform:uppercase;letter-spacing:.5px}
      .dashboard-header{background:linear-gradient(90deg,#1a202c 0%,#2d3748 100%);padding:18px;border-radius:10px;margin:0 0 18px 0;border:1px solid #4a5568}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------
# Helpers
# --------------------------------------
def read_csv_smart(filename: str) -> pd.DataFrame:
    """Try data/filename then filename; raise if not found."""
    for path in (os.path.join("data", filename), filename):
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find {filename} (tried data/{filename} and {filename})")


def create_metric_card(title, value, change=None, fmt="number") -> str:
    if fmt == "currency":
        disp = f"${value:,.0f}" if isinstance(value, (int, float)) else str(value)
    elif fmt == "percentage":
        disp = f"{value:.1f}%" if isinstance(value, (int, float)) else str(value)
    else:
        if isinstance(value, (int, float)):
            disp = f"{value:,.0f}" if value >= 1000 else f"{value:.1f}"
        else:
            disp = str(value)
    delta_html = ""
    if change is not None:
        arrow = "↗️" if change > 0 else ("↘️" if change < 0 else "→")
        color = "#68d391" if change > 0 else ("#fc8181" if change < 0 else "#90cdf4")
        delta_html = f'<div style="color:{color};font-size:.8rem;margin-top:6px">{arrow} {change:+.1f}%</div>'
    return f"""
    <div class="metric-container">
      <div class="metric-label">{title}</div>
      <div class="metric-value">{disp}</div>
      {delta_html}
    </div>
    """

# --------------------------------------
# Data loading
# --------------------------------------
@st.cache_data(ttl=3600)
def load_enhanced_data():
    # Try advanced datasets first
    leads_df = calls_df = schedule_df = agent_perf_df = None
    try:
        leads_df = read_csv_smart('enhanced_leads_advanced.csv')
        calls_df = read_csv_smart('enhanced_calls_advanced.csv')
        schedule_df = read_csv_smart('enhanced_schedule_advanced.csv')
        agent_perf_df = read_csv_smart('agent_performance_advanced.csv')
    except Exception:
        # Fallback to basic names if advanced not present
        if leads_df is None:
            leads_df = read_csv_smart('enhanced_leads.csv')
        if calls_df is None:
            calls_df = read_csv_smart('enhanced_calls.csv')
        if schedule_df is None:
            schedule_df = read_csv_smart('enhanced_schedule.csv')
        if agent_perf_df is None:
            agent_perf_df = pd.DataFrame()

    # Parse dates if present
    for col in ("CreatedOn", "ModifiedOn"):
        if col in leads_df.columns:
            leads_df[col] = pd.to_datetime(leads_df[col], errors='coerce')
    if 'CallDateTime' in calls_df.columns:
        calls_df['CallDateTime'] = pd.to_datetime(calls_df['CallDateTime'], errors='coerce')
    if 'ScheduledDate' in schedule_df.columns:
        schedule_df['ScheduledDate'] = pd.to_datetime(schedule_df['ScheduledDate'], errors='coerce')

    # Ensure IsSuccessful exists on calls
    if 'IsSuccessful' not in calls_df.columns and 'CallStatusId' in calls_df.columns:
        calls_df['IsSuccessful'] = (calls_df['CallStatusId'] == 1)

    # Ensure Country label exists on leads
    if 'Country' not in leads_df.columns and 'CountryId' in leads_df.columns:
        country_map = {1:'Saudi Arabia',2:'UAE',3:'India',4:'United Kingdom',5:'United States'}
        leads_df['Country'] = leads_df['CountryId'].map(country_map)

    return leads_df, calls_df, schedule_df, agent_perf_df

# --------------------------------------
# Dashboards
# --------------------------------------
def create_enhanced_executive_summary(leads_df, calls_df, schedule_df, agent_perf_df):
    st.markdown(
        """
        <div class="dashboard-header">
          <h1 style="color:#f59e0b;margin:0">🚀 Executive CRM Dashboard</h1>
          <p style="color:#a0aec0;margin:6px 0 0 0">AI-Powered Real Estate Analytics with Predictive Insights</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        total_leads = len(leads_df)
        hot = (leads_df['LeadScoringId']==1).sum() if 'LeadScoringId' in leads_df.columns else 0
        st.markdown(create_metric_card("Total Leads", total_leads, 15.2), unsafe_allow_html=True)
        st.caption(f"🔥 {hot} Hot Leads")
    with c2:
        pipeline = float(leads_df.get('RevenuePotential', pd.Series([0])).sum())
        expected = float(leads_df.get('ExpectedRevenue', pd.Series([0])).sum())
        st.markdown(create_metric_card("Pipeline Value", pipeline, 8.7, "currency"), unsafe_allow_html=True)
        st.caption(f"💰 ${expected:,.0f} Expected")
    with c3:
        success_rate = float(calls_df['IsSuccessful'].mean()*100) if len(calls_df)>0 and 'IsSuccessful' in calls_df.columns else 0.0
        st.markdown(create_metric_card("Call Success Rate", success_rate, -2.3, "percentage"), unsafe_allow_html=True)
        st.caption(f"📞 {len(calls_df)} Total Calls")
    with c4:
        avg_eng = float(leads_df.get('EngagementScore', pd.Series([0])).mean())
        hi_eng = int((leads_df.get('EngagementScore', pd.Series([0]))>80).sum())
        st.markdown(create_metric_card("Avg Engagement", avg_eng, 5.1), unsafe_allow_html=True)
        st.caption(f"⭐ {hi_eng} High Engagement")
    with c5:
        avg_conv = float(leads_df.get('ConversionProbability', pd.Series([0])).mean()*100)
        likely = int((leads_df.get('ConversionProbability', pd.Series([0]))>0.7).sum())
        st.markdown(create_metric_card("Conversion Rate", avg_conv, 12.4, "percentage"), unsafe_allow_html=True)
        st.caption(f"🎯 {likely} Likely Converts")


def create_enhanced_lead_status_dashboard(leads_df):
    st.subheader("📊 Advanced Lead Analytics")
    c1,c2 = st.columns(2)
    with c1:
        if 'LeadStageId' in leads_df.columns:
            stage_map = {1:'New',2:'Qualified',3:'Nurtured',4:'Converted'}
            cnt = leads_df['LeadStageId'].value_counts().sort_index()
            labels = [stage_map.get(i, str(i)) for i in cnt.index]
            fig = go.Figure(go.Funnel(y=labels, x=cnt.values, textinfo="value+percent initial",
                                      marker_color=['#3b82f6','#f59e0b','#10b981','#ef4444']))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if 'BehavioralSegment' in leads_df.columns:
            seg = leads_df['BehavioralSegment'].value_counts()
            colors = {'Champions':'#10b981','Loyal Customers':'#3b82f6','Potential Loyalists':'#f59e0b','At Risk':'#ef4444','Need Attention':'#8b5cf6'}
            fig = go.Figure(go.Pie(labels=seg.index, values=seg.values, hole=.4,
                                   marker_colors=[colors.get(s,'#6b7280') for s in seg.index]))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌡️ Temperature & Engagement")
    c3,c4,c5 = st.columns(3)
    with c3:
        if 'TemperatureTrend' in leads_df.columns:
            temp = leads_df['TemperatureTrend'].value_counts()
            fig = go.Figure(go.Bar(x=temp.index, y=temp.values, marker_color=['#10b981','#f59e0b','#ef4444']))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        if 'EngagementScore' in leads_df.columns:
            fig = go.Figure(go.Histogram(x=leads_df['EngagementScore'], nbinsx=12, marker_color='#8b5cf6'))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
    with c5:
        if 'LeadVelocity' in leads_df.columns:
            avg_v = float(leads_df['LeadVelocity'].mean())
            fig = go.Figure(go.Indicator(mode='gauge+number', value=avg_v, title={'text':'Avg Lead Velocity (days)'},
                                         gauge={'axis':{'range':[0,50]}, 'bar':{'color':'#f59e0b'},
                                                'steps':[{'range':[0,15],'color':'#10b981'},{'range':[15,30],'color':'#f59e0b'},{'range':[30,50],'color':'#ef4444'}]}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
            st.plotly_chart(fig, use_container_width=True)


def create_enhanced_call_activity_dashboard(calls_df):
    st.subheader("🤖 Advanced Call Analytics & AI Insights")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(create_metric_card("Total Calls", len(calls_df), 23.1), unsafe_allow_html=True)
    with c2:
        rate = float(calls_df['IsSuccessful'].mean()*100) if 'IsSuccessful' in calls_df.columns and len(calls_df)>0 else 0.0
        st.markdown(create_metric_card("Success Rate", rate, -5.2, "percentage"), unsafe_allow_html=True)
    with c3:
        if 'DurationSeconds' in calls_df.columns and 'IsSuccessful' in calls_df.columns:
            s = calls_df[calls_df['IsSuccessful']]['DurationSeconds']
            avg_min = float(s.mean()/60) if len(s)>0 else 0.0
            st.markdown(create_metric_card("Avg Duration", avg_min, 8.7), unsafe_allow_html=True)
            st.caption("Minutes")
    with c4:
        if 'CallEfficiency' in calls_df.columns:
            eff = float(calls_df['CallEfficiency'].mean()*100)
            st.markdown(create_metric_card("Efficiency", eff, 12.3, "percentage"), unsafe_allow_html=True)

    if 'CallHour' in calls_df.columns:
        hourly = calls_df.groupby('CallHour').agg(Total=('IsSuccessful','count'), Successful=('IsSuccessful','sum'))
        hourly['SuccessRate'] = (hourly['Successful']/hourly['Total']*100).fillna(0)
        fig = go.Figure()
        fig.add_bar(x=hourly.index, y=hourly['Total'], name='Total Calls', marker_color='#3b82f6', opacity=.7)
        fig.add_scatter(x=hourly.index, y=hourly['SuccessRate'], name='Success %', yaxis='y2', line=dict(color='#10b981', width=3))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                          yaxis=dict(title='Calls'), yaxis2=dict(title='Success %', overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)

# --------- SAFETY PREP FOR TASKS (fix for IndexingError) ---------
def _prepare_schedule(schedule_df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return (days_to_due, status_col) with safe alignment and types."""
    # Ensure ScheduledDate as datetime
    if 'ScheduledDate' in schedule_df.columns:
        schedule_df['ScheduledDate'] = pd.to_datetime(schedule_df['ScheduledDate'], errors='coerce')
    # Derive days_to_due aligned to df length
    if 'DaysUntilDue' in schedule_df.columns:
        days_to_due = pd.to_numeric(schedule_df['DaysUntilDue'], errors='coerce')
        # If it's all NaN or wrong length (shouldn't happen), recompute from ScheduledDate
        if days_to_due.shape[0] != len(schedule_df) or days_to_due.isna().all():
            scheduled = schedule_df.get('ScheduledDate')
            days_to_due = (scheduled - pd.Timestamp.now()).dt.days if scheduled is not None else pd.Series(np.nan, index=schedule_df.index)
    else:
        scheduled = schedule_df.get('ScheduledDate')
        days_to_due = (scheduled - pd.Timestamp.now()).dt.days if scheduled is not None else pd.Series(np.nan, index=schedule_df.index)
    # Choose status column
    status_col = 'TaskStatusId' if 'TaskStatusId' in schedule_df.columns else ('TaskStatus' if 'TaskStatus' in schedule_df.columns else None)
    return days_to_due, status_col
# -----------------------------------------------------------------

def create_enhanced_task_dashboard(schedule_df):
    st.subheader("📅 Advanced Task Management & SLA Tracking")

    # SAFE: always aligned boolean mask
    days_to_due, status_col = _prepare_schedule(schedule_df)

    # Metrics
    total_tasks = len(schedule_df)
    overdue_mask = days_to_due.lt(0)

    # Normalize open/completed using either Id or text
    if status_col is not None:
        if status_col == 'TaskStatusId':
            open_mask = schedule_df[status_col].isin([1, 2, 5])   # Pending / In Progress / Overdue
            completed_mask = schedule_df[status_col].isin([3])    # Completed
        else:
            open_mask = schedule_df[status_col].astype(str).isin(['Pending', 'In Progress', 'Overdue'])
            completed_mask = schedule_df[status_col].astype(str).isin(['Completed'])
    else:
        # If no status column, assume all open and none completed
        open_mask = pd.Series(True, index=schedule_df.index)
        completed_mask = pd.Series(False, index=schedule_df.index)

    overdue_tasks = int((overdue_mask & open_mask).sum())
    high_priority = int((schedule_df.get('Priority', pd.Series('Medium', index=schedule_df.index)) == 'High').sum())
    avg_completion_prob = float(schedule_df.get('CompletionProbability', pd.Series(np.nan)).mean() * 100)

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(create_metric_card("Total Tasks", total_tasks, 15.3), unsafe_allow_html=True)
    with c2:
        st.markdown(create_metric_card("Overdue Tasks", overdue_tasks, -12.5), unsafe_allow_html=True)
    with c3:
        st.markdown(create_metric_card("High Priority", high_priority, 8.2), unsafe_allow_html=True)
    with c4:
        if not np.isnan(avg_completion_prob):
            st.markdown(create_metric_card("Completion Rate", avg_completion_prob, 5.7, "percentage"), unsafe_allow_html=True)

    # SLA donut
    if 'SLAStatus' in schedule_df.columns:
        sla = schedule_df['SLAStatus'].value_counts()
        colors = {'On Track':'#10b981','At Risk':'#f59e0b','Breach':'#ef4444'}
        fig = go.Figure(go.Pie(labels=sla.index, values=sla.values, hole=.45,
                               marker_colors=[colors.get(s,'#6b7280') for s in sla.index]))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), title='SLA Status Distribution')
        st.plotly_chart(fig, use_container_width=True)

    # Upcoming 7 days (SAFE FILTER) — this replaces the buggy get(..., pd.Series([999])) usage
    upcoming_week = schedule_df[days_to_due.between(0, 7, inclusive='both')]
    st.caption(f"Tasks due in next 7 days: {len(upcoming_week)}")


def create_enhanced_agent_dashboard(agent_perf_df, schedule_df):
    st.subheader("👥 Advanced Agent Performance Analytics")
    if agent_perf_df is None or len(agent_perf_df)==0:
        st.info("No agent performance data available.")
        return
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(create_metric_card("Total Agents", len(agent_perf_df)), unsafe_allow_html=True)
    with c2:
        if 'PerformanceScore' in agent_perf_df.columns:
            st.markdown(create_metric_card("Avg Performance", float(agent_perf_df['PerformanceScore'].mean()), 7.3), unsafe_allow_html=True)
    with c3:
        if 'CallSuccessRate' in agent_perf_df.columns:
            st.markdown(create_metric_card("Call Success", float(agent_perf_df['CallSuccessRate'].mean()*100), -2.1, "percentage"), unsafe_allow_html=True)
    with c4:
        if 'ConversionRate' in agent_perf_df.columns:
            st.markdown(create_metric_card("Conversion Rate", float(agent_perf_df['ConversionRate'].mean()*100), 12.8, "percentage"), unsafe_allow_html=True)

    # Comparison chart
    if {'AgentName','PerformanceScore'}.issubset(agent_perf_df.columns):
        fig = go.Figure()
        fig.add_bar(x=agent_perf_df['AgentName'], y=agent_perf_df['PerformanceScore'], name='Performance', marker_color='#3b82f6')
        if 'EfficiencyScore' in agent_perf_df.columns:
            fig.add_bar(x=agent_perf_df['AgentName'], y=agent_perf_df['EfficiencyScore']*100, name='Efficiency', marker_color='#10b981')
        fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)


def create_enhanced_conversion_dashboard(leads_df):
    st.subheader("💼 Advanced Conversion & Revenue Intelligence")
    total_leads = len(leads_df)
    converted = int((leads_df.get('LeadStageId', pd.Series([0]))==4).sum())
    pipeline = float(leads_df.get('RevenuePotential', pd.Series([0])).sum())
    expected = float(leads_df.get('ExpectedRevenue', pd.Series([0])).sum())
    conv_rate = (converted/total_leads*100) if total_leads>0 else 0
    eff = (expected/pipeline*100) if pipeline>0 else 0
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(create_metric_card("Conversion Rate", conv_rate, 8.5, "percentage"), unsafe_allow_html=True)
    with c2: st.markdown(create_metric_card("Pipeline Value", pipeline, 15.2, "currency"), unsafe_allow_html=True)
    with c3: st.markdown(create_metric_card("Expected Revenue", expected, 12.8, "currency"), unsafe_allow_html=True)
    with c4: st.markdown(create_metric_card("Pipeline Efficiency", eff, 5.3, "percentage"), unsafe_allow_html=True)

    if {'ConversionProbability','RevenuePotential'}.issubset(leads_df.columns):
        fig = go.Figure()
        if 'LeadScoringId' in leads_df.columns:
            colors = {1:'#dc2626',2:'#f59e0b',3:'#3b82f6',4:'#6b7280'}
            for sid, grp in leads_df.groupby('LeadScoringId'):
                label = {1:'HOT',2:'WARM',3:'COLD',4:'DEAD'}.get(sid,str(sid))
                fig.add_scatter(x=grp['ConversionProbability'], y=grp['RevenuePotential'], mode='markers', name=label,
                                marker=dict(size=10, color=colors.get(sid,'#999'), opacity=.8))
        else:
            fig.add_scatter(x=leads_df['ConversionProbability'], y=leads_df['RevenuePotential'], mode='markers', name='Leads', marker=dict(size=9, color='#3b82f6'))
        fig.update_layout(title='Revenue vs Conversion Probability', xaxis_title='Conversion Probability', yaxis_title='Revenue ($)',
                          paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)


def create_enhanced_geographic_dashboard(leads_df):
    st.subheader("🌍 Global Market Intelligence Dashboard")
    if 'Country' not in leads_df.columns:
        st.info("No geographic information available.")
        return
    geo = leads_df.groupby('Country').agg(Lead_Count=('LeadId','count'), Total_Pipeline=('RevenuePotential','sum'),
                                          Avg_Deal_Size=('RevenuePotential','mean'), Conversion_Rate=('ConversionProbability','mean'),
                                          Expected_Revenue=('ExpectedRevenue','sum')).fillna(0).reset_index()
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=geo['Country'], y=geo['Lead_Count'], marker_color='#3b82f6'))
        fig.update_layout(title='Lead Count by Country', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Bar(x=geo['Country'], y=geo['Expected_Revenue'], marker_color='#10b981'))
        fig.update_layout(title='Expected Revenue by Country', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)


def create_advanced_ai_insights_dashboard(leads_df, calls_df, schedule_df, agent_perf_df):
    st.subheader("🧠 AI/ML Intelligence Center")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(create_metric_card("Model Accuracy", 87.3, 2.1, "percentage"), unsafe_allow_html=True)
    with c2: st.markdown(create_metric_card("Churn Confidence", 82.1, -1.5, "percentage"), unsafe_allow_html=True)
    with c3: st.markdown(create_metric_card("Forecast Precision", 79.8, 5.3, "percentage"), unsafe_allow_html=True)
    with c4: st.markdown(create_metric_card("Action Success", 74.2, 8.7, "percentage"), unsafe_allow_html=True)

# --------------------------------------
# Main
# --------------------------------------
def main():
    leads_df, calls_df, schedule_df, agent_perf_df = load_enhanced_data()

    # Sidebar filters (display only – wiring to filters can be extended)
    st.sidebar.header("🎛️ Advanced Analytics Filters")
    st.sidebar.selectbox("Select Time Period", ["Last 7 days","Last 30 days","Last 90 days","All time"], index=1)
    st.sidebar.multiselect("Select Agents", ['All Agents','Jasmin Ahmed','Mohammed Ali','Sarah Johnson','Ahmed Hassan','Fatima Al-Zahra'], default=['All Agents'])
    st.sidebar.multiselect("Select Markets", ['All Markets','Saudi Arabia','UAE','India','United Kingdom','United States'], default=['All Markets'])
    st.sidebar.selectbox("Lead Temperature Filter", ["All Temperatures","HOT","WARM","COLD","DEAD"], index=0)
    if 'BehavioralSegment' in leads_df.columns:
        segments = ['All Segments'] + sorted(leads_df['BehavioralSegment'].dropna().unique().tolist())
        st.sidebar.multiselect("Behavioral Segments", segments, default=['All Segments'])
    st.sidebar.checkbox("Auto-refresh (30 seconds)", value=False)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🚀 Executive Overview",
        "📊 Lead Intelligence",
        "🤖 Call Analytics",
        "📅 Task Management",
        "👥 Agent Performance",
        "💼 Revenue Intelligence",
        "🌍 Market Analysis",
        "🧠 AI Command Center"
    ])

    with tab1:
        create_enhanced_executive_summary(leads_df, calls_df, schedule_df, agent_perf_df)

    with tab2:
        create_enhanced_lead_status_dashboard(leads_df)

    with tab3:
        create_enhanced_call_activity_dashboard(calls_df)

    with tab4:
        create_enhanced_task_dashboard(schedule_df)

    with tab5:
        create_enhanced_agent_dashboard(agent_perf_df, schedule_df)

    with tab6:
        create_enhanced_conversion_dashboard(leads_df)

    with tab7:
        create_enhanced_geographic_dashboard(leads_df)

    with tab8:
        create_advanced_ai_insights_dashboard(leads_df, calls_df, schedule_df, agent_perf_df)


if __name__ == "__main__":
    main()
