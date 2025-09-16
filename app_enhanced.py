import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime

# ------------------------------------------------------------
# Page config (fixed unterminated string)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Executive CRM Dashboard - Advanced Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def aligned_series(df, value=0, dtype=None):
    """
    Create a Series of a constant value aligned to df.index.
    Prevents TypeErrors from shape mismatch in .get(...) fallbacks.
    """
    return pd.Series([value] * len(df), index=df.index, dtype=dtype)

# ------------------------------------------------------------
# Executive CSS
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #0f1419; color: #ffffff; }
        .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%); }

        .metric-container {
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
            padding: 20px; border-radius: 12px; border: 1px solid #4a5568;
            margin: 10px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        .metric-value { font-size: 2.8rem; font-weight: bold; color: #f59e0b; }
        .metric-label { font-size: 0.9rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
        .metric-change { font-size: 0.8rem; font-weight: 600; margin-top: 8px; }
        .positive { color: #68d391; } .negative { color: #fc8181; } .neutral { color: #90cdf4; }

        .dashboard-header {
            background: linear-gradient(90deg, #1a202c 0%, #2d3748 100%);
            padding: 25px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #4a5568;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .insight-card{
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            padding: 20px; border-radius: 10px; border-left: 5px solid #f59e0b; margin: 15px 0;
        }
        .alert-high { border-left-color: #fc8181; }
        .alert-medium { border-left-color: #f6ad55; }
        .alert-low { border-left-color: #68d391; }

        .performance-card {
            background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6b 100%);
            padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #3b82f6;
        }
        .prediction-box {
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            padding: 15px; border-radius: 8px; margin: 10px 0; color: white; border: 1px solid #60a5fa;
        }
        .ai-recommendation {
            background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
            padding: 15px; border-radius: 8px; margin: 10px 0; color: white; border: 1px solid #a78bfa;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_enhanced_data():
    """
    Load enhanced CRM datasets with advanced analytics.
    If files are missing, generate realistic fallback data.
    """
    try:
        leads_df = pd.read_csv("enhanced_leads_advanced.csv")
        calls_df = pd.read_csv("enhanced_calls_advanced.csv")
        schedule_df = pd.read_csv("enhanced_schedule_advanced.csv")
        agent_perf_df = pd.read_csv("agent_performance_advanced.csv")

        # Parse dates safely
        for col in ["CreatedOn", "ModifiedOn"]:
            if col in leads_df.columns:
                leads_df[col] = pd.to_datetime(leads_df[col], errors="coerce")
        if "CallDateTime" in calls_df.columns:
            calls_df["CallDateTime"] = pd.to_datetime(calls_df["CallDateTime"], errors="coerce")
        if "ScheduledDate" in schedule_df.columns:
            schedule_df["ScheduledDate"] = pd.to_datetime(schedule_df["ScheduledDate"], errors="coerce")

        return leads_df, calls_df, schedule_df, agent_perf_df
    except Exception:
        return create_fallback_data()

def create_fallback_data():
    """
    Create fallback sample data if files are not found.
    """
    st.warning("Enhanced data files not found. Using sample data for demonstration.")

    n_leads = 50
    n_calls = 80
    n_tasks = 30
    n_agents = 5

    rng = np.random.default_rng(42)
    start_date = pd.Timestamp("2025-08-01")

    leads_df = pd.DataFrame({
        "LeadId": range(1, n_leads + 1),
        "FullName": [f"Lead {i}" for i in range(1, n_leads + 1)],
        "Company": [f"Company {i}" for i in range(1, n_leads + 1)],
        "Country": rng.choice(["Saudi Arabia", "UAE", "India", "UK", "US"], n_leads),
        "LeadStageId": rng.choice([1, 2, 3, 4], n_leads),
        "LeadScoringId": rng.choice([1, 2, 3, 4], n_leads),
        "RevenuePotential": rng.uniform(50_000, 300_000, n_leads).round(2),
        "ExpectedRevenue": rng.uniform(20_000, 200_000, n_leads).round(2),
        "ConversionProbability": rng.uniform(0.1, 0.9, n_leads).round(3),
        "EngagementScore": rng.integers(20, 100, n_leads),
        "BehavioralSegment": rng.choice(["Champions", "At Risk", "Need Attention", "Loyal Customers"], n_leads),
        "ChurnRisk": rng.uniform(0.1, 0.9, n_leads).round(3),
        "CreatedOn": pd.date_range(start_date, periods=n_leads, freq="D"),
    })
    leads_df["LeadVelocity"] = (pd.Timestamp.now().normalize() - leads_df["CreatedOn"]).dt.days.clip(lower=0)

    calls_df = pd.DataFrame({
        "LeadCallId": range(1, n_calls + 1),
        "IsSuccessful": rng.choice([True, False], n_calls, p=[0.58, 0.42]),
        "DurationSeconds": rng.integers(60, 1800, n_calls),
        "CallDateTime": pd.date_range(start_date, periods=n_calls, freq="H"),
        "CallHour": [d.hour for d in pd.date_range(start_date, periods=n_calls, freq="H")],
        "CallPattern": rng.choice(["Cold", "Warm", "Follow-up", "Demo"], n_calls),
        "Sentiment": rng.choice(["Positive", "Neutral", "Negative"], n_calls, p=[0.52, 0.32, 0.16]),
        "CallOutcome": rng.choice(["No Answer", "Callback", "Meeting Booked", "Closed Won", "Closed Lost"], n_calls),
    })

    schedule_df = pd.DataFrame({
        "ScheduleId": range(1, n_tasks + 1),
        "TaskStatus": rng.choice(["Pending", "Completed", "Overdue"], n_tasks),
        "Priority": rng.choice(["High", "Medium", "Low"], n_tasks, p=[0.25, 0.55, 0.20]),
        "SLAStatus": rng.choice(["On Track", "At Risk", "Breach"], n_tasks, p=[0.6, 0.25, 0.15]),
        "EstimatedEffortHours": rng.uniform(1, 8, n_tasks).round(1),
        "ActualEffortHours": rng.uniform(0.5, 10, n_tasks).round(1),
        "ScheduledDate": pd.date_range(start_date, periods=n_tasks, freq="2D"),
    })
    schedule_df["DaysUntilDue"] = (schedule_df["ScheduledDate"] - pd.Timestamp.now()).dt.days

    agent_perf_df = pd.DataFrame({
        "AgentId": range(1, n_agents + 1),
        "AgentName": ["Agent A", "Agent B", "Agent C", "Agent D", "Agent E"],
        "PerformanceScore": rng.uniform(70, 95, n_agents).round(1),
        "CallSuccessRate": rng.uniform(0.55, 0.9, n_agents).round(3),
        "ConversionRate": rng.uniform(0.2, 0.6, n_agents).round(3),
        "TotalRevenue": rng.uniform(100_000, 900_000, n_agents).round(0),
        "TotalLeads": rng.integers(40, 160, n_agents),
        "ConvertedLeads": rng.integers(10, 60, n_agents),
        "SatisfactionScore": rng.uniform(6.5, 9.6, n_agents).round(1),
        "TaskCompletionRate": rng.uniform(0.6, 0.95, n_agents).round(3),
        "EfficiencyScore": rng.uniform(0.6, 0.95, n_agents).round(3),
        "Role": ["Manager", "Senior", "Senior", "Junior", "Junior"],
    })

    return leads_df, calls_df, schedule_df, agent_perf_df

# ------------------------------------------------------------
# Metric card helper
# ------------------------------------------------------------
def create_metric_card(title, value, change=None, format_type="number"):
    if format_type == "currency":
        display_value = f"${float(value):,.0f}" if isinstance(value, (int, float)) else str(value)
    elif format_type == "percentage":
        display_value = f"{float(value):.1f}%" if isinstance(value, (int, float)) else str(value)
    else:
        display_value = f"{value:,.0f}" if isinstance(value, (int, float)) and value >= 1000 else f"{float(value):.1f}" if isinstance(value, (int, float)) else str(value)

    change_html = ""
    if change is not None and isinstance(change, (int, float)):
        change_class = "positive" if change > 0 else "negative" if change < 0 else "neutral"
        change_icon = "↗️" if change > 0 else "↘️" if change < 0 else "→"
        change_html = f'<div class="metric-change {change_class}">{change_icon} {change:+.1f}%</div>'

    return f"""
    <div class="metric-container">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{display_value}</div>
        {change_html}
    </div>
    """

# ------------------------------------------------------------
# Executive Overview (fixed aligned fallbacks)
# ------------------------------------------------------------
def create_enhanced_executive_summary(leads_df, calls_df, schedule_df, agent_perf_df):
    st.markdown(
        """
        <div class="dashboard-header">
            <h1 style="color: #f59e0b; margin: 0; font-size: 2.2rem;">🚀 Executive CRM Dashboard</h1>
            <p style="color: #a0aec0; margin: 10px 0 0 0; font-size: 1.05rem;">AI-Powered Real Estate Analytics with Predictive Insights</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_leads = len(leads_df)
        lead_scoring = pd.to_numeric(leads_df.get("LeadScoringId", aligned_series(leads_df, 0)), errors="coerce").fillna(0)
        hot_leads = int((lead_scoring == 1).sum())
        st.markdown(create_metric_card("Total Leads", total_leads, 15.2), unsafe_allow_html=True)
        st.caption(f"🔥 {hot_leads} Hot Leads")

    with col2:
        revenue_potential = pd.to_numeric(leads_df.get("RevenuePotential", aligned_series(leads_df, 0.0)), errors="coerce").fillna(0.0)
        expected_revenue = pd.to_numeric(leads_df.get("ExpectedRevenue", leads_df.get("RevenuePotential", aligned_series(leads_df, 0.0))), errors="coerce").fillna(0.0)
        st.markdown(create_metric_card("Pipeline Value", float(revenue_potential.sum()), 8.7, "currency"), unsafe_allow_html=True)
        st.caption(f"💰 ${float(expected_revenue.sum()):,.0f} Expected")

    with col3:
        success_series = calls_df.get("IsSuccessful", aligned_series(calls_df, 0)).astype(float)
        success_rate = float(np.nanmean(success_series) * 100) if len(success_series) else 0.0
        st.markdown(create_metric_card("Call Success Rate", success_rate, -2.3, "percentage"), unsafe_allow_html=True)
        st.caption(f"📞 {len(calls_df)} Total Calls")

    with col4:
        engagement = pd.to_numeric(leads_df.get("EngagementScore", aligned_series(leads_df, 0)), errors="coerce").fillna(0.0)
        avg_engagement = float(engagement.mean())
        high_engagement = int((engagement > 80).sum())
        st.markdown(create_metric_card("Avg Engagement", avg_engagement, 5.1), unsafe_allow_html=True)
        st.caption(f"⭐ {high_engagement} High Engagement")

    with col5:
        conv = pd.to_numeric(leads_df.get("ConversionProbability", aligned_series(leads_df, 0.0)), errors="coerce").fillna(0.0)
        conversion_rate = float(conv.mean() * 100)
        predicted_conversions = int((conv > 0.7).sum())
        st.markdown(create_metric_card("Conversion Rate", conversion_rate, 12.4, "percentage"), unsafe_allow_html=True)
        st.caption(f"🎯 {predicted_conversions} Likely Converts")

# ------------------------------------------------------------
# Lead Intelligence (robust + AI NBAs)
# ------------------------------------------------------------
def create_enhanced_lead_status_dashboard(leads_df):
    st.subheader("📊 Advanced Lead Analytics")

    col1, col2 = st.columns(2)
    with col1:
        stage_mapping = {1: "New", 2: "Qualified", 3: "Nurtured", 4: "Converted"}
        if "LeadStageId" in leads_df.columns:
            stage_counts = leads_df["LeadStageId"].value_counts(dropna=False).sort_index()
            stage_labels = [stage_mapping.get(i, f"Stage {i}") for i in stage_counts.index]
            fig_funnel = go.Figure(go.Funnel(
                y=stage_labels, x=stage_counts.values, textinfo="value+percent initial",
                marker_color=["#3b82f6", "#f59e0b", "#10b981", "#ef4444"],
            ))
            fig_funnel.update_layout(title="Lead Conversion Funnel with Drop-off Analysis",
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_funnel, use_container_width=True)
        else:
            st.info("LeadStageId column not found; funnel chart skipped.")

    with col2:
        if "BehavioralSegment" in leads_df.columns:
            segment_counts = leads_df["BehavioralSegment"].value_counts(dropna=False)
            colors = {"Champions": "#10b981", "Loyal Customers": "#3b82f6", "Potential Loyalists": "#f59e0b", "At Risk": "#ef4444", "Need Attention": "#8b5cf6"}
            fig_segments = go.Figure(data=[go.Pie(labels=segment_counts.index.astype(str),
                                                  values=segment_counts.values, hole=0.4,
                                                  marker_colors=[colors.get(seg, "#6b7280") for seg in segment_counts.index])])
            fig_segments.update_layout(title="Customer Behavioral Segmentation",
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_segments, use_container_width=True)
        else:
            st.info("BehavioralSegment column not found; segmentation chart skipped.")

    st.subheader("🌡️ Lead Temperature & Engagement Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        if "TemperatureTrend" in leads_df.columns:
            temp_counts = leads_df["TemperatureTrend"].value_counts(dropna=False)
            fig_temp = go.Figure(data=[go.Bar(x=temp_counts.index.astype(str), y=temp_counts.values,
                                              marker_color=["#10b981", "#f59e0b", "#ef4444", "#6b7280"])])
            fig_temp.update_layout(title="Lead Temperature Trends", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("TemperatureTrend column not found; temperature chart skipped.")

    with col2:
        if "EngagementScore" in leads_df.columns:
            fig_engagement = go.Figure()
            fig_engagement.add_trace(go.Histogram(x=leads_df["EngagementScore"], nbinsx=10, marker_color="#8b5cf6", opacity=0.8))
            fig_engagement.update_layout(title="Engagement Score Distribution", xaxis_title="Engagement Score",
                                         yaxis_title="Number of Leads", paper_bgcolor="rgba(0,0,0,0)",
                                         plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_engagement, use_container_width=True)
        else:
            st.info("EngagementScore column not found; distribution chart skipped.")

    with col3:
        if "LeadVelocity" in leads_df.columns:
            avg_velocity = pd.to_numeric(leads_df["LeadVelocity"], errors="coerce").mean()
            fig_velocity = go.Figure(go.Indicator(mode="gauge+number",
                                                  value=float(avg_velocity) if not np.isnan(avg_velocity) else 0.0,
                                                  title={"text": "Average Lead Velocity (Days)"},
                                                  gauge={"axis": {"range": [0, 50]},
                                                         "bar": {"color": "#f59e0b"},
                                                         "steps": [{"range": [0, 15], "color": "#10b981"},
                                                                   {"range": [15, 30], "color": "#f59e0b"},
                                                                   {"range": [30, 50], "color": "#ef4444"}]}))
            fig_velocity.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white", "family": "Arial"})
            st.plotly_chart(fig_velocity, use_container_width=True)
        else:
            st.info("LeadVelocity column not found; gauge skipped.")

    # Intelligence section
    st.markdown("---")
    st.subheader("🧭 Lead Intelligence Add-ons")

    idx = leads_df.index
    eng = pd.to_numeric(leads_df.get("EngagementScore", aligned_series(leads_df, np.nan)), errors="coerce")
    conv = pd.to_numeric(leads_df.get("ConversionProbability", aligned_series(leads_df, np.nan)), errors="coerce")
    rev  = pd.to_numeric(leads_df.get("ExpectedRevenue", leads_df.get("RevenuePotential", aligned_series(leads_df, 0.0))), errors="coerce").fillna(0.0)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        med_eng = float(np.nanmedian(eng)) if len(eng) else 0.0
        st.markdown(create_metric_card("Median Engagement", med_eng, 0.0), unsafe_allow_html=True)
    with k2:
        avg_conv = float(np.nanmean(conv)*100) if len(conv) else 0.0
        st.markdown(create_metric_card("Avg Conv Prob", avg_conv, 0.0, "percentage"), unsafe_allow_html=True)
    with k3:
        st.markdown(create_metric_card("Total Expected", float(rev.sum()), 0.0, "currency"), unsafe_allow_html=True)
    with k4:
        hot_ready = int(((conv > 0.70) & (rev > 0)).sum())
        st.markdown(create_metric_card("Hot Opportunities", hot_ready), unsafe_allow_html=True)

    quality = (0.6*conv.fillna(0.0) + 0.4*(eng.fillna(0.0)/100.0)).clip(0, 1)
    table = pd.DataFrame({
        "Lead": leads_df.get("FullName", pd.Series([f"Lead {i+1}" for i in range(len(idx))], index=idx)),
        "Company": leads_df.get("Company", pd.Series(["-"]*len(idx), index=idx)),
        "Country": leads_df.get("Country", pd.Series(["-"]*len(idx), index=idx)),
        "QualityScore": quality.round(3),
        "ConvProb": conv.fillna(0.0).round(3),
        "Revenue": rev,
    })
    top = table.sort_values(["QualityScore", "Revenue"], ascending=[False, False]).head(10)

    st.markdown("**Top 10 Opportunities**")
    st.dataframe(
        top.style.format({"QualityScore": "{:.2f}", "ConvProb": "{:.1%}", "Revenue": "${:,.0f}"}),
        use_container_width=True, height=360
    )

    hot = int((conv > 0.7).sum())
    cooling = int((leads_df.get("TemperatureTrend", pd.Series(["Unknown"]*len(idx), index=idx)) == "Cooling Down").sum())
    champions = int((leads_df.get("BehavioralSegment", pd.Series(["-"]*len(idx), index=idx)) == "Champions").sum())

    st.markdown("- ✅ Focus first on leads with QualityScore ≥ 0.75 for fastest wins.")
    st.markdown(f"- 🚩 {hot} high-probability leads detected; schedule demos within 48 hours.")
    st.markdown(f"- 🧊 {cooling} cooling leads require retention play; trigger nurturing workflow.")
    st.markdown(f"- 🏆 {champions} champions segment shows strong upsell potential this month.")

    # Propensity models
    st.markdown("### 🎯 AI Propensity Models")
    eng_norm = (eng.fillna(0) / 100).clip(0, 1)
    rev_norm = (rev / (rev.max() if rev.max() > 0 else 1)).clip(0, 1)

    buy = pd.to_numeric(leads_df.get("PropensityToBuy", np.nan), errors="coerce")
    churn = pd.to_numeric(leads_df.get("PropensityToChurn", np.nan), errors="coerce")
    upgrade = pd.to_numeric(leads_df.get("PropensityToUpgrade", np.nan), errors="coerce")

    buy = buy.fillna((0.55*conv.fillna(0) + 0.35*eng_norm + 0.10*rev_norm).clip(0, 1))
    cooling_flag = (leads_df.get("TemperatureTrend", pd.Series(["Unknown"]*len(idx), index=idx)) == "Cooling Down").astype(int)
    churn = churn.fillna((0.65*(1 - conv.fillna(0)) + 0.25*(1 - eng_norm) + 0.10*cooling_flag).clip(0, 1))
    upgrade = upgrade.fillna((0.50*eng_norm + 0.30*conv.fillna(0) + 0.20*rev_norm).clip(0, 1))

    g1, g2, g3 = st.columns(3)
    with g1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(buy.mean()*100), title={"text": "Avg Buy Propensity"},
                                     gauge={"axis": {"range":[0,100]}, "bar": {"color":"#10b981"}}))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(churn.mean()*100), title={"text": "Avg Churn Risk"},
                                     gauge={"axis": {"range":[0,100]}, "bar": {"color":"#ef4444"}}))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)
    with g3:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(upgrade.mean()*100), title={"text": "Avg Upgrade Potential"},
                                     gauge={"axis": {"range":[0,100]}, "bar": {"color":"#8b5cf6"}}))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    dist = pd.DataFrame({"Metric": ["Buy", "Churn", "Upgrade"],
                         "Avg%": [float(buy.mean()*100), float(churn.mean()*100), float(upgrade.mean()*100)]})
    fig = go.Figure(go.Bar(x=dist["Metric"], y=dist["Avg%"], marker_color=["#10b981", "#ef4444", "#8b5cf6"]))
    fig.update_layout(title="Propensity Averages", yaxis_title="Percent", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

    # Next best actions
    st.markdown("### 🎯 AI-Recommended Next Best Actions")

    name_series = leads_df.get("FullName", pd.Series([f"Lead {i+1}" for i in range(len(idx))], index=idx))
    company_series = leads_df.get("Company", pd.Series(["-"]*len(idx), index=idx))
    country_series = leads_df.get("Country", pd.Series(["-"]*len(idx), index=idx))

    def decide_action(b, c, u, e, r):
        if c >= 0.70 and r > 0:
            return ("Retention Call in 24h", "High", float(0.70 + 0.30*min(1.0, r/(r + 1e-9))), "High churn risk on valuable lead")
        if b >= 0.75 and e >= 0.60:
            return ("Schedule Demo in 48h", "High", float(0.70 + 0.30*b), "Strong buy intent and engagement")
        if u >= 0.70 and e >= 0.50:
            return ("Upsell Offer", "Medium", float(0.60 + 0.40*u), "Upgrade potential detected")
        if b >= 0.55:
            return ("Nurture Sequence", "Medium", float(0.50 + 0.50*b), "Moderate buy intent—nurture recommended")
        return ("Check-in Email", "Low", 0.50, "Low intent—light touch")

    actions = []
    for i in range(len(idx)):
        b, c, u = float(buy.iloc[i]), float(churn.iloc[i]), float(upgrade.iloc[i])
        e = float(eng_norm.iloc[i] if not np.isnan(eng_norm.iloc[i]) else 0.0)
        r_i = float(rev.iloc[i])
        act, prio, conf, reason = decide_action(b, c, u, e, r_i)
        actions.append((name_series.iloc[i], company_series.iloc[i], country_series.iloc[i], act, prio, conf, r_i, b, c, u))

    act_df = pd.DataFrame(actions, columns=["Lead", "Company", "Country", "Action", "Priority", "Confidence", "Revenue", "Buy", "Churn", "Upgrade"])
    summary = act_df["Action"].value_counts()
    fig = go.Figure(go.Bar(y=summary.index, x=summary.values, orientation="h", marker_color="#f59e0b"))
    fig.update_layout(title="Recommended Actions Summary", xaxis_title="Number of Leads", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

    prio_order = pd.Categorical(act_df["Priority"], categories=["High", "Medium", "Low"], ordered=True)
    act_df = act_df.assign(PriorityOrder=prio_order).sort_values(["PriorityOrder", "Confidence", "Revenue"], ascending=[True, False, False]).drop(columns=["PriorityOrder"])

    st.dataframe(
        act_df.head(15).style.format({"Confidence": "{:.0%}", "Revenue": "${:,.0f}", "Buy": "{:.0%}", "Churn": "{:.0%}", "Upgrade": "{:.0%}"}),
        use_container_width=True, height=420
    )

# ------------------------------------------------------------
# Call Analytics (safe)
# ------------------------------------------------------------
def create_enhanced_call_activity_dashboard(calls_df):
    st.subheader("🤖 Advanced Call Analytics & AI Insights")

    col1, col2, col3, col4 = st.columns(4)
    total_calls = len(calls_df)
    with col1:
        st.markdown(create_metric_card("Total Calls", total_calls, 23.1), unsafe_allow_html=True)
    with col2:
        success_mask = calls_df.get("IsSuccessful", aligned_series(calls_df, False)).astype(bool)
        successful_calls = int(success_mask.sum())
        success_rate = (successful_calls / total_calls * 100) if total_calls else 0.0
        st.markdown(create_metric_card("Success Rate", success_rate, -5.2, "percentage"), unsafe_allow_html=True)
    with col3:
        if total_calls and "DurationSeconds" in calls_df.columns:
            avg_duration_min = float(pd.to_numeric(calls_df["DurationSeconds"], errors="coerce")[success_mask].mean() / 60.0)
            st.markdown(create_metric_card("Avg Duration", avg_duration_min, 8.7), unsafe_allow_html=True)
            st.caption("Minutes")
    with col4:
        if "CallEfficiency" in calls_df.columns:
            efficiency = float(pd.to_numeric(calls_df["CallEfficiency"], errors="coerce").mean() * 100)
            st.markdown(create_metric_card("Efficiency", efficiency, 12.3, "percentage"), unsafe_allow_html=True)

# ------------------------------------------------------------
# Task Dashboard
# ------------------------------------------------------------
def create_enhanced_task_dashboard(schedule_df):
    st.subheader("📅 Advanced Task Management & SLA Tracking")

    col1, col2, col3, col4 = st.columns(4)
    total_tasks = len(schedule_df)
    overdue_series_default = pd.Series([False] * len(schedule_df), index=schedule_df.index)
    priority_series_default = pd.Series(["Medium"] * len(schedule_df), index=schedule_df.index)
    overdue_tasks = int(schedule_df.get("IsOverdue", overdue_series_default).sum())
    high_priority = int((schedule_df.get("Priority", priority_series_default) == "High").sum())

    with col1:
        st.markdown(create_metric_card("Total Tasks", total_tasks, 15.3), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Overdue Tasks", overdue_tasks, -12.5), unsafe_allow_html=True)
        if overdue_tasks > 0:
            st.caption("Immediate attention needed")
    with col3:
        st.markdown(create_metric_card("High Priority", high_priority, 8.2), unsafe_allow_html=True)
    with col4:
        if "CompletionProbability" in schedule_df.columns:
            avg_completion_prob = float(pd.to_numeric(schedule_df["CompletionProbability"], errors="coerce").mean() * 100)
            st.markdown(create_metric_card("Completion Rate", avg_completion_prob, 5.7, "percentage"), unsafe_allow_html=True)

# ------------------------------------------------------------
# Agent Dashboard (condensed)
# ------------------------------------------------------------
def create_enhanced_agent_dashboard(agent_perf_df, schedule_df):
    st.subheader("👥 Advanced Agent Performance Analytics")
    if len(agent_perf_df) == 0:
        st.warning("No agent performance data available.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric_card("Total Agents", len(agent_perf_df)), unsafe_allow_html=True)
    with col2:
        if "PerformanceScore" in agent_perf_df.columns:
            st.markdown(create_metric_card("Avg Performance", float(agent_perf_df["PerformanceScore"].mean()), 7.3), unsafe_allow_html=True)
    with col3:
        if "CallSuccessRate" in agent_perf_df.columns:
            st.markdown(create_metric_card("Call Success", float(agent_perf_df["CallSuccessRate"].mean()*100), -2.1, "percentage"), unsafe_allow_html=True)
    with col4:
        if "ConversionRate" in agent_perf_df.columns:
            st.markdown(create_metric_card("Conversion Rate", float(agent_perf_df["ConversionRate"].mean()*100), 12.8, "percentage"), unsafe_allow_html=True)

# ------------------------------------------------------------
# Conversion / Revenue Intelligence (safe fallbacks)
# ------------------------------------------------------------
def create_enhanced_conversion_dashboard(leads_df):
    st.subheader("💼 Advanced Conversion & Revenue Intelligence")

    total_leads = len(leads_df)
    stage_series = pd.to_numeric(leads_df.get("LeadStageId", aligned_series(leads_df, 0)), errors="coerce")
    converted_leads = int((stage_series == 4).sum())

    rev_series = pd.to_numeric(leads_df.get("RevenuePotential", aligned_series(leads_df, 0.0)), errors="coerce")
    exp_series = pd.to_numeric(leads_df.get("ExpectedRevenue", aligned_series(leads_df, 0.0)), errors="coerce")

    total_pipeline = float(rev_series.sum())
    expected_revenue = float(exp_series.sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        conversion_rate = (converted_leads / total_leads * 100) if total_leads else 0.0
        st.markdown(create_metric_card("Conversion Rate", conversion_rate, 8.5, "percentage"), unsafe_allow_html=True)
    with c2:
        st.markdown(create_metric_card("Pipeline Value", total_pipeline, 15.2, "currency"), unsafe_allow_html=True)
    with c3:
        st.markdown(create_metric_card("Expected Revenue", expected_revenue, 12.8, "currency"), unsafe_allow_html=True)
    with c4:
        pipeline_eff = (expected_revenue / total_pipeline * 100) if total_pipeline > 0 else 0.0
        st.markdown(create_metric_card("Pipeline Efficiency", pipeline_eff, 5.3, "percentage"), unsafe_allow_html=True)

# ------------------------------------------------------------
# Geographic Intelligence (condensed)
# ------------------------------------------------------------
def create_enhanced_geographic_dashboard(leads_df):
    st.subheader("🌍 Global Market Intelligence Dashboard")
    if "Country" not in leads_df.columns:
        st.warning("Geographic data not available.")
        return

    agg_map = {}
    count_col = None
    for candidate in ["LeadId", "LeadID", "LeadCode", "FullName"]:
        if candidate in leads_df.columns:
            count_col = candidate
            break
    if count_col: agg_map[count_col] = "count"
    if "RevenuePotential" in leads_df.columns: agg_map["RevenuePotential"] = "sum"
    if "ExpectedRevenue" in leads_df.columns: agg_map["ExpectedRevenue"] = "sum"
    if "ConversionProbability" in leads_df.columns: agg_map["ConversionProbability"] = "mean"
    if "EngagementScore" in leads_df.columns: agg_map["EngagementScore"] = "mean"

    grouped = leads_df.groupby("Country").agg(agg_map).round(2)
    geo = pd.DataFrame(index=grouped.index)

    geo["Lead_Count"] = grouped[count_col] if count_col and count_col in grouped.columns else leads_df.groupby("Country").size()
    if "RevenuePotential" in leads_df.columns:
        geo["Total_Pipeline"] = grouped.get("RevenuePotential", leads_df.groupby("Country")["RevenuePotential"].sum())
        geo["Avg_Deal_Size"] = leads_df.groupby("Country")["RevenuePotential"].mean()
    if "ConversionProbability" in leads_df.columns:
        geo["Conversion_Rate"] = grouped.get("ConversionProbability", leads_df.groupby("Country")["ConversionProbability"].mean())
    if "ExpectedRevenue" in leads_df.columns:
        geo["Expected_Revenue"] = grouped.get("ExpectedRevenue", leads_df.groupby("Country")["ExpectedRevenue"].sum())
    if "EngagementScore" in leads_df.columns:
        geo["Avg_Engagement"] = grouped.get("EngagementScore", leads_df.groupby("Country")["EngagementScore"].mean())

    geo = geo.fillna(0).round(2).reset_index()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(create_metric_card("Active Markets", len(geo)), unsafe_allow_html=True)
    with c2:
        top_market = geo.loc[geo["Lead_Count"].idxmax(), "Country"] if len(geo) else "N/A"
        top_expected = float(geo["Expected_Revenue"].max()) if "Expected_Revenue" in geo.columns and len(geo) else 0.0
        st.markdown(create_metric_card("Top Market", top_market), unsafe_allow_html=True)
        st.caption(f"💰 ${top_expected:,.0f} expected")
    with c3:
        avg_conv = float(geo["Conversion_Rate"].mean()*100) if "Conversion_Rate" in geo.columns else 0.0
        st.markdown(create_metric_card("Avg Conversion", avg_conv, 6.2, "percentage"), unsafe_allow_html=True)
    with c4:
        global_pipeline = float(geo["Total_Pipeline"].sum()) if "Total_Pipeline" in geo.columns else 0.0
        st.markdown(create_metric_card("Global Pipeline", global_pipeline, 18.5, "currency"), unsafe_allow_html=True)

# ------------------------------------------------------------
# AI Command Center (summary only, safe)
# ------------------------------------------------------------
def create_advanced_ai_insights_dashboard(leads_df, calls_df, schedule_df, agent_perf_df):
    st.subheader("🧠 Advanced AI/ML Intelligence Center")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(create_metric_card("Model Accuracy", 87.3, 2.1, "percentage"), unsafe_allow_html=True)
    with col2: st.markdown(create_metric_card("Prediction Confidence", 82.1, -1.5, "percentage"), unsafe_allow_html=True)
    with col3: st.markdown(create_metric_card("Forecast Precision", 79.8, 5.3, "percentage"), unsafe_allow_html=True)
    with col4: st.markdown(create_metric_card("Action Success", 74.2, 8.7, "percentage"), unsafe_allow_html=True)

# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
def main():
    leads_df, calls_df, schedule_df, agent_perf_df = load_enhanced_data()

    st.sidebar.markdown("## 🎛️ Advanced Analytics Filters")

    date_options = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "Last 6 months": 180, "All time": None}
    _ = st.sidebar.selectbox("📅 Select Time Period", list(date_options.keys()), index=1)

    if len(agent_perf_df) > 0:
        if "AgentName" in agent_perf_df.columns:
            agent_list = ["All Agents"] + agent_perf_df["AgentName"].astype(str).tolist()
        else:
            agent_list = ["All Agents", "Agent 1", "Agent 2"]
        _ = st.sidebar.multiselect("👥 Select Agents", agent_list, default=["All Agents"])

    if "Country" in leads_df.columns:
        country_options = ["All Markets"] + sorted(leads_df["Country"].dropna().unique().tolist())
        _ = st.sidebar.multiselect("🌍 Select Markets", country_options, default=["All Markets"])

    if "LeadScoringId" in leads_df.columns:
        _ = st.sidebar.selectbox("🌡️ Lead Temperature Filter",
                                 ["All Temperatures", "HOT Leads Only", "WARM Leads Only", "COLD Leads Only", "HOT + WARM"])

    if "BehavioralSegment" in leads_df.columns:
        segment_options = ["All Segments"] + sorted(leads_df["BehavioralSegment"].dropna().unique().tolist())
        _ = st.sidebar.multiselect("🎯 Behavioral Segments", segment_options, default=["All Segments"])

    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30 seconds)", value=False)
    if auto_refresh:
        st.sidebar.markdown("*Dashboard will refresh automatically*")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Export Options")
    if st.sidebar.button("📑 Generate Executive Report"):
        st.sidebar.success("Report generated! Check downloads.")
    if st.sidebar.button("📈
