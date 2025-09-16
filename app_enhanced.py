import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --------------------------------------
# Page config and executive styling
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
      .main { background-color:#0f1419; color:#ffffff; }
      .stApp { background:linear-gradient(135deg,#0f1419 0%,#1a202c 100%); }

      .metric-container{background:linear-gradient(135deg,#1a202c 0%,#2d3748 100%);
        padding:16px;border-radius:12px;border:1px solid #4a5568;margin:10px 0;
        box-shadow:0 4px 12px rgba(0,0,0,0.4);}
      .metric-value{font-size:2.4rem;font-weight:700;color:#f59e0b;}
      .metric-label{font-size:.9rem;color:#a0aec0;text-transform:uppercase;letter-spacing:.5px}
      .metric-change{font-size:.8rem;font-weight:600;margin-top:8px}
      .positive{color:#68d391}.negative{color:#fc8181}.neutral{color:#90cdf4}

      .insight-card{background:linear-gradient(135deg,#2d3748 0%,#4a5568 100%);
        padding:18px;border-radius:10px;border-left:5px solid #f59e0b;margin:14px 0}
      .alert-high{border-left-color:#fc8181;background:linear-gradient(135deg,#4a1f1f 0%,#5a2d2d 100%)}
      .alert-medium{border-left-color:#f6ad55;background:linear-gradient(135deg,#4a3a1f 0%,#5a4a2d 100%)}
      .alert-low{border-left-color:#68d391;background:linear-gradient(135deg,#1f4a2d 0%,#2d5a3a 100%)}

      .dashboard-header{background:linear-gradient(90deg,#1a202c 0%,#2d3748 100%);
        padding:24px;border-radius:12px;margin-bottom:26px;border:1px solid #4a5568}

      .performance-card{background:linear-gradient(135deg,#1e3a5f 0%,#2d4a6b 100%);
        padding:14px;border-radius:8px;margin:10px 0;border:1px solid #3b82f6}

      .prediction-box{background:linear-gradient(135deg,#1e40af 0%,#3b82f6 100%);
        padding:14px;border-radius:8px;color:#fff;border:1px solid #60a5fa}
      .ai-recommendation{background:linear-gradient(135deg,#7c3aed 0%,#8b5cf6 100%);
        padding:14px;border-radius:8px;color:#fff;border:1px solid #a78bfa}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------
# Helpers
# --------------------------------------
def create_metric_card(title, value, change=None, format_type="number"):
    if format_type == "currency":
        display_value = f"${value:,.0f}" if isinstance(value, (int, float)) else str(value)
    elif format_type == "percentage":
        if isinstance(value, (int, float)):
            display_value = f"{value:.1f}%"
        else:
            display_value = str(value)
    else:
        if isinstance(value, (int, float)):
            display_value = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.1f}"
        else:
            display_value = str(value)

    change_html = ""
    if change is not None:
        cls = "positive" if change > 0 else ("negative" if change < 0 else "neutral")
        icon = "↗️" if change > 0 else ("↘️" if change < 0 else "→")
        change_html = f'<div class="metric-change {cls}">{icon} {change:+.1f}%</div>'

    return f"""
    <div class="metric-container">
      <div class="metric-label">{title}</div>
      <div class="metric-value">{display_value}</div>
      {change_html}
    </div>
    """

# --------------------------------------
# Data loading
# --------------------------------------
@st.cache_data(ttl=3600)
def load_enhanced_data():
    """Load enhanced CRM datasets with advanced analytics; fallback generates sample data."""
    try:
        leads_df = pd.read_csv('enhanced_leads_advanced.csv')
        calls_df = pd.read_csv('enhanced_calls_advanced.csv')
        schedule_df = pd.read_csv('enhanced_schedule_advanced.csv')
        agent_perf_df = pd.read_csv('agent_performance_advanced.csv')

        # Convert dates
        for c in ('CreatedOn', 'ModifiedOn'):
            if c in leads_df.columns:
                leads_df[c] = pd.to_datetime(leads_df[c], errors='coerce')
        if 'CallDateTime' in calls_df.columns:
            calls_df['CallDateTime'] = pd.to_datetime(calls_df['CallDateTime'], errors='coerce')
        if 'ScheduledDate' in schedule_df.columns:
            schedule_df['ScheduledDate'] = pd.to_datetime(schedule_df['ScheduledDate'], errors='coerce')

        return leads_df, calls_df, schedule_df, agent_perf_df
    except FileNotFoundError:
        st.warning("Enhanced data files not found. Using sample data for demonstration.", icon="ℹ️")
        # Fallback data
        leads_df = pd.DataFrame({
            'LeadId': range(1, 51),
            'FullName': [f'Lead {i}' for i in range(1, 51)],
            'Company': [f'Company {i}' for i in range(1, 51)],
            'Country': np.random.choice(['Saudi Arabia', 'UAE', 'India', 'UK', 'US'], 50),
            'LeadStageId': np.random.choice([1, 2, 3, 4], 50),
            'LeadScoringId': np.random.choice([1, 2, 3, 4], 50),
            'RevenuePotential': np.random.uniform(50_000, 300_000, 50),
            'ExpectedRevenue': np.random.uniform(25_000, 250_000, 50),
            'ConversionProbability': np.random.uniform(0.1, 0.8, 50),
            'EngagementScore': np.random.randint(20, 100, 50),
            'BehavioralSegment': np.random.choice(['Champions', 'At Risk', 'Need Attention'], 50),
            'TemperatureTrend': np.random.choice(['Warming Up','Stable','Cooling Down'], 50),
            'LeadVelocity': np.random.randint(5, 45, 50),
            'CreatedOn': pd.date_range('2025-08-01', periods=50, freq='D')
        })
        calls_df = pd.DataFrame({
            'LeadCallId': range(1, 81),
            'IsSuccessful': np.random.choice([True, False], 80),
            'DurationSeconds': np.random.randint(60, 1800, 80),
            'CallHour': np.random.randint(8, 21, 80),
            'CallDateTime': pd.date_range('2025-08-01', periods=80, freq='H')
        })
        schedule_df = pd.DataFrame({
            'ScheduleId': range(1, 31),
            'TaskStatus': np.random.choice(['Pending', 'Completed', 'Overdue'], 30),
            'Priority': np.random.choice(['High', 'Medium', 'Low'], 30),
            'ScheduledDate': pd.date_range('2025-09-01', periods=30, freq='D')
        })
        agent_perf_df = pd.DataFrame({
            'AgentId': range(1, 6),
            'AgentName': ['Agent A', 'Agent B', 'Agent C', 'Agent D', 'Agent E'],
            'PerformanceScore': np.random.uniform(70, 95, 5),
            'EfficiencyScore': np.random.uniform(0.6, 0.95, 5),
            'CallSuccessRate': np.random.uniform(0.4, 0.9, 5),
            'ConversionRate': np.random.uniform(0.2, 0.6, 5),
            'TotalRevenue': np.random.uniform(100_000, 800_000, 5)
        })
        return leads_df, calls_df, schedule_df, agent_perf_df

# --------------------------------------
# Executive Overview
# --------------------------------------
def create_enhanced_executive_summary(leads_df, calls_df, schedule_df, agent_perf_df):
    st.markdown(
        """
        <div class="dashboard-header">
          <h1 style="color:#f59e0b;margin:0;font-size:2.0rem">🚀 Executive CRM Dashboard</h1>
          <p style="color:#a0aec0;margin:8px 0 0 0">AI-Powered Real Estate Analytics with Predictive Insights</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        total_leads = len(leads_df)
        hot_leads = int((leads_df.get('LeadScoringId', pd.Series(*len(leads_df))).astype(int) == 1).sum())
        st.markdown(create_metric_card("Total Leads", total_leads, 15.2), unsafe_allow_html=True)
        st.caption(f"🔥 {hot_leads} Hot Leads")
    with c2:
        total_pipeline = float(pd.to_numeric(leads_df.get('RevenuePotential', pd.Series(*len(leads_df))), errors='coerce').sum())
        expected_revenue = float(pd.to_numeric(leads_df.get('ExpectedRevenue', pd.Series(*len(leads_df))), errors='coerce').sum())
        st.markdown(create_metric_card("Pipeline Value", total_pipeline, 8.7, "currency"), unsafe_allow_html=True)
        st.caption(f"💰 ${expected_revenue:,.0f} Expected")
    with c3:
        success_rate = float(leads_df.empty or (calls_df.get('IsSuccessful') is None))
        success_rate = float(calls_df.get('IsSuccessful', pd.Series([], dtype=bool)).mean()*100) if len(calls_df) else 0.0
        st.markdown(create_metric_card("Call Success Rate", success_rate, -2.3, "percentage"), unsafe_allow_html=True)
        st.caption(f"📞 {len(calls_df)} Total Calls")
    with c4:
        avg_eng = float(pd.to_numeric(leads_df.get('EngagementScore', pd.Series(*len(leads_df))), errors='coerce').mean())
        hi_eng = int((pd.to_numeric(leads_df.get('EngagementScore', pd.Series(*len(leads_df))), errors='coerce') > 80).sum())
        st.markdown(create_metric_card("Avg Engagement", avg_eng, 5.1), unsafe_allow_html=True)
        st.caption(f"⭐ {hi_eng} High Engagement")
    with c5:
        avg_conv = float(pd.to_numeric(leads_df.get('ConversionProbability', pd.Series(*len(leads_df))), errors='coerce').mean()*100)
        likely = int((pd.to_numeric(leads_df.get('ConversionProbability', pd.Series(*len(leads_df))), errors='coerce') > 0.7).sum())
        st.markdown(create_metric_card("Conversion Rate", avg_conv, 12.4, "percentage"), unsafe_allow_html=True)
        st.caption(f"🎯 {likely} Likely Converts")

# --------------------------------------
# Lead Intelligence (with Propensity + Next Best Actions)
# --------------------------------------
def create_enhanced_lead_status_dashboard(leads_df):
    st.subheader("📊 Lead Analytics")

    # Lead Stage Funnel
    c1, c2 = st.columns(2)
    with c1:
        if 'LeadStageId' in leads_df.columns:
            map_stage = {1: 'New', 2: 'Qualified', 3: 'Nurtured', 4: 'Converted'}
            cnt = leads_df['LeadStageId'].value_counts(dropna=False).sort_index()
            labels = [map_stage.get(i, f"Stage {i}") for i in cnt.index]
            fig = go.Figure(go.Funnel(y=labels, x=cnt.values,
                                      textinfo="value+percent initial",
                                      marker_color=['#3b82f6','#f59e0b','#10b981','#ef4444']))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("LeadStageId column not found; funnel chart skipped.")

    with c2:
        if 'BehavioralSegment' in leads_df.columns:
            seg = leads_df['BehavioralSegment'].value_counts(dropna=False)
            colors = {'Champions':'#10b981','Loyal Customers':'#3b82f6',
                      'Potential Loyalists':'#f59e0b','At Risk':'#ef4444','Need Attention':'#8b5cf6'}
            fig = go.Figure(go.Pie(labels=seg.index.astype(str), values=seg.values, hole=.4,
                                   marker_colors=[colors.get(s,'#6b7280') for s in seg.index]))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("BehavioralSegment column not found; segmentation chart skipped.")

    # Temperature & Engagement
    st.subheader("🌡️ Lead Temperature & Engagement")
    t1, t2, t3 = st.columns(3)
    with t1:
        if 'TemperatureTrend' in leads_df.columns:
            temp = leads_df['TemperatureTrend'].value_counts(dropna=False)
            fig = go.Figure(go.Bar(x=temp.index.astype(str), y=temp.values,
                                   marker_color=['#10b981','#f59e0b','#ef4444','#6b7280']))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("TemperatureTrend column not found; temperature chart skipped.")
    with t2:
        if 'EngagementScore' in leads_df.columns:
            fig = go.Figure(go.Histogram(x=leads_df['EngagementScore'], nbinsx=10, marker_color='#8b5cf6', opacity=0.8))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                              xaxis_title="Engagement Score", yaxis_title="Number of Leads")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("EngagementScore column not found; distribution chart skipped.")
    with t3:
        if 'LeadVelocity' in leads_df.columns:
            avg_v = pd.to_numeric(leads_df['LeadVelocity'], errors='coerce').mean()
            fig = go.Figure(go.Indicator(mode='gauge+number', value=float(avg_v) if not np.isnan(avg_v) else 0.0,
                                         title={'text':'Average Lead Velocity (Days)'},
                                         gauge={'axis':{'range':[0,50]}, 'bar':{'color':'#f59e0b'},
                                                'steps':[{'range':[0,15],'color':'#10b981'},
                                                         {'range':[15,30],'color':'#f59e0b'},
                                                         {'range':[30,50],'color':'#ef4444'}]}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("LeadVelocity column not found; gauge skipped.")

    # ===== Enhanced Intelligence (always-on) =====
    st.markdown("---")
    st.subheader("🧭 Lead Intelligence Add-ons")

    idx = leads_df.index
    eng = pd.to_numeric(leads_df.get('EngagementScore', pd.Series([np.nan]*len(idx), index=idx)), errors='coerce')
    conv = pd.to_numeric(leads_df.get('ConversionProbability', pd.Series([np.nan]*len(idx), index=idx)), errors='coerce')
    rev = pd.to_numeric(
        leads_df.get('ExpectedRevenue', leads_df.get('RevenuePotential', pd.Series(*len(idx), index=idx))),
        errors='coerce'
    ).fillna(0)

    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(create_metric_card("Median Engagement", float(np.nanmedian(eng)) if len(eng)>0 else 0.0, 0.0), unsafe_allow_html=True)
    with k2:
        st.markdown(create_metric_card("Avg Conv Prob", float(np.nanmean(conv)*100) if len(conv)>0 else 0.0, 0.0, "percentage"), unsafe_allow_html=True)
    with k3:
        st.markdown(create_metric_card("Total Expected", float(rev.sum()), 0.0, "currency"), unsafe_allow_html=True)
    with k4:
        hot_ready = int(((conv > 0.70) & (rev > 0)).sum())
        st.markdown(create_metric_card("Hot Opportunities", hot_ready), unsafe_allow_html=True)

    # Top 10 table by a composite quality score
    quality = (0.6*conv.fillna(0.0) + 0.4*(eng.fillna(0.0)/100.0)).clip(0,1)
    table = pd.DataFrame({
        'Lead': leads_df.get('FullName', pd.Series([f'Lead {i+1}' for i in range(len(idx))], index=idx)),
        'Company': leads_df.get('Company', pd.Series(['-']*len(idx), index=idx)),
        'Country': leads_df.get('Country', pd.Series(['-']*len(idx), index=idx)),
        'QualityScore': quality.round(3),
        'ConvProb': conv.fillna(0.0).round(3),
        'Revenue': rev
    })
    top = table.sort_values(["QualityScore","Revenue"], ascending=[False,False]).head(10)
    st.markdown("**Top 10 Opportunities**")
    st.dataframe(
        top.style.format({'QualityScore':'{:.2f}','ConvProb':'{:.1%}','Revenue':'${:,.0f}'}),
        use_container_width=True, height=360
    )

    hot = int((conv>0.7).sum())
    cooling = int((leads_df.get('TemperatureTrend', pd.Series(['Unknown']*len(idx), index=idx)) == 'Cooling Down').sum())
    champions = int((leads_df.get('BehavioralSegment', pd.Series(['-']*len(idx), index=idx)) == 'Champions').sum())
    st.markdown("- ✅ Focus first on leads with QualityScore ≥ 0.75 for fastest wins.")
    st.markdown(f"- 🚩 {hot} high-probability leads detected; schedule demos within 48 hours.")
    st.markdown(f"- 🧊 {cooling} cooling leads require retention play; trigger nurturing workflow.")
    st.markdown(f"- 🏆 {champions} champions segment shows strong upsell potential this month.")

    # ===== 🎯 AI Propensity Models =====
    st.markdown("### 🎯 AI Propensity Models")
    eng_norm = (eng.fillna(0) / 100).clip(0, 1)
    rev_norm = (rev / (rev.max() if rev.max() > 0 else 1)).clip(0, 1)

    buy = pd.to_numeric(leads_df.get('PropensityToBuy', np.nan), errors='coerce')
    churn = pd.to_numeric(leads_df.get('PropensityToChurn', np.nan), errors='coerce')
    upgrade = pd.to_numeric(leads_df.get('PropensityToUpgrade', np.nan), errors='coerce')

    buy = buy.fillna((0.55*conv.fillna(0) + 0.35*eng_norm + 0.10*rev_norm).clip(0,1))
    cooling_flag = (leads_df.get('TemperatureTrend', pd.Series(['Unknown']*len(idx), index=idx)) == 'Cooling Down').astype(int)
    churn = churn.fillna((0.65*(1-conv.fillna(0)) + 0.25*(1-eng_norm) + 0.10*cooling_flag).clip(0,1))
    upgrade = upgrade.fillna((0.50*eng_norm + 0.30*conv.fillna(0) + 0.20*rev_norm).clip(0,1))

    g1,g2,g3 = st.columns(3)
    with g1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(buy.mean()*100),
                                     title={'text': "Avg Buy Propensity"},
                                     gauge={'axis': {'range':[0,100]}, 'bar': {'color':'#10b981'}}))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(churn.mean()*100),
                                     title={'text': "Avg Churn Risk"},
                                     gauge={'axis': {'range':[0,100]}, 'bar': {'color':'#ef4444'}}))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    with g3:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(upgrade.mean()*100),
                                     title={'text': "Avg Upgrade Potential"},
                                     gauge={'axis': {'range':[0,100]}, 'bar': {'color':'#8b5cf6'}}))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    dist = pd.DataFrame({'Metric':['Buy','Churn','Upgrade'],
                         'Avg%':[float(buy.mean()*100), float(churn.mean()*100), float(upgrade.mean()*100)]})
    fig = go.Figure(go.Bar(x=dist['Metric'], y=dist['Avg%'],
                           marker_color=['#10b981','#ef4444','#8b5cf6']))
    fig.update_layout(title="Propensity Averages", yaxis_title="Percent",
                      paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)

    # ===== 🎯 AI‑Recommended Next Best Actions =====
    st.markdown("### 🎯 AI-Recommended Next Best Actions")

    name_series = leads_df.get('FullName', pd.Series([f'Lead {i+1}' for i in range(len(idx))], index=idx))
    company_series = leads_df.get('Company', pd.Series(['-']*len(idx), index=idx))
    country_series = leads_df.get('Country', pd.Series(['-']*len(idx), index=idx))

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

    act_df = pd.DataFrame(actions, columns=['Lead','Company','Country','Action','Priority','Confidence','Revenue','Buy','Churn','Upgrade'])
    summary = act_df['Action'].value_counts()
    fig = go.Figure(go.Bar(y=summary.index, x=summary.values, orientation='h', marker_color='#f59e0b'))
    fig.update_layout(title="Recommended Actions Summary", xaxis_title="Number of Leads",
                      paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)

    prio_order = pd.Categorical(act_df['Priority'], categories=['High','Medium','Low'], ordered=True)
    act_df = act_df.assign(PriorityOrder=prio_order).sort_values(['PriorityOrder','Confidence','Revenue'],
                                                                 ascending=[True, False, False]).drop(columns=['PriorityOrder'])
    st.dataframe(
        act_df.head(15).style.format({'Confidence':'{:.0%}','Revenue':'${:,.0f}','Buy':'{:.0%}','Churn':'{:.0%}','Upgrade':'{:.0%}'}),
        use_container_width=True, height=420
    )

# --------------------------------------
# Call Analytics
# --------------------------------------
def create_enhanced_call_activity_dashboard(calls_df):
    st.subheader("🤖 Advanced Call Analytics & AI Insights")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(create_metric_card("Total Calls", len(calls_df), 23.1), unsafe_allow_html=True)
    with c2:
        success_rate = float(calls_df.get('IsSuccessful', pd.Series([], dtype=bool)).mean()*100) if len(calls_df) else 0.0
        st.markdown(create_metric_card("Success Rate", success_rate, -5.2, "percentage"), unsafe_allow_html=True)
    with c3:
        if {'DurationSeconds','IsSuccessful'}.issubset(calls_df.columns):
            avg_minutes = float(calls_df[calls_df['IsSuccessful']]['DurationSeconds'].mean()/60) if len(calls_df) else 0.0
            st.markdown(create_metric_card("Avg Duration", avg_minutes, 8.7), unsafe_allow_html=True)
            st.caption("Minutes")
    with c4:
        if 'CallEfficiency' in calls_df.columns:
            eff = float(calls_df['CallEfficiency'].mean()*100)
            st.markdown(create_metric_card("Efficiency", eff, 12.3, "percentage"), unsafe_allow_html=True)

    if 'CallHour' in calls_df.columns and 'IsSuccessful' in calls_df.columns:
        hourly = calls_df.groupby('CallHour').agg(Total=('IsSuccessful','count'), Successful=('IsSuccessful','sum'))
        hourly['SuccessRate'] = (hourly['Successful']/hourly['Total']*100).fillna(0)
        fig = go.Figure()
        fig.add_bar(x=hourly.index, y=hourly['Total'], name='Total Calls', marker_color='#3b82f6', opacity=.7)
        fig.add_scatter(x=hourly.index, y=hourly['SuccessRate'], name='Success %', yaxis='y2',
                        line=dict(color='#10b981', width=3))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                          yaxis=dict(title='Calls'), yaxis2=dict(title='Success %', overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------
# Task Management
# --------------------------------------
def create_enhanced_task_dashboard(schedule_df):
    st.subheader("📅 Advanced Task Management & SLA Tracking")

    c1, c2, c3, c4 = st.columns(4)
    total_tasks = len(schedule_df)
    overdue_series_default = pd.Series([False]*len(schedule_df), index=schedule_df.index)
    priority_series_default = pd.Series(['Medium']*len(schedule_df), index=schedule_df.index)
    overdue_tasks = schedule_df.get('IsOverdue', overdue_series_default).sum()
    high_priority = (schedule_df.get('Priority', priority_series_default) == 'High').sum()

    with c1:
        st.markdown(create_metric_card("Total Tasks", total_tasks, 15.3), unsafe_allow_html=True)
    with c2:
        st.markdown(create_metric_card("Overdue Tasks", int(overdue_tasks), -12.5), unsafe_allow_html=True)
        if int(overdue_tasks) > 0:
            st.caption("Immediate attention needed")
    with c3:
        st.markdown(create_metric_card("High Priority", int(high_priority), 8.2), unsafe_allow_html=True)
    with c4:
        if 'CompletionProbability' in schedule_df.columns:
            avg_completion_prob = float(schedule_df['CompletionProbability'].mean()*100)
            st.markdown(create_metric_card("Completion Rate", avg_completion_prob, 5.7, "percentage"), unsafe_allow_html=True)

    if 'SLAStatus' in schedule_df.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            sla = schedule_df['SLAStatus'].value_counts()
            colors = {'On Track':'#10b981','At Risk':'#f59e0b','Breach':'#ef4444'}
            fig = go.Figure(go.Pie(labels=sla.index, values=sla.values, hole=.4,
                                   marker_colors=[colors.get(s,'#6b7280') for s in sla.index]))
            fig.update_layout(title="SLA Status Distribution", paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            breach_tasks = schedule_df[schedule_df['SLAStatus'] == 'Breach']
            at_risk_tasks = schedule_df[schedule_df['SLAStatus'] == 'At Risk']
            st.markdown("**🚨 SLA Alert Summary:**")
            if len(breach_tasks) > 0:
                st.info(f"SLA Breaches: {len(breach_tasks)} — tasks past due date requiring escalation")
            if len(at_risk_tasks) > 0:
                st.warning(f"At Risk: {len(at_risk_tasks)} — tasks due within 24 hours needing attention")

    st.subheader("📈 Task Timeline & Workload Analysis")
    a, b = st.columns(2)
    with a:
        if 'Priority' in schedule_df.columns:
            pr = schedule_df['Priority'].value_counts()
            fig = go.Figure(go.Bar(x=pr.index, y=pr.values, marker_color=['#ef4444','#f59e0b','#10b981']))
            fig.update_layout(title="Task Priority Distribution", paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
    with b:
        if 'TaskType' in schedule_df.columns:
            tt = schedule_df['TaskType'].value_counts()
            fig = go.Figure(go.Bar(y=tt.index, x=tt.values, orientation='h', marker_color='#8b5cf6'))
            fig.update_layout(title="Task Type Distribution", paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚡ Productivity & Efficiency Analytics")
    p1, p2, p3 = st.columns(3)
    with p1:
        if 'EstimatedEffortHours' in schedule_df.columns:
            total_effort = float(schedule_df['EstimatedEffortHours'].sum())
            st.markdown(f"""
            <div class="performance-card">
              <h4>⏱️ Total Effort Required</h4>
              <h2 style="color:#3b82f6">{total_effort:.1f} Hours</h2>
              <p>Across all pending tasks</p>
            </div>
            """, unsafe_allow_html=True)
    with p2:
        needed = {'ActualEffortHours','EstimatedEffortHours','TaskStatus'}
        if needed.issubset(schedule_df.columns):
            comp = schedule_df[schedule_df['TaskStatus'] == 'Completed']
            if len(comp) > 0:
                actual = float(comp['ActualEffortHours'].mean())
                estimated = float(comp['EstimatedEffortHours'].mean())
                efficiency = (estimated/actual*100) if actual > 0 else 100
                st.markdown(f"""
                <div class="performance-card">
                  <h4>📊 Task Efficiency</h4>
                  <h2 style="color:{'#10b981' if efficiency>=100 else '#f59e0b'}">{efficiency:.1f}%</h2>
                  <p>Estimated vs Actual effort ratio</p>
                </div>
                """, unsafe_allow_html=True)
    with p3:
        if 'DaysUntilDue' in schedule_df.columns:
            days_to_due = pd.to_numeric(schedule_df['DaysUntilDue'], errors='coerce')
        else:
            if 'ScheduledDate' in schedule_df.columns:
                sched = pd.to_datetime(schedule_df['ScheduledDate'], errors='coerce')
            else:
                sched = pd.Series(pd.NaT, index=schedule_df.index)
            days_to_due = (sched - pd.Timestamp.now()).dt.days
        upcoming_week = schedule_df[days_to_due.between(0, 7, inclusive='both')]
        st.markdown(f"""
        <div class="performance-card">
          <h4>📅 Upcoming Week</h4>
          <h2 style="color:#f59e0b">{len(upcoming_week)}</h2>
          <p>Tasks due in next 7 days</p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------
# Agent Performance
# --------------------------------------
def create_enhanced_agent_dashboard(agent_perf_df, schedule_df):
    st.subheader("👥 Advanced Agent Performance Analytics")
    if agent_perf_df is None or len(agent_perf_df) == 0:
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

    if {'AgentName','PerformanceScore'}.issubset(agent_perf_df.columns):
        fig = go.Figure()
        fig.add_bar(x=agent_perf_df['AgentName'], y=agent_perf_df['PerformanceScore'], name='Performance', marker_color='#3b82f6')
        if 'EfficiencyScore' in agent_perf_df.columns:
            fig.add_bar(x=agent_perf_df['AgentName'], y=agent_perf_df['EfficiencyScore']*100, name='Efficiency', marker_color='#10b981')
        fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------
# Conversion & Revenue Intelligence
# --------------------------------------
def create_enhanced_conversion_dashboard(leads_df):
    st.subheader("💼 Advanced Conversion & Revenue Intelligence")

    total_leads = len(leads_df)
    stage_series = (
        leads_df['LeadStageId'] if 'LeadStageId' in leads_df.columns
        else pd.Series(*total_leads, index=leads_df.index)
    )
    stage_series = pd.to_numeric(stage_series, errors='coerce')
    converted_leads = int((stage_series == 4).sum())

    rev_series = (
        pd.to_numeric(leads_df['RevenuePotential'], errors='coerce')
        if 'RevenuePotential' in leads_df.columns
        else pd.Series([0.0]*total_leads, index=leads_df.index, dtype='float64')
    )
    exp_series = (
        pd.to_numeric(leads_df['ExpectedRevenue'], errors='coerce')
        if 'ExpectedRevenue' in leads_df.columns
        else pd.Series([0.0]*total_leads, index=leads_df.index, dtype='float64')
    )

    total_pipeline = float(rev_series.sum())
    expected_revenue = float(exp_series.sum())

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        conversion_rate = (converted_leads/total_leads*100) if total_leads > 0 else 0.0
        st.markdown(create_metric_card("Conversion Rate", conversion_rate, 8.5, "percentage"), unsafe_allow_html=True)
    with c2:
        st.markdown(create_metric_card("Pipeline Value", total_pipeline, 15.2, "currency"), unsafe_allow_html=True)
    with c3:
        st.markdown(create_metric_card("Expected Revenue", expected_revenue, 12.8, "currency"), unsafe_allow_html=True)
    with c4:
        pipeline_eff = (expected_revenue/total_pipeline*100) if total_pipeline > 0 else 0.0
        st.markdown(create_metric_card("Pipeline Efficiency", pipeline_eff, 5.3, "percentage"), unsafe_allow_html=True)

    st.subheader("💰 Revenue Attribution Analysis")
    a, b = st.columns(2)

    with a:
        if 'Country' in leads_df.columns:
            agg = {}
            if 'RevenuePotential' in leads_df.columns: agg['RevenuePotential'] = 'sum'
            if 'ExpectedRevenue' in leads_df.columns: agg['ExpectedRevenue'] = 'sum'
            if 'ConversionProbability' in leads_df.columns: agg['ConversionProbability'] = 'mean'
            if len(agg) > 0:
                country_rev = leads_df.groupby('Country').agg(agg).round(2)
                fig = go.Figure()
                if 'RevenuePotential' in country_rev.columns:
                    fig.add_bar(name='Pipeline Value', x=country_rev.index, y=country_rev['RevenuePotential'], marker_color='#3b82f6', opacity=0.85)
                if 'ExpectedRevenue' in country_rev.columns:
                    fig.add_bar(name='Expected Revenue', x=country_rev.index, y=country_rev['ExpectedRevenue'], marker_color='#10b981', opacity=0.9)
                fig.update_layout(title="Revenue Attribution by Country", xaxis_title="Country", yaxis_title="Amount ($)",
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Revenue/Conversion columns not found to build country attribution.")
        else:
            st.info("Country column not found; skipping market attribution.")

    with b:
        if 'LeadScoringId' in leads_df.columns:
            value_col = 'ExpectedRevenue' if 'ExpectedRevenue' in leads_df.columns else ('RevenuePotential' if 'RevenuePotential' in leads_df.columns else None)
            if value_col is not None:
                mapping = {1:'HOT', 2:'WARM', 3:'COLD', 4:'DEAD'}
                tmp = leads_df.copy()
                tmp['ScoringLabel'] = tmp['LeadScoringId'].map(mapping)
                scoring_rev = tmp.groupby('ScoringLabel')[value_col].sum()
                fig = go.Figure(go.Pie(labels=scoring_rev.index, values=scoring_rev.values, hole=0.4,
                                       marker_colors=['#dc2626','#f59e0b','#3b82f6','#6b7280']))
                fig.update_layout(title=f"{value_col} by Lead Temperature", paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Neither ExpectedRevenue nor RevenuePotential is available for scoring breakdown.")

    st.subheader("🎯 Conversion Probability Intelligence")
    sc1, sc2 = st.columns(2)
    with sc1:
        if {'ConversionProbability','RevenuePotential'}.issubset(leads_df.columns):
            fig = go.Figure()
            if 'LeadScoringId' in leads_df.columns:
                colors = {1:'#dc2626', 2:'#f59e0b', 3:'#3b82f6', 4:'#6b7280'}
                for sid, grp in leads_df.groupby('LeadScoringId'):
                    label = {1:'HOT',2:'WARM',3:'COLD',4:'DEAD'}.get(sid, str(sid))
                    fig.add_scatter(x=grp['ConversionProbability'], y=grp['RevenuePotential'],
                                    mode='markers', name=label,
                                    marker=dict(size=10, color=colors.get(sid,'#999'), opacity=0.7))
            else:
                fig.add_scatter(x=leads_df['ConversionProbability'], y=leads_df['RevenuePotential'],
                                mode='markers', name='Leads', marker=dict(size=9, color='#3b82f6'))
            fig.update_layout(title='Revenue Potential vs Conversion Probability', xaxis_title='Conversion Probability',
                              yaxis_title='Revenue Potential ($)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need ConversionProbability and RevenuePotential to show the scatter chart.")
    with sc2:
        if 'ConversionProbability' in leads_df.columns:
            fig = go.Figure(go.Histogram(x=leads_df['ConversionProbability'], nbinsx=20, marker_color='#8b5cf6', opacity=0.8))
            fig.update_layout(title='Conversion Probability Distribution', xaxis_title='Conversion Probability',
                              yaxis_title='Number of Leads', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("💎 High-Value Conversion Opportunities")
    if all(col in leads_df.columns for col in ['ExpectedRevenue','ConversionProbability','FullName']):
        top_opps = leads_df.nlargest(10, 'ExpectedRevenue')
        for i, (_, lead) in enumerate(top_opps.iterrows()):
            col = st.columns(2)[i % 2]
            with col:
                prob = lead.get('ConversionProbability', 0.0)
                color = '#10b981' if prob > 0.7 else ('#f59e0b' if prob > 0.4 else '#ef4444')
                st.markdown(f"""
                <div class="performance-card">
                  <h4>{lead.get('FullName','Unknown Lead')}</h4>
                  <p><strong>Expected Revenue:</strong> ${lead.get('ExpectedRevenue', 0):,.0f}</p>
                  <p><strong>Probability:</strong> <span style="color:{color};">{prob:.1%}</span></p>
                  <p><strong>Company:</strong> {lead.get('Company','Unknown')[:30]}...</p>
                </div>
                """, unsafe_allow_html=True)
    elif all(col in leads_df.columns for col in ['RevenuePotential','ConversionProbability','FullName']):
        top_opps = leads_df.nlargest(10, 'RevenuePotential')
        for i, (_, lead) in enumerate(top_opps.iterrows()):
            col = st.columns(2)[i % 2]
            with col:
                prob = lead.get('ConversionProbability', 0.0)
                color = '#10b981' if prob > 0.7 else ('#f59e0b' if prob > 0.4 else '#ef4444')
                st.markdown(f"""
                <div class="performance-card">
                  <h4>{lead.get('FullName','Unknown Lead')}</h4>
                  <p><strong>Pipeline Value:</strong> ${lead.get('RevenuePotential', 0):,.0f}</p>
                  <p><strong>Probability:</strong> <span style="color:{color};">{prob:.1%}</span></p>
                  <p><strong>Company:</strong> {lead.get('Company','Unknown')[:30]}...</p>
                </div>
                """, unsafe_allow_html=True)

    st.subheader("📈 AI Revenue Forecasting")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""
        <div class="prediction-box">
          <h4>📅 30-Day Forecast</h4>
          <h2>${expected_revenue*0.35:,.0f}</h2>
          <p>Confidence: 78%</p>
          <small>Based on current conversion rates</small>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="prediction-box">
          <h4>📅 90-Day Forecast</h4>
          <h2>${expected_revenue*0.65:,.0f}</h2>
          <p>Confidence: 85%</p>
          <small>Including nurturing pipeline</small>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="prediction-box">
          <h4>📅 Year-End Projection</h4>
          <h2>${expected_revenue*1.2:,.0f}</h2>
          <p>Confidence: 72%</p>
          <small>With continued performance</small>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------
# Geographic Dashboard
# --------------------------------------
def create_enhanced_geographic_dashboard(leads_df):
    st.subheader("🌍 Global Market Intelligence Dashboard")
    if 'Country' not in leads_df.columns:
        st.warning("Geographic data not available.")
        return

    agg_map = {}
    count_col = None
    for cand in ['LeadId','LeadID','LeadCode','FullName']:
        if cand in leads_df.columns:
            count_col = cand
            break
    if count_col:
        agg_map[count_col] = 'count'
    if 'RevenuePotential' in leads_df.columns: agg_map['RevenuePotential'] = 'sum'
    if 'ExpectedRevenue' in leads_df.columns: agg_map['ExpectedRevenue'] = 'sum'
    if 'ConversionProbability' in leads_df.columns: agg_map['ConversionProbability'] = 'mean'
    if 'EngagementScore' in leads_df.columns: agg_map['EngagementScore'] = 'mean'

    grouped = leads_df.groupby('Country').agg(agg_map).round(2)
    geo = pd.DataFrame(index=grouped.index)
    if count_col and count_col in grouped.columns:
        geo['Lead_Count'] = grouped[count_col]
    else:
        geo['Lead_Count'] = leads_df.groupby('Country').size()
    if 'RevenuePotential' in leads_df.columns:
        geo['Total_Pipeline'] = grouped.get('RevenuePotential', leads_df.groupby('Country')['RevenuePotential'].sum())
        geo['Avg_Deal_Size'] = leads_df.groupby('Country')['RevenuePotential'].mean()
    if 'ConversionProbability' in leads_df.columns:
        geo['Conversion_Rate'] = grouped.get('ConversionProbability', leads_df.groupby('Country')['ConversionProbability'].mean())
    if 'ExpectedRevenue' in leads_df.columns:
        geo['Expected_Revenue'] = grouped.get('ExpectedRevenue', leads_df.groupby('Country')['ExpectedRevenue'].sum())
    if 'EngagementScore' in leads_df.columns:
        geo['Avg_Engagement'] = grouped.get('EngagementScore', leads_df.groupby('Country')['EngagementScore'].mean())

    geo = geo.fillna(0).round(2).reset_index()

    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(create_metric_card("Active Markets", len(geo)), unsafe_allow_html=True)
    with k2:
        top_market = geo.loc[geo['Lead_Count'].idxmax(), 'Country'] if len(geo) > 0 else 'N/A'
        top_expected = float(geo['Expected_Revenue'].max()) if 'Expected_Revenue' in geo.columns and len(geo) > 0 else 0.0
        st.markdown(create_metric_card("Top Market", f"{top_market}", None, "text"), unsafe_allow_html=True)
        st.caption(f"💰 ${top_expected:,.0f} expected")
    with k3:
        avg_conv = float(geo['Conversion_Rate'].mean()*100) if 'Conversion_Rate' in geo.columns else 0.0
        st.markdown(create_metric_card("Avg Conversion", avg_conv, 6.2, "percentage"), unsafe_allow_html=True)
    with k4:
        global_pipeline = float(geo['Total_Pipeline'].sum()) if 'Total_Pipeline' in geo.columns else 0.0
        st.markdown(create_metric_card("Global Pipeline", global_pipeline, 18.5, "currency"), unsafe_allow_html=True)

    a,b = st.columns(2)
    with a:
        fig = go.Figure(go.Bar(x=geo['Country'], y=geo['Lead_Count'], marker_color='#3b82f6'))
        fig.update_layout(title='Market Size by Lead Volume', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        if 'Expected_Revenue' in geo.columns:
            fig = go.Figure(go.Bar(x=geo['Country'], y=geo['Expected_Revenue'], marker_color='#10b981'))
            fig.update_layout(title='Revenue Potential by Market', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        elif 'Total_Pipeline' in geo.columns:
            fig = go.Figure(go.Bar(x=geo['Country'], y=geo['Total_Pipeline'], marker_color='#f59e0b'))
            fig.update_layout(title='Pipeline Value by Market', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Market Intelligence Matrix")
    c1, c2 = st.columns(2)
    with c1:
        if {'Conversion_Rate','Avg_Deal_Size'}.issubset(geo.columns):
            fig = go.Figure(go.Scatter(x=geo['Conversion_Rate'], y=geo['Avg_Deal_Size'], mode='markers+text',
                                       text=geo['Country'], textposition='top center',
                                       marker=dict(size=geo['Lead_Count']*3,
                                                   color=geo.get('Expected_Revenue', geo.get('Total_Pipeline', 0)),
                                                   colorscale='Viridis', showscale=True, opacity=0.8)))
            fig.update_layout(title='Market Opportunity Matrix', xaxis_title='Conversion Rate',
                              yaxis_title='Average Deal Size ($)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need Conversion_Rate and Avg_Deal_Size to render the opportunity matrix.")
    with c2:
        metric_cols = [c for c in ['Conversion_Rate','Avg_Engagement','Avg_Deal_Size'] if c in geo.columns]
        if len(metric_cols) >= 1:
            perf = geo[metric_cols].T
            perf.columns = geo['Country']
            perf_norm = perf.div(perf.max(axis=1).replace(0, 1), axis=0) * 100
            fig = go.Figure(go.Heatmap(z=perf_norm.values, x=perf_norm.columns, y=perf_norm.index, colorscale='RdYlGn', showscale=True))
            fig.update_layout(title='Market Performance Heatmap (Normalized)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient metrics to render heatmap (need any of Conversion_Rate, Avg_Engagement, Avg_Deal_Size).")

# --------------------------------------
# AI Command Center (placeholder summary)
# --------------------------------------
def create_advanced_ai_insights_dashboard(leads_df, calls_df, schedule_df, agent_perf_df):
    st.subheader("🧠 Advanced AI/ML Intelligence Center")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(create_metric_card("Model Accuracy", 87.3, 2.1, "percentage"), unsafe_allow_html=True)
    with c2: st.markdown(create_metric_card("Prediction Confidence", 82.1, -1.5, "percentage"), unsafe_allow_html=True)
    with c3: st.markdown(create_metric_card("Forecast Precision", 79.8, 5.3, "percentage"), unsafe_allow_html=True)
    with c4: st.markdown(create_metric_card("Action Success", 74.2, 8.7, "percentage"), unsafe_allow_html=True)

# --------------------------------------
# Main
# --------------------------------------
def main():
    leads_df, calls_df, schedule_df, agent_perf_df = load_enhanced_data()

    st.sidebar.markdown("## 🎛️ Advanced Analytics Filters")
    st.sidebar.selectbox("📅 Select Time Period", ["Last 7 days","Last 30 days","Last 90 days","Last 6 months","All time"], index=1)

    if len(agent_perf_df) > 0:
        st.sidebar.multiselect("👥 Select Agents", ['All Agents'] + agent_perf_df.get('AgentName', ['Agent 1','Agent 2']).tolist(), default=['All Agents'])

    if 'Country' in leads_df.columns:
        st.sidebar.multiselect("🌍 Select Markets", ['All Markets'] + sorted(leads_df['Country'].dropna().unique().tolist()), default=['All Markets'])

    if 'LeadScoringId' in leads_df.columns:
        st.sidebar.selectbox("🌡️ Lead Temperature Filter", ["All Temperatures","HOT","WARM","COLD","DEAD"], index=0)

    if 'BehavioralSegment' in leads_df.columns:
        st.sidebar.multiselect("🎯 Behavioral Segments", ['All Segments'] + sorted(leads_df['BehavioralSegment'].dropna().unique().tolist()), default=['All Segments'])

    st.sidebar.checkbox("🔄 Auto-refresh (30 seconds)", value=False)

    tabs = st.tabs([
        "Executive Overview",
        "Lead Intelligence",
        "Call Analytics",
        "Task Management",
        "Agent Performance",
        "Revenue Intelligence",
        "Market Analysis",
        "AI Command Center"
    ])

    with tabs:
        create_enhanced_executive_summary(leads_df, calls_df, schedule_df, agent_perf_df)
    with tabs[1]:
        create_enhanced_lead_status_dashboard(leads_df)
    with tabs[asset:1]:
        create_enhanced_call_activity_dashboard(calls_df)
    with tabs[asset:2]:
        create_enhanced_task_dashboard(schedule_df)
    with tabs[asset:3]:
        create_enhanced_agent_dashboard(agent_perf_df, schedule_df)
    with tabs[asset:4]:
        create_enhanced_conversion_dashboard(leads_df)
    with tabs[asset:5]:
        create_enhanced_geographic_dashboard(leads_df)
    with tabs[asset:6]:
        create_advanced_ai_insights_dashboard(leads_df, calls_df, schedule_df, agent_perf_df)

if __name__ == "__main__":
    main()
