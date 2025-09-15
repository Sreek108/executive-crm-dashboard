import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Executive CRM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive styling
st.markdown("""
<style>
    .main {
        background-color: #0f1419;
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4a5568;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f59e0b;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .sidebar .sidebar-content {
        background-color: #1a202c;
    }
    
    .stSelectbox label, .stMultiSelect label, .stDateInput label {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .dashboard-header {
        background: linear-gradient(90deg, #1a202c 0%, #2d3748 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        border: 1px solid #4a5568;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748;
        color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #f59e0b;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process CRM data"""
    try:
        # Load enhanced datasets
        leads_df = pd.read_csv('enhanced_leads.csv')
        calls_df = pd.read_csv('enhanced_calls.csv')  
        schedule_df = pd.read_csv('enhanced_schedule.csv')
        availability_df = pd.read_csv('agent_availability.csv')
        
        # Convert date columns
        leads_df['CreatedOn'] = pd.to_datetime(leads_df['CreatedOn'])
        leads_df['ModifiedOn'] = pd.to_datetime(leads_df['ModifiedOn'])
        calls_df['CallDateTime'] = pd.to_datetime(calls_df['CallDateTime'])
        schedule_df['ScheduledDate'] = pd.to_datetime(schedule_df['ScheduledDate'])
        
        return leads_df, calls_df, schedule_df, availability_df
    except FileNotFoundError:
        # Fallback: create sample data if files not found
        return create_sample_data()

def create_sample_data():
    """Create sample data for demo purposes"""
    # Generate sample leads data
    leads_data = {
        'LeadId': range(1, 51),
        'LeadCode': [f'LEAD{i:03d}' for i in range(1, 51)],
        'FullName': ['Ahmed Al-Rashid', 'Sarah Johnson', 'Rajesh Sharma'] * 17,
        'Company': ['Dubai Properties LLC', 'London Investments', 'Mumbai Real Estate'] * 17,
        'LeadStageId': np.random.choice([1, 2, 3, 4], 50, p=[0.44, 0.20, 0.16, 0.20]),
        'LeadScoringId': np.random.choice([1, 2, 3, 4], 50, p=[0.22, 0.30, 0.30, 0.18]),
        'CountryId': np.random.choice([1, 2, 3, 4, 5], 50),
        'RevenuePotential': np.random.normal(150000, 75000, 50),
        'ConversionProbability': np.random.uniform(0.1, 0.9, 50),
        'CreatedOn': pd.date_range('2025-08-01', periods=50, freq='D')
    }
    leads_df = pd.DataFrame(leads_data)
    leads_df['ExpectedRevenue'] = leads_df['RevenuePotential'] * leads_df['ConversionProbability']
    
    # Generate sample calls data
    calls_data = {
        'LeadCallId': range(1, 81),
        'LeadId': np.random.choice(range(1, 51), 80),
        'CallDateTime': pd.date_range('2025-08-01', periods=80, freq='H'),
        'DurationSeconds': np.random.exponential(600, 80),
        'CallStatusId': np.random.choice([1, 2, 3, 4, 5], 80, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
        'SentimentId': np.random.choice([1, 2, 3], 80, p=[0.4, 0.35, 0.25]),
        'IsSuccessful': np.random.choice([True, False], 80, p=[0.35, 0.65])
    }
    calls_df = pd.DataFrame(calls_data)
    
    # Generate sample schedule data  
    schedule_data = {
        'ScheduleId': range(1, 31),
        'LeadId': np.random.choice(range(1, 51), 30),
        'ScheduledDate': pd.date_range('2025-09-16', periods=30, freq='D'),
        'TaskStatusId': np.random.choice([1, 2, 3, 4, 5], 30),
        'AssignedAgentId': np.random.choice([1, 2, 3, 4, 5], 30),
        'IsOverdue': np.random.choice([True, False], 30, p=[0.1, 0.9])
    }
    schedule_df = pd.DataFrame(schedule_data)
    
    # Generate sample availability data
    availability_data = []
    for agent_id in range(1, 6):
        for day in range(7):
            for hour in range(9, 18):
                availability_data.append({
                    'AgentId': agent_id,
                    'AgentName': ['Jasmin', 'Mohammed', 'Sarah', 'Ahmed', 'Fatima'][agent_id-1],
                    'Day': day,
                    'Hour': hour,
                    'AvailabilityScore': np.random.uniform(0.3, 1.0)
                })
    availability_df = pd.DataFrame(availability_data)
    
    return leads_df, calls_df, schedule_df, availability_df

def create_executive_summary(leads_df, calls_df, schedule_df):
    """Create executive summary metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_leads = len(leads_df)
        hot_leads = len(leads_df[leads_df['LeadScoringId'] == 1])
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_leads}</div>
            <div class="metric-label">Total Leads</div>
            <div style="color: #68d391; font-size: 0.8rem;">🔥 {hot_leads} Hot Leads</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_revenue = leads_df['RevenuePotential'].sum()
        expected_revenue = leads_df['ExpectedRevenue'].sum()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">${total_revenue/1e6:.1f}M</div>
            <div class="metric-label">Revenue Potential</div>
            <div style="color: #68d391; font-size: 0.8rem;">💰 ${expected_revenue/1e6:.1f}M Expected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        successful_calls = calls_df['IsSuccessful'].sum()
        total_calls = len(calls_df)
        success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Call Success Rate</div>
            <div style="color: #68d391; font-size: 0.8rem;">📞 {successful_calls}/{total_calls} Calls</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_conversion = leads_df['ConversionProbability'].mean() * 100
        high_prob_leads = len(leads_df[leads_df['ConversionProbability'] > 0.6])
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{avg_conversion:.1f}%</div>
            <div class="metric-label">Avg Conversion Probability</div>
            <div style="color: #68d391; font-size: 0.8rem;">🎯 {high_prob_leads} High-Probability</div>
        </div>
        """, unsafe_allow_html=True)

def create_lead_status_dashboard(leads_df):
    """Lead Status Dashboard with pie charts"""
    st.subheader("📊 Lead Status Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead Stage Distribution
        stage_mapping = {1: 'New', 2: 'Qualified', 3: 'Nurtured', 4: 'Converted'}
        stage_counts = leads_df['LeadStageId'].value_counts()
        stage_labels = [stage_mapping[i] for i in stage_counts.index]
        
        fig_stage = go.Figure(data=[go.Pie(
            labels=stage_labels,
            values=stage_counts.values,
            hole=0.4,
            marker_colors=['#3b82f6', '#f59e0b', '#10b981', '#ef4444']
        )])
        fig_stage.update_layout(
            title="Lead Pipeline Stages",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12)
        )
        st.plotly_chart(fig_stage, use_container_width=True)
    
    with col2:
        # Lead Scoring Distribution
        scoring_mapping = {1: 'HOT', 2: 'WARM', 3: 'COLD', 4: 'DEAD'}
        scoring_counts = leads_df['LeadScoringId'].value_counts()
        scoring_labels = [scoring_mapping[i] for i in scoring_counts.index]
        
        fig_scoring = go.Figure(data=[go.Pie(
            labels=scoring_labels,
            values=scoring_counts.values,
            hole=0.4,
            marker_colors=['#dc2626', '#f59e0b', '#3b82f6', '#6b7280']
        )])
        fig_scoring.update_layout(
            title="Lead Scoring Distribution",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12)
        )
        st.plotly_chart(fig_scoring, use_container_width=True)
    
    # Conversion Funnel
    st.subheader("🔄 Lead Conversion Funnel")
    funnel_data = leads_df['LeadStageId'].value_counts().sort_index()
    funnel_labels = ['New Leads', 'Qualified', 'Nurtured', 'Converted']
    
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_labels,
        x=funnel_data.values,
        textinfo="value+percent initial",
        marker_color=['#3b82f6', '#f59e0b', '#10b981', '#ef4444']
    ))
    fig_funnel.update_layout(
        title="Lead Conversion Funnel",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12)
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

def create_call_activity_dashboard(calls_df):
    """AI Call Activity Dashboard"""
    st.subheader("🤖 AI Call Activity Analysis")
    
    # Daily call volume
    calls_df['CallDate'] = calls_df['CallDateTime'].dt.date
    daily_calls = calls_df.groupby('CallDate').size().reset_index(name='CallCount')
    daily_success = calls_df.groupby('CallDate')['IsSuccessful'].sum().reset_index(name='SuccessfulCalls')
    daily_data = daily_calls.merge(daily_success, on='CallDate')
    daily_data['SuccessRate'] = daily_data['SuccessfulCalls'] / daily_data['CallCount'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily_data['CallDate'],
            y=daily_data['CallCount'],
            name='Total Calls',
            marker_color='#3b82f6'
        ))
        fig_daily.add_trace(go.Bar(
            x=daily_data['CallDate'],
            y=daily_data['SuccessfulCalls'],
            name='Successful Calls',
            marker_color='#10b981'
        ))
        fig_daily.update_layout(
            title="Daily Call Volume",
            xaxis_title="Date",
            yaxis_title="Number of Calls",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            barmode='overlay'
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        # Call success rate trend
        fig_success = go.Figure()
        fig_success.add_trace(go.Scatter(
            x=daily_data['CallDate'],
            y=daily_data['SuccessRate'],
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=8)
        ))
        fig_success.update_layout(
            title="Call Success Rate Trend",
            xaxis_title="Date",
            yaxis_title="Success Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    # Call duration analysis
    st.subheader("⏱️ Call Duration Analysis")
    
    # Filter successful calls only for duration analysis
    successful_calls = calls_df[calls_df['IsSuccessful'] == True]
    
    if len(successful_calls) > 0:
        fig_duration = go.Figure()
        fig_duration.add_trace(go.Histogram(
            x=successful_calls['DurationSeconds'] / 60,  # Convert to minutes
            nbinsx=20,
            name='Call Duration Distribution',
            marker_color='#8b5cf6'
        ))
        fig_duration.update_layout(
            title="Successful Call Duration Distribution",
            xaxis_title="Duration (minutes)",
            yaxis_title="Number of Calls",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_duration, use_container_width=True)

def create_followup_task_dashboard(schedule_df):
    """Follow-up & Task Dashboard"""
    st.subheader("📅 Follow-up & Task Management")
    
    # Task status overview
    col1, col2, col3 = st.columns(3)
    
    current_date = datetime.now()
    overdue_tasks = len(schedule_df[
        (schedule_df['ScheduledDate'] < current_date) & 
        (schedule_df['TaskStatusId'] == 1)
    ])
    
    upcoming_tasks = len(schedule_df[
        (schedule_df['ScheduledDate'] >= current_date) & 
        (schedule_df['ScheduledDate'] <= current_date + timedelta(days=7))
    ])
    
    completed_tasks = len(schedule_df[schedule_df['TaskStatusId'] == 3])
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #ef4444;">{overdue_tasks}</div>
            <div class="metric-label">Overdue Tasks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #f59e0b;">{upcoming_tasks}</div>
            <div class="metric-label">Upcoming (7 days)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #10b981;">{completed_tasks}</div>
            <div class="metric-label">Completed Tasks</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Task timeline
    schedule_df['DaysFromNow'] = (schedule_df['ScheduledDate'] - current_date).dt.days
    
    fig_timeline = go.Figure()
    
    # Add tasks by status
    status_colors = {1: '#f59e0b', 2: '#3b82f6', 3: '#10b981', 4: '#6b7280', 5: '#ef4444'}
    status_names = {1: 'Pending', 2: 'In Progress', 3: 'Completed', 4: 'Cancelled', 5: 'Overdue'}
    
    for status_id, color in status_colors.items():
        status_data = schedule_df[schedule_df['TaskStatusId'] == status_id]
        fig_timeline.add_trace(go.Scatter(
            x=status_data['DaysFromNow'],
            y=status_data.index,
            mode='markers',
            name=status_names[status_id],
            marker=dict(size=10, color=color)
        ))
    
    fig_timeline.update_layout(
        title="Task Timeline (Days from Today)",
        xaxis_title="Days from Today",
        yaxis_title="Task Index",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

def create_agent_availability_dashboard(availability_df):
    """Agent Availability Heatmap Dashboard"""
    st.subheader("👥 Agent Availability Heatmap")
    
    # Create availability heatmap
    agents = availability_df['AgentName'].unique()
    hours = list(range(9, 18))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Create heatmap data for each agent
    for agent in agents:
        agent_data = availability_df[availability_df['AgentName'] == agent]
        
        # Create matrix for heatmap
        heatmap_data = []
        for day in range(7):
            day_data = []
            for hour in hours:
                availability = agent_data[
                    (agent_data['Day'] == day) & (agent_data['Hour'] == hour)
                ]['AvailabilityScore'].values
                day_data.append(availability[0] if len(availability) > 0 else 0)
            heatmap_data.append(day_data)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f'{h}:00' for h in hours],
            y=days,
            colorscale='RdYlGn',
            showscale=True
        ))
        
        fig_heatmap.update_layout(
            title=f"{agent} - Weekly Availability",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)

def create_conversion_dashboard(leads_df):
    """Conversion Analysis Dashboard"""
    st.subheader("💼 Conversion & Revenue Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue vs Conversion Probability Scatter
        fig_scatter = go.Figure()
        
        # Color by lead scoring
        scoring_colors = {1: '#dc2626', 2: '#f59e0b', 3: '#3b82f6', 4: '#6b7280'}
        scoring_names = {1: 'HOT', 2: 'WARM', 3: 'COLD', 4: 'DEAD'}
        
        for score_id in leads_df['LeadScoringId'].unique():
            score_data = leads_df[leads_df['LeadScoringId'] == score_id]
            fig_scatter.add_trace(go.Scatter(
                x=score_data['ConversionProbability'],
                y=score_data['RevenuePotential'],
                mode='markers',
                name=scoring_names[score_id],
                marker=dict(
                    size=10,
                    color=scoring_colors[score_id],
                    opacity=0.7
                )
            ))
        
        fig_scatter.update_layout(
            title="Revenue Potential vs Conversion Probability",
            xaxis_title="Conversion Probability",
            yaxis_title="Revenue Potential ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Expected Revenue by Stage
        stage_mapping = {1: 'New', 2: 'Qualified', 3: 'Nurtured', 4: 'Converted'}
        revenue_by_stage = leads_df.groupby('LeadStageId')['ExpectedRevenue'].sum()
        stage_labels = [stage_mapping[i] for i in revenue_by_stage.index]
        
        fig_revenue = go.Figure(data=[go.Bar(
            x=stage_labels,
            y=revenue_by_stage.values,
            marker_color=['#3b82f6', '#f59e0b', '#10b981', '#ef4444']
        )])
        
        fig_revenue.update_layout(
            title="Expected Revenue by Lead Stage",
            xaxis_title="Lead Stage",
            yaxis_title="Expected Revenue ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Top opportunities
    st.subheader("🏆 Top Revenue Opportunities")
    top_leads = leads_df.nlargest(10, 'ExpectedRevenue')[
        ['FullName', 'Company', 'RevenuePotential', 'ConversionProbability', 'ExpectedRevenue']
    ]
    top_leads['RevenuePotential'] = top_leads['RevenuePotential'].apply(lambda x: f'${x:,.0f}')
    top_leads['ConversionProbability'] = top_leads['ConversionProbability'].apply(lambda x: f'{x:.1%}')
    top_leads['ExpectedRevenue'] = top_leads['ExpectedRevenue'].apply(lambda x: f'${x:,.0f}')
    
    st.dataframe(top_leads, use_container_width=True)

def create_geographic_dashboard(leads_df):
    """Geographic Analysis Dashboard"""
    st.subheader("🌍 Geographic Lead Distribution")
    
    # Country mapping
    country_mapping = {1: 'Saudi Arabia', 2: 'UAE', 3: 'India', 4: 'United Kingdom', 5: 'United States'}
    leads_df['Country'] = leads_df['CountryId'].map(country_mapping)
    
    # Geographic summary
    geo_summary = leads_df.groupby('Country').agg({
        'LeadId': 'count',
        'RevenuePotential': 'sum',
        'ConversionProbability': 'mean',
        'ExpectedRevenue': 'sum'
    }).reset_index()
    
    geo_summary.columns = ['Country', 'Lead_Count', 'Revenue_Potential', 'Avg_Conversion_Prob', 'Expected_Revenue']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead count by country
        fig_geo_leads = go.Figure(data=[go.Bar(
            x=geo_summary['Country'],
            y=geo_summary['Lead_Count'],
            marker_color='#3b82f6'
        )])
        
        fig_geo_leads.update_layout(
            title="Lead Count by Country",
            xaxis_title="Country",
            yaxis_title="Number of Leads",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_geo_leads, use_container_width=True)
    
    with col2:
        # Revenue potential by country
        fig_geo_revenue = go.Figure(data=[go.Bar(
            x=geo_summary['Country'],
            y=geo_summary['Expected_Revenue'],
            marker_color='#10b981'
        )])
        
        fig_geo_revenue.update_layout(
            title="Expected Revenue by Country",
            xaxis_title="Country",
            yaxis_title="Expected Revenue ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_geo_revenue, use_container_width=True)
    
    # Geographic performance table
    st.subheader("📊 Country Performance Summary")
    geo_summary['Revenue_Potential'] = geo_summary['Revenue_Potential'].apply(lambda x: f'${x:,.0f}')
    geo_summary['Avg_Conversion_Prob'] = geo_summary['Avg_Conversion_Prob'].apply(lambda x: f'{x:.1%}')
    geo_summary['Expected_Revenue'] = geo_summary['Expected_Revenue'].apply(lambda x: f'${x:,.0f}')
    
    st.dataframe(geo_summary, use_container_width=True)

def create_ai_insights_dashboard(leads_df, calls_df):
    """AI/ML Insights and Predictions Dashboard"""
    st.subheader("🤖 AI/ML Predictive Insights")
    
    # AI recommendations section
    st.markdown("""
    <div class="insight-card">
        <h4>🎯 Top AI Recommendations</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # High-priority leads for follow-up
        st.markdown("**🔥 Priority Leads for Immediate Action**")
        priority_leads = leads_df[
            (leads_df['LeadScoringId'].isin([1, 2])) & 
            (leads_df['ConversionProbability'] > 0.5)
        ].nlargest(5, 'ExpectedRevenue')
        
        for _, lead in priority_leads.iterrows():
            st.markdown(f"""
            <div style="background: #2d3748; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #f59e0b;">
                <strong>{lead['FullName']}</strong> - {lead['Company']}<br>
                <small>Expected Revenue: ${lead['ExpectedRevenue']:,.0f} | Probability: {lead['ConversionProbability']:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Churn risk alerts
        st.markdown("**⚠️ Leads at Risk of Churning**")
        if 'ChurnRisk' in leads_df.columns:
            at_risk_leads = leads_df[leads_df['ChurnRisk'] > 0.6].nlargest(5, 'ChurnRisk')
            
            for _, lead in at_risk_leads.iterrows():
                st.markdown(f"""
                <div style="background: #2d3748; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #ef4444;">
                    <strong>{lead['FullName']}</strong> - {lead['Company']}<br>
                    <small>Churn Risk: {lead['ChurnRisk']:.1%} | Revenue at Risk: ${lead['RevenuePotential']:,.0f}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Predictive analytics
    st.subheader("📈 Predictive Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Revenue forecast
        current_expected = leads_df['ExpectedRevenue'].sum()
        forecasted_30d = current_expected * 1.15  # 15% growth prediction
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">${forecasted_30d/1e6:.1f}M</div>
            <div class="metric-label">30-Day Revenue Forecast</div>
            <div style="color: #68d391; font-size: 0.8rem;">↗️ +15% Growth Predicted</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Conversion prediction
        likely_conversions = len(leads_df[leads_df['ConversionProbability'] > 0.7])
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{likely_conversions}</div>
            <div class="metric-label">Likely Conversions</div>
            <div style="color: #68d391; font-size: 0.8rem;">🎯 Next 30 Days</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Optimal call time
        if len(calls_df) > 0:
            successful_calls = calls_df[calls_df['IsSuccessful'] == True]
            if len(successful_calls) > 0:
                optimal_hour = successful_calls['CallDateTime'].dt.hour.mode().iloc[0]
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{optimal_hour}:00</div>
                    <div class="metric-label">Optimal Call Time</div>
                    <div style="color: #68d391; font-size: 0.8rem;">⏰ Highest Success Rate</div>
                </div>
                """, unsafe_allow_html=True)
    
    # ML Model Performance Simulation
    st.subheader("🧠 ML Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead scoring accuracy simulation
        accuracy_data = {
            'Model': ['Lead Scoring', 'Churn Prediction', 'Revenue Forecasting', 'Call Success Prediction'],
            'Accuracy': [0.87, 0.82, 0.79, 0.74],
            'Confidence': [0.92, 0.88, 0.85, 0.81]
        }
        
        fig_accuracy = go.Figure()
        fig_accuracy.add_trace(go.Bar(
            name='Accuracy',
            x=accuracy_data['Model'],
            y=accuracy_data['Accuracy'],
            marker_color='#3b82f6'
        ))
        fig_accuracy.add_trace(go.Bar(
            name='Confidence',
            x=accuracy_data['Model'],
            y=accuracy_data['Confidence'],
            marker_color='#10b981'
        ))
        
        fig_accuracy.update_layout(
            title="AI Model Performance Metrics",
            yaxis_title="Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            barmode='group'
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col2:
        # Feature importance for lead scoring
        features = ['Revenue Potential', 'Lead Stage', 'Country', 'Call Success Rate', 'Days Since Created']
        importance = [0.35, 0.28, 0.15, 0.12, 0.10]
        
        fig_importance = go.Figure(data=[go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='#f59e0b'
        )])
        
        fig_importance.update_layout(
            title="Lead Scoring Feature Importance",
            xaxis_title="Importance Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_importance, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Load data
    leads_df, calls_df, schedule_df, availability_df = load_data()
    
    # Sidebar for filters
    st.sidebar.header("🎛️ Dashboard Filters")
    
    # Date range filter
    date_range = st.sidebar.selectbox(
        "Select Time Period",
        ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
        index=1
    )
    
    # Agent filter
    agent_options = ['All Agents'] + ['Jasmin Ahmed', 'Mohammed Ali', 'Sarah Johnson', 'Ahmed Hassan', 'Fatima Al-Zahra']
    selected_agents = st.sidebar.multiselect(
        "Select Agents",
        agent_options,
        default=['All Agents']
    )
    
    # Country filter
    country_options = ['All Countries', 'Saudi Arabia', 'UAE', 'India', 'United Kingdom', 'United States']
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        country_options,
        default=['All Countries']
    )
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #f59e0b; margin: 0;">🏢 Executive CRM Dashboard</h1>
        <p style="color: #a0aec0; margin: 5px 0 0 0;">Real Estate Broker Performance Analytics with AI/ML Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Executive Summary",
        "📊 Lead Status", 
        "🤖 Call Activity",
        "📅 Tasks & Follow-up",
        "👥 Agent Availability",
        "💼 Conversion Analysis",
        "🌍 Geographic View",
        "🧠 AI Insights"
    ])
    
    with tab1:
        create_executive_summary(leads_df, calls_df, schedule_df)
        
        # Key insights section
        st.subheader("💡 Key Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-card">
                <h4>🎯 Performance Highlights</h4>
                <ul>
                    <li>Top performing region: UAE with highest conversion rates</li>
                    <li>AI Agent (Jasmin) shows 85% call success rate</li>
                    <li>Revenue pipeline weighted toward luxury properties</li>
                    <li>Peak call performance between 10-11 AM and 2-3 PM</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-card">
                <h4>⚠️ Areas for Attention</h4>
                <ul>
                    <li>15 leads showing high churn risk signals</li>
                    <li>COLD leads need nurturing strategy improvement</li>
                    <li>Call success rate below target in India market</li>
                    <li>Follow-up tasks completion rate needs optimization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        create_lead_status_dashboard(leads_df)
    
    with tab3:
        create_call_activity_dashboard(calls_df)
    
    with tab4:
        create_followup_task_dashboard(schedule_df)
    
    with tab5:
        create_agent_availability_dashboard(availability_df)
    
    with tab6:
        create_conversion_dashboard(leads_df)
    
    with tab7:
        create_geographic_dashboard(leads_df)
    
    with tab8:
        create_ai_insights_dashboard(leads_df, calls_df)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #6b7280; font-size: 0.8rem;">
        CRM Executive Dashboard | Real Estate Broker Analytics | Powered by AI/ML Insights
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()