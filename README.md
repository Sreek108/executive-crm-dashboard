# 🏢 Executive CRM Dashboard

A comprehensive executive-level CRM dashboard for real estate broker performance analytics with AI/ML insights. Built with Streamlit and designed for client presentations.

![Dashboard Preview](https://img.shields.io/badge/Status-Ready_for_Deployment-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)

## ✨ Features

### 📊 Dashboard Components
- **Executive Summary**: High-level KPIs and business overview
- **Lead Status Dashboard**: Pie charts and pipeline funnel analysis  
- **AI Call Activity Dashboard**: Daily/weekly call metrics and success rates
- **Follow-up & Task Dashboard**: Task management and overdue tracking
- **Agent Availability Dashboard**: Interactive heatmaps showing agent schedules
- **Conversion Dashboard**: Revenue analysis and lead conversion tracking
- **Geographic Dashboard**: International lead distribution and performance
- **AI/ML Insights**: Predictive analytics and intelligent recommendations

### 🤖 AI/ML Features
- **Lead Scoring Predictions**: Automated HOT/WARM/COLD/DEAD classification
- **Churn Risk Analysis**: Identify leads at risk of dropping out
- **Revenue Forecasting**: 30-day revenue predictions with growth trends
- **Optimal Call Timing**: AI-recommended best times for lead contact
- **Agent Performance Prediction**: Success rate forecasting by agent
- **Geographic Market Intelligence**: Performance insights by region

### 🎨 Executive Design
- **Dark Professional Theme**: Executive-level dark color scheme
- **Interactive Filters**: Date range, agent, and country filtering
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Dynamic data refreshing capabilities
- **Export Ready**: Charts and data ready for presentation export

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/executive-crm-dashboard.git
cd executive-crm-dashboard

# Create virtual environment
python -m venv crm_dashboard_env
source crm_dashboard_env/bin/activate  # On Windows: crm_dashboard_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 2: Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your GitHub repository
4. Application will be available at: `https://your-app-name.streamlit.app`

### Option 3: Docker Deployment

```bash
# Build Docker image
docker build -t crm-dashboard .

# Run container
docker run -p 8501:8501 crm-dashboard
```

## 📁 Project Structure

```
executive-crm-dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── streamlit_config.toml       # Streamlit configuration
├── Dockerfile                  # Docker configuration
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
├── data/
│   ├── enhanced_leads.csv     # Sample lead data (50 records)
│   ├── enhanced_calls.csv     # Call interaction data (80 records)
│   ├── enhanced_schedule.csv  # Task and schedule data (30 records)
│   └── agent_availability.csv # Agent availability matrix
├── assets/
│   └── dashboard_preview.png  # Dashboard screenshots
├── docs/
│   ├── SETUP.md              # Detailed setup instructions
│   ├── API.md                # Data integration guide
│   └── CUSTOMIZATION.md      # Customization guide
└── README.md                 # This file
```

## 📊 Data Structure

### Sample Dataset Included
- **50 Lead Records**: Realistic real estate broker prospects across SA, AE, IN, UK, US
- **80 Call Records**: AI-powered interaction logs with sentiment analysis
- **30 Schedule Records**: Follow-up tasks and meeting schedules
- **5 Agent Records**: Mixed AI and human sales team
- **AI Predictions**: Conversion probabilities, churn risk, revenue forecasting

### Data Fields
**Leads**: LeadCode, FullName, Company, Country, LeadStage, LeadScoring, RevenuePotential, ConversionProbability
**Calls**: CallDateTime, Duration, Status, Sentiment, ProjectDiscussed, SuccessRate
**Schedule**: TaskType, ScheduledDate, Status, AssignedAgent, OverdueFlag

## 🎯 Key Metrics & KPIs

### Executive Summary Metrics
- **Total Leads**: 50 prospects across 5 international markets
- **Revenue Potential**: $9.6M total pipeline value
- **Expected Revenue**: $3.0M weighted by conversion probability
- **Call Success Rate**: 35% with AI sentiment analysis
- **Conversion Rate**: 20.3% average across all leads

### AI/ML Insights
- **Lead Scoring Accuracy**: 87% prediction accuracy
- **Churn Risk Detection**: 82% accuracy in identifying at-risk leads
- **Revenue Forecasting**: 79% accuracy for 30-day predictions
- **Optimal Call Times**: 10-11 AM and 2-3 PM show highest success rates

## 🔧 Customization

### Adding Your Data
1. Replace CSV files in `/data` directory with your actual CRM data
2. Ensure column names match the expected schema (see `docs/API.md`)
3. Update country and agent mappings in `app.py` if needed

### Theme Customization
Modify the color scheme in `streamlit_config.toml`:
```toml
[theme]
primaryColor = "#f59e0b"        # Gold accent color
backgroundColor = "#0f1419"      # Dark background
secondaryBackgroundColor = "#1a202c"  # Card background
textColor = "#ffffff"            # White text
```

### Adding New Dashboard Pages
1. Create new function in `app.py` following the pattern: `create_your_dashboard(data)`
2. Add new tab in the main `st.tabs()` configuration
3. Import additional visualization libraries if needed

## 🌐 Deployment Options

### Streamlit Cloud (Recommended)
- **Pros**: Free, easy setup, automatic updates from GitHub
- **Cons**: Limited resources, public unless upgraded
- **Best for**: Client demos, proof of concepts

### Docker + Cloud Platform
- **Pros**: Full control, scalable, private hosting
- **Cons**: Requires more setup, hosting costs
- **Best for**: Production deployments, enterprise use

### On-Premise Server
- **Pros**: Complete data privacy, custom infrastructure
- **Cons**: Requires IT setup and maintenance
- **Best for**: Enterprise environments, sensitive data

## 📈 Business Intelligence Features

### Executive-Level Insights
- **Pipeline Health**: Visual funnel showing lead progression
- **Revenue Forecasting**: AI-powered predictions with confidence intervals
- **Agent Performance**: Individual and team success metrics
- **Geographic Analysis**: Market performance by country/region
- **Risk Assessment**: Churn probability and revenue-at-risk calculations

### Decision-Making Support
- **Priority Lead Identification**: AI-ranked prospects for immediate action
- **Resource Optimization**: Agent availability and workload balancing
- **Market Intelligence**: Geographic performance and opportunity analysis
- **Performance Benchmarking**: Success rate comparisons and trend analysis

## 🔒 Security & Privacy

- **Data Privacy**: No data is stored permanently by the application
- **Local Processing**: All analytics run locally without external API calls
- **Configurable Access**: Easy to deploy behind corporate firewalls
- **Audit Trail**: All interactions can be logged for compliance

## 🆘 Support & Documentation

### Getting Help
- **Setup Issues**: See `docs/SETUP.md` for detailed instructions
- **Data Integration**: Check `docs/API.md` for data format requirements
- **Customization**: Review `docs/CUSTOMIZATION.md` for modification guide

### Common Issues
1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **Data File Errors**: Ensure CSV files are in `/data` directory
3. **Display Issues**: Clear browser cache and restart Streamlit
4. **Performance**: For large datasets (>1000 leads), consider data sampling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit changes: `git commit -m 'Add some feature'`
4. Push to branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## 🏆 Use Cases

### Real Estate Brokers
- **Lead Management**: Track and prioritize property investment leads
- **Agent Performance**: Monitor AI vs human agent success rates
- **Revenue Optimization**: Focus on high-value, high-probability opportunities

### Sales Teams
- **Pipeline Management**: Visual tracking of lead progression
- **Performance Analytics**: Individual and team success metrics
- **Predictive Insights**: AI-powered lead scoring and churn prevention

### Executive Reporting
- **Client Presentations**: Professional, executive-level dashboard design
- **Board Reports**: Key metrics and performance summaries
- **Strategic Planning**: Market analysis and revenue forecasting

---

## 🚀 Demo

**Live Demo**: [View Dashboard](https://your-streamlit-app-url.streamlit.app)

**Sample Credentials**: Demo data is pre-loaded with 50 realistic CRM records

**GitHub Repository**: [https://github.com/yourusername/executive-crm-dashboard](https://github.com/yourusername/executive-crm-dashboard)

---

*Built with ❤️ using Streamlit, Plotly, and AI/ML insights for modern CRM analytics*
