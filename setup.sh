#!/bin/bash

# Executive CRM Dashboard Setup Script
echo "🏢 Setting up Executive CRM Dashboard..."

# Create project directory structure
echo "📁 Creating directory structure..."
mkdir -p .streamlit
mkdir -p data
mkdir -p docs
mkdir -p assets

# Copy Streamlit config
echo "⚙️ Setting up Streamlit configuration..."
cp streamlit_config.toml .streamlit/config.toml

# Create virtual environment
echo "🐍 Creating virtual environment..."
python -m venv crm_dashboard_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    source crm_dashboard_env/Scripts/activate
else
    source crm_dashboard_env/bin/activate
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Move data files to data directory
echo "📊 Setting up sample data..."
if [ -f "enhanced_leads.csv" ]; then
    mv enhanced_leads.csv data/
fi
if [ -f "enhanced_calls.csv" ]; then
    mv enhanced_calls.csv data/
fi
if [ -f "enhanced_schedule.csv" ]; then
    mv enhanced_schedule.csv data/
fi
if [ -f "agent_availability.csv" ]; then
    mv agent_availability.csv data/
fi

echo "✅ Setup complete!"
echo ""
echo "🚀 To run the dashboard:"
echo "   1. Activate virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo "      source crm_dashboard_env/Scripts/activate"
else
    echo "      source crm_dashboard_env/bin/activate"
fi
echo "   2. Run the application:"
echo "      streamlit run app.py"
echo ""
echo "📖 For detailed instructions, see README.md"