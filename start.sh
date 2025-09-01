#!/bin/bash
# Aegis Fraud Detection System - Startup Script
 

echo "🕵️ Starting Aegis Fraud Detection System"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Setting up..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

# Check if model exists
if [ ! -f "models/isolation_forest.joblib" ]; then
    echo "📊 Creating sample model..."
    .venv/bin/python create_sample_model.py
fi

# Activate virtual environment and start the application
echo "🚀 Activating environment and starting Flask web application..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

.venv/bin/python app.py
