# 🕵️ AI-Powered Fraud Detection System

This project builds an advanced anomaly detection system using historical transaction behaviour profiling to flag suspicious transactions. It now includes both a modern web-based UI and traditional CLI tools.

### Quickstart

1. **Install Python 3.10+**
2. **Install dependencies (choose one option):**

**Option A: Automatic Installation (Recommended)**
```bash
# On Windows (double-click this file):
install_requirements.bat

# On Linux/Mac:
python install_requirements.py
```

**Option B: Manual Installation**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

3. Train model on CSV data:

```bash
python -m src.fraud_detector.cli train --data data/sample.csv --model_dir models
```

4. Score and generate flags CSV:

```bash
python -m src.fraud_detector.cli score --data data/sample.csv --model_dir models --out reports/flags.csv
```

5. Launch dashboard (choose one option):

**Option A: Streamlit Dashboard (Recommended)**
```bash
python run_dashboard.py
```

**Option B: Flask Web App**
```bash
python app.py
```

**Option C: CLI Dashboard (Legacy)**
```bash
python -m src.fraud_detector.cli dashboard
```

### Web UI Features

The system now provides two modern web interfaces:

#### 🚀 Streamlit Dashboard (Recommended)
- **Drag & Drop**: Easy CSV file upload
- **Real-time Analysis**: Instant fraud detection results
- **Interactive Charts**: Visual score distributions and metrics
- **Configurable Thresholds**: Adjustable detection sensitivity
- **Results Export**: Download suspicious transactions as CSV

#### 🌐 Flask Web Application
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Bootstrap-based interface with gradients
- **Session Management**: Multiple user sessions supported
- **RESTful API**: Backend endpoints for integration

### Config

See `config.yaml` for thresholds and feature mapping.

### Features
- **Modern Web UI**: Streamlit-based dashboard with drag-and-drop file upload
- **Alternative Web App**: Flask-based web application with responsive design
- **Data ingestion**: CSV/JSON/SQL support (CSV to start)
- **Preprocessing**: Advanced data cleaning with anonymization
- **Behaviour profiling**: Sophisticated feature engineering
- **AI Detection**: IsolationForest anomaly detection with configurable thresholds
- **Real-time Analysis**: Interactive results dashboard with visualizations
- **Export Options**: CSV download of suspicious transactions
- **CLI Tools**: Command-line interface for batch processing
- **Model Management**: Training, saving, and loading of detection models

### License
MIT