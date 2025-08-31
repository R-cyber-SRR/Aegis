#!/usr/bin/env python3
"""
Flask Web Application for Fraud Detection
Alternative to Streamlit for users who prefer traditional web frameworks
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime

# Import fraud detection modules
from src.fraud_detector.config import AppConfig
from src.fraud_detector.data_ingestion import DataIngestion
from src.fraud_detector.preprocessing import preprocess
from src.fraud_detector.profiling import create_transaction_features, select_feature_matrix
from src.fraud_detector.model import AnomalyModel
from src.fraud_detector.reporting import generate_reasons, write_flags_csv

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to store session data
session_data = {}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Read CSV data
        df = pd.read_csv(file)
        
        # Store in session
        session_id = request.form.get('session_id', 'default')
        session_data[session_id] = {
            'data': df.to_dict('records'),
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'upload_time': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully. Shape: {df.shape}',
            'columns': df.columns.tolist(),
            'preview': df.head(5).to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/detect', methods=['POST'])
def detect_fraud():
    """Run fraud detection"""
    try:
        session_id = request.form.get('session_id', 'default')
        # Use fixed threshold from config - ignore user input for security
        model_dir = request.form.get('model_dir', 'models/')
        
        if session_id not in session_data:
            return jsonify({'error': 'No data uploaded. Please upload a file first.'}), 400
        
        # Get data from session
        data = session_data[session_id]['data']
        df = pd.DataFrame(data)
        
        # Load configuration
        config_path = Path("config.yaml")
        if not config_path.exists():
            return jsonify({'error': 'Configuration file not found. Please ensure config.yaml exists.'}), 400
        
        cfg = AppConfig.from_yaml(config_path)
        # Use fixed threshold from configuration (immutable)
        threshold = cfg.model.threshold
        
        # Load model
        model_path = Path(model_dir)
        if not model_path.exists():
            return jsonify({'error': f'Model directory not found: {model_dir}. Please run create_sample_model.py first to create a sample model.'}), 400
        
        try:
            model = AnomalyModel.load(model_path)
        except FileNotFoundError:
            return jsonify({'error': f'No trained model found in {model_dir}. Please run create_sample_model.py first to create a sample model.'}), 400
        
        # Preprocess and analyze
        df_processed = preprocess(df, cfg.features)
        feat = create_transaction_features(df_processed, cfg.features)
        X = select_feature_matrix(feat, cfg.features)
        
        # Score data
        scores = model.score(X)
        feat["suspicion_score"] = scores
        
        # Flag suspicious transactions
        flags = feat[feat["suspicion_score"] >= threshold].copy()
        flags["reasons"] = flags.apply(generate_reasons, axis=1)
        
        # Store results
        session_data[session_id]['results'] = {
            'scores': scores.tolist(),
            'flags': flags.to_dict('records'),
            'features': feat.to_dict('records'),
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'message': f'Fraud detection completed. Found {len(flags)} suspicious transactions.',
            'total_transactions': len(feat),
            'suspicious_count': len(flags),
            'suspicious_rate': len(flags) / len(feat) * 100
        })
        
    except Exception as e:
        return jsonify({'error': f'Error during fraud detection: {str(e)}'}), 500

@app.route('/results')
def get_results():
    """Get fraud detection results"""
    session_id = request.args.get('session_id', 'default')
    
    if session_id not in session_data or 'results' not in session_data[session_id]:
        return jsonify({'error': 'No results available. Please run fraud detection first.'}), 400
    
    return jsonify(session_data[session_id]['results'])

@app.route('/download')
def download_results():
    """Download results as CSV"""
    session_id = request.args.get('session_id', 'default')
    
    if session_id not in session_data or 'results' not in session_data[session_id]:
        return jsonify({'error': 'No results available'}), 400
    
    results = session_data[session_id]['results']
    flags_df = pd.DataFrame(results['flags'])
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        flags_df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f"fraud_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mimetype='text/csv'
        )
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

if __name__ == '__main__':
    print("🚀 Starting Fraud Detection Web Application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the application")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
