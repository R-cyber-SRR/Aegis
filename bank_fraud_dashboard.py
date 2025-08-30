#!/usr/bin/env python3
"""
Bank Fraud Detection Dashboard
A Streamlit-based web interface for the bank fraud detection system.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

# Page configuration
st.set_page_config(page_title="Bank Fraud Detection", page_icon="🏦", layout="wide")

def load_model():
    """Load the trained model"""
    try:
        model_dir = Path("models")
        model = joblib.load(model_dir / "bank_fraud_model.joblib")
        scaler = joblib.load(model_dir / "bank_fraud_scaler.joblib")
        with open(model_dir / "bank_fraud_features.json", 'r') as f:
            feature_info = json.load(f)
        return model, scaler, feature_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def preprocess_data(df):
    """Preprocess the data"""
    df_processed = df.copy()
    
    # Basic cleaning
    df_processed['TransactionDate'] = pd.to_datetime(df_processed['TransactionDate'], errors='coerce')
    df_processed['TransactionAmount'] = pd.to_numeric(df_processed['TransactionAmount'], errors='coerce')
    df_processed = df_processed.dropna(subset=['TransactionDate', 'TransactionAmount', 'AccountID'])
    
    # Create features
    df_processed['hour'] = df_processed['TransactionDate'].dt.hour
    df_processed['amount_log'] = np.log1p(df_processed['TransactionAmount'].abs())
    
    # Account stats
    account_stats = df_processed.groupby('AccountID').agg({
        'TransactionAmount': ['mean', 'std', 'count']
    }).reset_index()
    account_stats.columns = ['AccountID', 'avg_amount', 'std_amount', 'tx_count']
    df_processed = df_processed.merge(account_stats, on='AccountID', how='left')
    
    # Risk features
    df_processed['amount_to_avg_ratio'] = df_processed['TransactionAmount'] / df_processed['avg_amount'].replace(0, 1)
    df_processed['unusual_time_flag'] = ((df_processed['hour'] < 6) | (df_processed['hour'] > 22)).astype(int)
    
    return df_processed

def main():
    st.title("🏦 Bank Fraud Detection System")
    
    # Load model
    model, scaler, feature_info = load_model()
    
    if model is None:
        st.error("Model not found! Please run 'train_bank_model.py' first.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} transactions")
        
        # Preprocess
        df_processed = preprocess_data(df)
        
        # Make predictions
        if st.button("Run Fraud Detection"):
            with st.spinner("Analyzing..."):
                # Select features
                X = df_processed[['amount_log', 'amount_to_avg_ratio', 'unusual_time_flag']].fillna(0)
                X_scaled = scaler.transform(X)
                
                # Predict
                scores = model.score_samples(X_scaled)
                suspicion_scores = -scores
                suspicion_scores = (suspicion_scores - suspicion_scores.min()) / (suspicion_scores.max() - suspicion_scores.min())
                
                # Add results
                df_processed['suspicion_score'] = suspicion_scores
                df_processed['fraud_flag'] = (suspicion_scores >= 0.7)
                
                # Show results
                st.subheader("Results")
                flagged = df_processed[df_processed['fraud_flag'] == True]
                st.write(f"Flagged transactions: {len(flagged)}")
                
                if len(flagged) > 0:
                    st.dataframe(flagged[['TransactionID', 'AccountID', 'TransactionAmount', 'suspicion_score']])
                
                # Download results
                csv = df_processed.to_csv(index=False)
                st.download_button("Download Results", csv, "fraud_results.csv")

if __name__ == "__main__":
    main()
