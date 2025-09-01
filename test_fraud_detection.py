#!/usr/bin/env python3
"""
Test script to verify fraud detection functionality
"""

import pandas as pd
from pathlib import Path
from src.fraud_detector.config import AppConfig
from src.fraud_detector.model import AnomalyModel
from src.fraud_detector.preprocessing import preprocess
from src.fraud_detector.profiling import create_transaction_features, select_feature_matrix

def test_fraud_detection():
    """Test the fraud detection system with sample data"""
    print("🧪 Testing Fraud Detection System...")
    
    # Load configuration
    config_path = Path("config.yaml")
    cfg = AppConfig.from_yaml(config_path)
    
    # Load sample data
    data_path = Path("sample_data.csv")
    if not data_path.exists():
        print(" Sample data not found. Please ensure sample_data.csv exists.")
        return
    
    df = pd.read_csv(data_path)
    print(f" Loaded sample data with {len(df)} transactions")
    
    # Load model
    model_path = Path("models")
    if not model_path.exists():
        print(" Model directory not found. Please run create_sample_model.py first.")
        return
    
    try:
        model = AnomalyModel.load(model_path)
        print(" Model loaded successfully")
    except Exception as e:
        print(f" Error loading model: {e}")
        return
    
    # Preprocess data
    print(" Preprocessing data...")
    df_processed = preprocess(df, cfg.features)
    feat = create_transaction_features(df_processed, cfg.features)
    X = select_feature_matrix(feat, cfg.features)
    
    # Score data
    print(" Scoring transactions...")
    scores = model.score(X)
    feat["suspicion_score"] = scores
    
    # Flag suspicious transactions
    threshold = 0.7
    flags = feat[feat["suspicion_score"] >= threshold].copy()
    
    print(f" Results:")
    print(f"   Total transactions: {len(feat)}")
    print(f"   Suspicious transactions: {len(flags)}")
    print(f"   Suspicious rate: {len(flags) / len(feat) * 100:.2f}%")
    print(f"   Average suspicion score: {scores.mean():.3f}")
    
    if len(flags) > 0:
        print(f"\n Suspicious transactions found:")
        for idx, row in flags.iterrows():
            print(f"   Transaction {idx}: Score = {row['suspicion_score']:.3f}, Amount = ${row['amount']:.2f}")
    
    print("\n Fraud detection test completed successfully!")

if __name__ == "__main__":
    test_fraud_detection()
