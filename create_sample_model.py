#!/usr/bin/env python3
"""
Script to create a sample model for demonstration purposes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.fraud_detector.config import AppConfig
from src.fraud_detector.model import AnomalyModel
from src.fraud_detector.preprocessing import preprocess
from src.fraud_detector.profiling import create_transaction_features, select_feature_matrix

def create_sample_data():
    """Create sample transaction data for training"""
    np.random.seed(42)
    n_transactions = 1000
    
    # Generate sample data with the required columns
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_transactions, freq='h'),
        'amount': np.random.lognormal(mean=4, sigma=1, size=n_transactions),
        'user_id': np.random.randint(1, 101, size=n_transactions),
        'merchant_id': np.random.randint(1, 51, size=n_transactions),
        'tx_type': np.random.choice(['purchase', 'transfer', 'withdrawal'], size=n_transactions),
        'ip_address': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_transactions)],
        'device_fingerprint': [f"device_{np.random.randint(1000,9999)}" for _ in range(n_transactions)],
        'latitude': np.random.uniform(30, 50, n_transactions),
        'longitude': np.random.uniform(-120, -70, n_transactions)
    }
    
    # Add some anomalies (fraudulent transactions)
    anomaly_indices = np.random.choice(n_transactions, size=int(n_transactions * 0.02), replace=False)
    for idx in anomaly_indices:
        data['amount'][idx] *= np.random.uniform(5, 20)  # Much larger amounts
        data['latitude'][idx] = np.random.uniform(-90, 90)  # Random location
        data['longitude'][idx] = np.random.uniform(-180, 180)
    
    return pd.DataFrame(data)

def main():
    """Create and save a sample model"""
    print("🔧 Creating sample model...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ Configuration file not found. Please ensure config.yaml exists.")
        return
    
    cfg = AppConfig.from_yaml(config_path)
    
    # Create sample data
    print("📊 Generating sample training data...")
    df = create_sample_data()
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    df_processed = preprocess(df, cfg.features)
    feat = create_transaction_features(df_processed, cfg.features)
    X = select_feature_matrix(feat, cfg.features)
    
    # Train model
    print("🏋️ Training model...")
    model = AnomalyModel(cfg.model)
    model.fit(X)
    
    # Save model
    print("💾 Saving model...")
    model_path = model.save(models_dir)
    
    print(f"✅ Sample model created successfully at: {model_path}")
    print("🎉 You can now use the fraud detection application!")

if __name__ == "__main__":
    main()
