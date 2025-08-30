#!/usr/bin/env python3
"""
Train a fixed fraud detection model using bank_transactions_data_2.csv
This script creates a model specifically tailored to the bank transaction data structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_bank_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the bank transaction data"""
    print(f"Loading data from {csv_path}...")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transactions with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Basic data cleaning
    print("Cleaning data...")
    
    # Convert TransactionDate to datetime
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
    
    # Convert TransactionAmount to numeric
    df['TransactionAmount'] = pd.to_numeric(df['TransactionAmount'], errors='coerce')
    
    # Convert AccountBalance to numeric
    df['AccountBalance'] = pd.to_numeric(df['AccountBalance'], errors='coerce')
    
    # Convert CustomerAge to numeric
    df['CustomerAge'] = pd.to_numeric(df['CustomerAge'], errors='coerce')
    
    # Convert TransactionDuration to numeric
    df['TransactionDuration'] = pd.to_numeric(df['TransactionDuration'], errors='coerce')
    
    # Convert LoginAttempts to numeric
    df['LoginAttempts'] = pd.to_numeric(df['LoginAttempts'], errors='coerce')
    
    # Drop rows with invalid essential fields
    initial_count = len(df)
    df = df.dropna(subset=['TransactionDate', 'TransactionAmount', 'AccountID'])
    print(f"Dropped {initial_count - len(df)} rows with missing essential data")
    
    # Create additional features
    print("Creating features...")
    
    # Time-based features
    df['hour'] = df['TransactionDate'].dt.hour
    df['day_of_week'] = df['TransactionDate'].dt.dayofweek
    df['month'] = df['TransactionDate'].dt.month
    
    # Time of day categories
    def categorize_time(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    df['time_of_day'] = df['hour'].apply(categorize_time)
    
    # Amount-based features
    df['amount_log'] = np.log1p(df['TransactionAmount'].abs())
    
    # Account behavior features
    account_stats = df.groupby('AccountID').agg({
        'TransactionAmount': ['mean', 'std', 'count', 'min', 'max'],
        'AccountBalance': 'mean',
        'CustomerAge': 'first'
    }).reset_index()
    
    account_stats.columns = ['AccountID', 'avg_amount', 'std_amount', 'tx_count', 'min_amount', 'max_amount', 'avg_balance', 'customer_age']
    
    # Merge account statistics back to main dataframe
    df = df.merge(account_stats, on='AccountID', how='left')
    
    # Create relative features
    df['amount_to_avg_ratio'] = df['TransactionAmount'] / df['avg_amount'].replace(0, 1)
    df['amount_zscore'] = (df['TransactionAmount'] - df['avg_amount']) / df['std_amount'].replace(0, 1)
    df['balance_to_amount_ratio'] = df['AccountBalance'] / (df['TransactionAmount'].abs() + 1)
    
    # Categorical encoding
    df['tx_type_encoded'] = pd.Categorical(df['TransactionType']).codes
    df['channel_encoded'] = pd.Categorical(df['Channel']).codes
    df['time_of_day_encoded'] = pd.Categorical(df['time_of_day']).codes
    
    # Location-based features (simplified)
    df['location_encoded'] = pd.Categorical(df['Location']).codes
    
    # Risk indicators
    df['high_amount_flag'] = (df['TransactionAmount'] > df['avg_amount'] * 3).astype(int)
    df['unusual_time_flag'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    df['multiple_login_flag'] = (df['LoginAttempts'] > 1).astype(int)
    
    print(f"Feature engineering complete. Final shape: {df.shape}")
    return df

def select_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Select the features to use for training the model"""
    feature_columns = [
        # Amount features
        'amount_log', 'amount_to_avg_ratio', 'amount_zscore', 'balance_to_amount_ratio',
        
        # Time features
        'hour', 'day_of_week', 'month', 'time_of_day_encoded',
        
        # Account behavior features
        'tx_count', 'avg_amount', 'std_amount', 'avg_balance', 'customer_age',
        
        # Transaction context features
        'tx_type_encoded', 'channel_encoded', 'location_encoded',
        'TransactionDuration', 'LoginAttempts',
        
        # Risk indicators
        'high_amount_flag', 'unusual_time_flag', 'multiple_login_flag'
    ]
    
    # Filter to only include columns that exist
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(available_features)} features for training: {available_features}")
    
    X = df[available_features].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Replace infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    return X

def train_isolation_forest_model(X: pd.DataFrame, contamination: float = 0.05) -> tuple:
    """Train an Isolation Forest model for anomaly detection"""
    print(f"Training Isolation Forest model with contamination={contamination}...")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the model
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    
    model.fit(X_scaled)
    
    print("Model training complete!")
    return model, scaler

def save_model_and_scaler(model, scaler, feature_names, model_dir: str = "models"):
    """Save the trained model and scaler"""
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    
    # Save the model
    model_file = model_path / "bank_fraud_model.joblib"
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    # Save the scaler
    scaler_file = model_path / "bank_fraud_scaler.joblib"
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to {scaler_file}")
    
    # Save feature names for later use
    features_file = model_path / "bank_fraud_features.json"
    import json
    with open(features_file, 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'model_type': 'isolation_forest',
            'contamination': 0.05,
            'random_state': 42
        }, f, indent=2)
    print(f"Feature information saved to {features_file}")

def main():
    """Main training function"""
    print("=== Bank Fraud Detection Model Training ===")
    
    # File paths
    csv_file = "bank_transactions_data_2.csv"
    model_dir = "models"
    
    # Check if CSV file exists
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        return
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_bank_data(csv_file)
        
        # Select features for training
        X = select_features_for_training(df)
        
        # Train the model
        model, scaler = train_isolation_forest_model(X, contamination=0.05)
        
        # Save the model and scaler
        save_model_and_scaler(model, scaler, list(X.columns), model_dir)
        
        # Test the model
        print("\n=== Model Testing ===")
        predictions = model.predict(X)
        scores = model.score_samples(X)
        
        # Convert to suspicion scores (higher = more suspicious)
        suspicion_scores = -scores  # Invert because Isolation Forest returns negative for anomalies
        
        # Normalize scores to 0-1 range
        suspicion_scores = (suspicion_scores - suspicion_scores.min()) / (suspicion_scores.max() - suspicion_scores.min())
        
        # Count anomalies
        anomaly_count = np.sum(predictions == -1)
        total_count = len(predictions)
        
        print(f"Total transactions: {total_count}")
        print(f"Detected anomalies: {anomaly_count}")
        print(f"Anomaly rate: {anomaly_count/total_count:.2%}")
        print(f"Suspicion score range: {suspicion_scores.min():.3f} - {suspicion_scores.max():.3f}")
        
        # Show some example suspicious transactions
        df['suspicion_score'] = suspicion_scores
        suspicious_transactions = df[df['suspicion_score'] > 0.8].sort_values('suspicion_score', ascending=False)
        
        if len(suspicious_transactions) > 0:
            print(f"\nTop 5 most suspicious transactions:")
            for idx, row in suspicious_transactions.head().iterrows():
                print(f"  Transaction {row['TransactionID']}: Score {row['suspicion_score']:.3f}, "
                      f"Amount ${row['TransactionAmount']:.2f}, Account {row['AccountID']}")
        
        print("\n=== Training Complete! ===")
        print("The model is now ready to use for fraud detection.")
        print(f"Model files saved in: {model_dir}/")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
