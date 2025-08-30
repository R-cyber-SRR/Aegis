#!/usr/bin/env python3
"""
Predict fraud using the trained bank fraud detection model
This script loads the trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def load_trained_model(model_dir: str = "models"):
    """Load the trained model, scaler, and feature information"""
    model_path = Path(model_dir)
    
    # Load model
    model_file = model_path / "bank_fraud_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    model = joblib.load(model_file)
    
    # Load scaler
    scaler_file = model_path / "bank_fraud_scaler.joblib"
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    scaler = joblib.load(scaler_file)
    
    # Load feature information
    features_file = model_path / "bank_fraud_features.json"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    with open(features_file, 'r') as f:
        feature_info = json.load(f)
    
    return model, scaler, feature_info

def preprocess_new_data(df: pd.DataFrame, feature_info: dict) -> pd.DataFrame:
    """Preprocess new data to match the training data format"""
    print("Preprocessing new data...")
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Basic data cleaning (same as training)
    df_processed['TransactionDate'] = pd.to_datetime(df_processed['TransactionDate'], errors='coerce')
    df_processed['TransactionAmount'] = pd.to_numeric(df_processed['TransactionAmount'], errors='coerce')
    df_processed['AccountBalance'] = pd.to_numeric(df_processed['AccountBalance'], errors='coerce')
    df_processed['CustomerAge'] = pd.to_numeric(df_processed['CustomerAge'], errors='coerce')
    df_processed['TransactionDuration'] = pd.to_numeric(df_processed['TransactionDuration'], errors='coerce')
    df_processed['LoginAttempts'] = pd.to_numeric(df_processed['LoginAttempts'], errors='coerce')
    
    # Drop rows with invalid essential fields
    initial_count = len(df_processed)
    df_processed = df_processed.dropna(subset=['TransactionDate', 'TransactionAmount', 'AccountID'])
    print(f"Dropped {initial_count - len(df_processed)} rows with missing essential data")
    
    # Create features (same as training)
    df_processed['hour'] = df_processed['TransactionDate'].dt.hour
    df_processed['day_of_week'] = df_processed['TransactionDate'].dt.dayofweek
    df_processed['month'] = df_processed['TransactionDate'].dt.month
    
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
    
    df_processed['time_of_day'] = df_processed['hour'].apply(categorize_time)
    
    # Amount-based features
    df_processed['amount_log'] = np.log1p(df_processed['TransactionAmount'].abs())
    
    # Account behavior features (using training data if available)
    # For new data, we'll use the current transaction as reference
    df_processed['avg_amount'] = df_processed['TransactionAmount'].mean()
    df_processed['std_amount'] = df_processed['TransactionAmount'].std()
    df_processed['tx_count'] = 1  # Default for new data
    df_processed['avg_balance'] = df_processed['AccountBalance'].mean()
    df_processed['customer_age'] = df_processed['CustomerAge'].mean()
    
    # Create relative features
    df_processed['amount_to_avg_ratio'] = df_processed['TransactionAmount'] / df_processed['avg_amount'].replace(0, 1)
    df_processed['amount_zscore'] = (df_processed['TransactionAmount'] - df_processed['avg_amount']) / df_processed['std_amount'].replace(0, 1)
    df_processed['balance_to_amount_ratio'] = df_processed['AccountBalance'] / (df_processed['TransactionAmount'].abs() + 1)
    
    # Categorical encoding
    df_processed['tx_type_encoded'] = pd.Categorical(df_processed['TransactionType']).codes
    df_processed['channel_encoded'] = pd.Categorical(df_processed['Channel']).codes
    df_processed['time_of_day_encoded'] = pd.Categorical(df_processed['time_of_day']).codes
    df_processed['location_encoded'] = pd.Categorical(df_processed['Location']).codes
    
    # Risk indicators
    df_processed['high_amount_flag'] = (df_processed['TransactionAmount'] > df_processed['avg_amount'] * 3).astype(int)
    df_processed['unusual_time_flag'] = ((df_processed['hour'] < 6) | (df_processed['hour'] > 22)).astype(int)
    df_processed['multiple_login_flag'] = (df_processed['LoginAttempts'] > 1).astype(int)
    
    return df_processed

def select_features_for_prediction(df: pd.DataFrame, feature_info: dict) -> pd.DataFrame:
    """Select the same features used during training"""
    feature_names = feature_info['feature_names']
    
    # Filter to only include columns that exist
    available_features = [col for col in feature_names if col in df.columns]
    missing_features = [col for col in feature_names if col not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        print("These will be filled with zeros.")
    
    # Create feature matrix
    X = pd.DataFrame(index=df.index)
    
    for feature in feature_names:
        if feature in df.columns:
            X[feature] = df[feature]
        else:
            X[feature] = 0
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Replace infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    return X

def predict_fraud(df: pd.DataFrame, model, scaler, feature_info: dict, threshold: float = 0.7) -> pd.DataFrame:
    """Make fraud predictions on the data"""
    print("Making fraud predictions...")
    
    # Preprocess the data
    df_processed = preprocess_new_data(df, feature_info)
    
    # Select features
    X = select_features_for_prediction(df_processed, feature_info)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)
    
    # Convert to suspicion scores (higher = more suspicious)
    suspicion_scores = -scores  # Invert because Isolation Forest returns negative for anomalies
    
    # Normalize scores to 0-1 range
    suspicion_scores = (suspicion_scores - suspicion_scores.min()) / (suspicion_scores.max() - suspicion_scores.min())
    
    # Add results to dataframe
    df_processed['suspicion_score'] = suspicion_scores
    df_processed['is_anomaly'] = (predictions == -1)
    df_processed['fraud_flag'] = (suspicion_scores >= threshold)
    
    # Add prediction confidence
    df_processed['prediction_confidence'] = np.abs(suspicion_scores - 0.5) * 2
    
    return df_processed

def generate_fraud_report(df_with_predictions: pd.DataFrame, threshold: float = 0.7) -> None:
    """Generate a comprehensive fraud detection report"""
    print("\n=== FRAUD DETECTION REPORT ===")
    
    total_transactions = len(df_with_predictions)
    flagged_transactions = len(df_with_predictions[df_with_predictions['fraud_flag'] == True])
    anomaly_transactions = len(df_with_predictions[df_with_predictions['is_anomaly'] == True])
    
    print(f"Total Transactions Analyzed: {total_transactions}")
    print(f"Transactions Flagged as Suspicious (Score >= {threshold}): {flagged_transactions}")
    print(f"Anomalies Detected: {anomaly_transactions}")
    print(f"Flag Rate: {flagged_transactions/total_transactions:.2%}")
    print(f"Anomaly Rate: {anomaly_transactions/total_transactions:.2%}")
    
    # Score distribution
    scores = df_with_predictions['suspicion_score']
    print(f"\nSuspicion Score Statistics:")
    print(f"  Min: {scores.min():.3f}")
    print(f"  Max: {scores.max():.3f}")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Median: {scores.median():.3f}")
    print(f"  Std: {scores.std():.3f}")
    
    # High-risk transactions
    high_risk = df_with_predictions[df_with_predictions['suspicion_score'] >= 0.9]
    if len(high_risk) > 0:
        print(f"\nHigh-Risk Transactions (Score >= 0.9): {len(high_risk)}")
        print("Top 5 highest risk transactions:")
        for idx, row in high_risk.nlargest(5, 'suspicion_score').iterrows():
            print(f"  {row['TransactionID']}: Score {row['suspicion_score']:.3f}, "
                  f"Amount ${row['TransactionAmount']:.2f}, Account {row['AccountID']}")
    
    # Risk factors analysis
    print(f"\nRisk Factor Analysis:")
    risk_factors = ['high_amount_flag', 'unusual_time_flag', 'multiple_login_flag']
    for factor in risk_factors:
        if factor in df_with_predictions.columns:
            factor_count = df_with_predictions[factor].sum()
            print(f"  {factor}: {factor_count} transactions ({factor_count/total_transactions:.1%})")

def save_results(df_with_predictions: pd.DataFrame, output_file: str = "fraud_predictions.csv") -> None:
    """Save the prediction results to a CSV file"""
    # Select important columns for output
    output_columns = [
        'TransactionID', 'AccountID', 'TransactionAmount', 'TransactionDate',
        'TransactionType', 'Location', 'Channel', 'suspicion_score', 'fraud_flag',
        'is_anomaly', 'prediction_confidence'
    ]
    
    # Filter to only include columns that exist
    available_output_columns = [col for col in output_columns if col in df_with_predictions.columns]
    
    output_df = df_with_predictions[available_output_columns].copy()
    
    # Sort by suspicion score (highest first)
    output_df = output_df.sort_values('suspicion_score', ascending=False)
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

def main():
    """Main prediction function"""
    print("=== Bank Fraud Detection - Prediction ===")
    
    # Check if model exists
    model_dir = "models"
    if not Path(model_dir).exists():
        print(f"Error: Model directory '{model_dir}' not found!")
        print("Please run 'train_bank_model.py' first to train the model.")
        return
    
    try:
        # Load the trained model
        print("Loading trained model...")
        model, scaler, feature_info = load_trained_model(model_dir)
        print("Model loaded successfully!")
        
        # Check if we have new data to predict on
        # For demonstration, we'll use the same training data
        csv_file = "bank_transactions_data_2.csv"
        if not Path(csv_file).exists():
            print(f"Error: {csv_file} not found!")
            return
        
        # Load new data
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} transactions")
        
        # Make predictions
        df_with_predictions = predict_fraud(df, model, scaler, feature_info, threshold=0.7)
        
        # Generate report
        generate_fraud_report(df_with_predictions, threshold=0.7)
        
        # Save results
        save_results(df_with_predictions, "fraud_predictions.csv")
        
        print("\n=== Prediction Complete! ===")
        print("Check 'fraud_predictions.csv' for detailed results.")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
