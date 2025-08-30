# Bank Fraud Detection System

A comprehensive fraud detection system specifically designed for bank transaction data using machine learning and anomaly detection techniques.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_bank_fraud.txt
```

### 2. Train the Model
```bash
python train_bank_model.py
```
This will:
- Load the `bank_transactions_data_2.csv` file
- Preprocess and engineer features
- Train an Isolation Forest model
- Save the trained model to the `models/` directory

### 3. Run Fraud Detection
```bash
python predict_bank_fraud.py
```
This will:
- Load the trained model
- Make predictions on the same data
- Generate a comprehensive fraud report
- Save results to `fraud_predictions.csv`

### 4. Launch Web Dashboard
```bash
streamlit run bank_fraud_dashboard.py
```
This opens an interactive web interface for:
- Uploading new transaction data
- Running fraud detection
- Viewing results and analytics

## 📁 Project Structure

```
FTHD/
├── bank_transactions_data_2.csv      # Training data
├── config_bank_data.yaml             # Configuration for bank data
├── train_bank_model.py               # Model training script
├── predict_bank_fraud.py             # Prediction script
├── bank_fraud_dashboard.py           # Streamlit web interface
├── requirements_bank_fraud.txt       # Python dependencies
├── models/                           # Trained models directory
│   ├── bank_fraud_model.joblib      # Trained Isolation Forest model
│   ├── bank_fraud_scaler.joblib     # Feature scaler
│   └── bank_fraud_features.json     # Feature information
└── README_BANK_FRAUD.md             # This file
```

## 🔧 Configuration

The system uses `config_bank_data.yaml` to configure:

- **Required columns**: TransactionDate, TransactionAmount, AccountID, MerchantID, TransactionType
- **Optional columns**: IP Address, DeviceID, Location, Channel, CustomerAge, CustomerOccupation, TransactionDuration, LoginAttempts, AccountBalance, PreviousTransactionDate
- **Model parameters**: Isolation Forest with 5% contamination rate

## 📊 Data Format

Your CSV file should contain these columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| TransactionID | string | Unique transaction identifier | ✅ |
| AccountID | string | Customer account identifier | ✅ |
| TransactionAmount | float | Transaction amount | ✅ |
| TransactionDate | datetime | Transaction timestamp | ✅ |
| TransactionType | string | Transaction type (Debit/Credit) | ✅ |
| MerchantID | string | Merchant identifier | ✅ |
| Location | string | Transaction location | ❌ |
| Channel | string | Transaction channel (ATM/Online/Branch) | ❌ |
| CustomerAge | int | Customer age | ❌ |
| CustomerOccupation | string | Customer occupation | ❌ |
| TransactionDuration | int | Transaction duration in seconds | ❌ |
| LoginAttempts | int | Number of login attempts | ❌ |
| AccountBalance | float | Account balance | ❌ |
| PreviousTransactionDate | datetime | Previous transaction timestamp | ❌ |

## 🎯 Features

### Feature Engineering
The system automatically creates these features:

- **Time-based features**: Hour, day of week, month, time of day categories
- **Amount features**: Log-transformed amounts, ratios to account averages
- **Behavioral features**: Account transaction patterns, frequency analysis
- **Risk indicators**: Unusual amounts, off-hours transactions, multiple login attempts
- **Categorical encodings**: Transaction types, channels, locations

### Model Architecture
- **Algorithm**: Isolation Forest (unsupervised anomaly detection)
- **Contamination**: 5% (configurable)
- **Features**: 20+ engineered features
- **Scaling**: StandardScaler for feature normalization

## 📈 Usage Examples

### Command Line Training
```bash
# Train a new model
python train_bank_model.py

# Make predictions
python predict_bank_fraud.py
```

### Web Interface
```bash
# Launch dashboard
streamlit run bank_fraud_dashboard.py
```

Then:
1. Upload your CSV file
2. Click "Run Fraud Detection"
3. View results and download reports

### Programmatic Usage
```python
from predict_bank_fraud import load_trained_model, predict_fraud
import pandas as pd

# Load model
model, scaler, feature_info = load_trained_model()

# Load data
df = pd.read_csv("your_transactions.csv")

# Make predictions
results = predict_fraud(df, model, scaler, feature_info, threshold=0.7)

# View results
print(f"Flagged transactions: {results['fraud_flag'].sum()}")
```

## 🧪 Model Performance

The trained model provides:

- **Suspicion Scores**: 0-1 scale (higher = more suspicious)
- **Anomaly Detection**: Binary classification (normal vs. anomalous)
- **Risk Factors**: Multiple risk indicators for each transaction
- **Confidence Metrics**: Prediction confidence scores

## 🔍 Fraud Detection Logic

The system flags transactions as suspicious based on:

1. **Amount Anomalies**: Transactions significantly above account average
2. **Time Anomalies**: Transactions during unusual hours
3. **Behavioral Changes**: Deviations from account patterns
4. **Risk Indicators**: Multiple login attempts, unusual locations
5. **Statistical Outliers**: Transactions outside normal distribution

## 📊 Output Files

### Training Output
- `models/bank_fraud_model.joblib`: Trained model
- `models/bank_fraud_scaler.joblib`: Feature scaler
- `models/bank_fraud_features.json`: Feature information

### Prediction Output
- `fraud_predictions.csv`: Complete results with suspicion scores
- Console report with summary statistics
- High-risk transaction details

## 🎨 Customization

### Adjusting Thresholds
- Modify the threshold in `predict_bank_fraud.py` or the dashboard
- Lower threshold = more sensitive (more false positives)
- Higher threshold = less sensitive (more false negatives)

### Adding New Features
- Extend the feature engineering in `train_bank_model.py`
- Update the feature selection in `select_features_for_training()`
- Retrain the model with new features

### Model Parameters
- Adjust contamination rate in `train_isolation_forest_model()`
- Modify Isolation Forest parameters (n_estimators, max_samples)
- Change random state for reproducibility

## 🚨 Troubleshooting

### Common Issues

1. **"Model not found"**
   - Solution: Run `python train_bank_model.py` first

2. **"Missing columns"**
   - Solution: Ensure your CSV has all required columns
   - Check column names match exactly (case-sensitive)

3. **"Import errors"**
   - Solution: Install dependencies with `pip install -r requirements_bank_fraud.txt`

4. **"Memory errors"**
   - Solution: Process data in smaller batches
   - Reduce feature complexity

### Debug Mode
Run scripts with verbose output:
```bash
python -u train_bank_model.py
python -u predict_bank_fraud.py
```

## 📈 Performance Tips

1. **Data Quality**: Clean data before training (handle missing values, outliers)
2. **Feature Selection**: Focus on most predictive features
3. **Threshold Tuning**: Balance false positives vs. false negatives
4. **Regular Retraining**: Update model with new data patterns
5. **Monitoring**: Track model performance over time

## 🔒 Security Considerations

- **Data Privacy**: Anonymize sensitive customer information
- **Model Security**: Protect trained models from unauthorized access
- **Audit Trail**: Log all fraud detection decisions
- **Compliance**: Ensure adherence to financial regulations

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify data format matches requirements
3. Ensure all dependencies are installed
4. Check console output for error messages

## 🎉 Success!

Your bank fraud detection system is now ready! The system can:
- Detect anomalous transactions using advanced ML techniques
- Provide detailed suspicion scores for each transaction
- Flag high-risk transactions above configurable thresholds
- Generate comprehensive reports and analytics
- Handle both command-line and web interfaces
- Scale to large transaction volumes

## 🔄 Future Enhancements

Potential improvements:
- **Deep Learning**: Neural networks for complex pattern recognition
- **Real-time Processing**: Stream processing for live transactions
- **Multi-model Ensemble**: Combine multiple detection algorithms
- **Explainable AI**: Provide reasoning for fraud flags
- **API Integration**: REST API for enterprise deployment
- **Advanced Analytics**: Customer risk profiling, trend analysis
