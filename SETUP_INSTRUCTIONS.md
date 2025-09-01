# Fraud Detection System - Setup & Usage Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Sample Model
```bash
python create_sample_model.py
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access the Web Interface
Open your browser and go to: http://localhost:5000

## 📁 Project Structure

```
FTHD/
├── app.py                          # Flask web application
├── run_dashboard.py                # Streamlit dashboard launcher
├── create_sample_model.py          # Script to create sample model
├── test_fraud_detection.py         # Test script
├── sample_data.csv                 # Sample transaction data
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── models/                         # Trained models directory
│   └── isolation_forest.joblib     # Sample trained model
├── src/fraud_detector/             # Core fraud detection modules
│   ├── config.py                   # Configuration management
│   ├── model.py                    # Anomaly detection model
│   ├── preprocessing.py            # Data preprocessing
│   ├── profiling.py                # Feature engineering
│   ├── reporting.py                # Results generation
│   └── dashboard.py                # Streamlit dashboard
└── templates/                      # Web templates
    └── index.html                  # Main web interface
```

## 🔧 Configuration

The system uses `config.yaml` to configure:

- **Required columns**: timestamp, amount, user_id, merchant_id, tx_type
- **Optional columns**: ip_address, device_fingerprint, latitude, longitude
- **Model parameters**: algorithm, contamination, threshold

## 📊 Data Format

Your CSV file should contain these columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| timestamp | datetime | Transaction timestamp | ✅ |
| amount | float | Transaction amount | ✅ |
| user_id | int/string | User identifier | ✅ |
| merchant_id | int/string | Merchant identifier | ✅ |
| tx_type | string | Transaction type | ✅ |
| ip_address | string | IP address | ❌ |
| device_fingerprint | string | Device identifier | ❌ |
| latitude | float | Location latitude | ❌ |
| longitude | float | Location longitude | ❌ |

## 🎯 Usage Examples

### Web Interface
1. Start the application: `python app.py`
2. Open http://localhost:5000
3. Upload your CSV file
4. Set detection threshold
5. Run fraud detection
6. View and download results

### Command Line
```bash
# Train a new model
python -m src.fraud_detector.cli train --data your_data.csv --model_dir models/ --config config.yaml

# Score new data
python -m src.fraud_detector.cli score --data new_data.csv --model_dir models/ --out results.csv

# Run Streamlit dashboard
python run_dashboard.py
```

### Programmatic Usage
```python
from src.fraud_detector.config import AppConfig
from src.fraud_detector.model import AnomalyModel
from src.fraud_detector.preprocessing import preprocess
from src.fraud_detector.profiling import create_transaction_features, select_feature_matrix

# Load configuration and model
cfg = AppConfig.from_yaml(Path("config.yaml"))
model = AnomalyModel.load(Path("models/"))

# Process data
df = pd.read_csv("your_data.csv")
df_processed = preprocess(df, cfg.features)
feat = create_transaction_features(df_processed, cfg.features)
X = select_feature_matrix(feat, cfg.features)

# Score transactions
scores = model.score(X)
```

## 🧪 Testing

Run the test script to verify everything works:
```bash
python test_fraud_detection.py
```

## 🔍 Troubleshooting

### Common Issues

1. **"Model directory not found"**
   - Solution: Run `python create_sample_model.py`

2. **"Configuration file not found"**
   - Solution: Ensure `config.yaml` exists in the project root

3. **"Missing required columns"**
   - Solution: Check your CSV file has all required columns

4. **Import errors**
   - Solution: Install dependencies with `pip install -r requirements.txt`

### Debug Mode

Run the application in debug mode for detailed error messages:
```bash
python app.py
```

## 📈 Model Training

To train a model on your own data:

1. Prepare your training data in CSV format
2. Ensure it has all required columns
3. Run: `python -m src.fraud_detector.cli train --data your_data.csv --model_dir models/`

## 🎨 Customization

- **Adjust threshold**: Modify the threshold in the web interface or config.yaml
- **Change algorithm**: Update the algorithm in config.yaml
- **Add features**: Extend the feature engineering in profiling.py
- **Custom preprocessing**: Modify preprocessing.py for your data format

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run the test script to verify functionality
3. Check the console output for error messages
4. Ensure all dependencies are installed correctly


