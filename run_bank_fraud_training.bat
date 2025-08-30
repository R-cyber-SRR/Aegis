@echo off
echo ========================================
echo    Bank Fraud Detection Training
echo ========================================
echo.

echo Installing dependencies...
pip install -r requirements_bank_fraud.txt

echo.
echo Training the fraud detection model...
python train_bank_model.py

echo.
echo Training complete! Press any key to exit.
pause > nul
