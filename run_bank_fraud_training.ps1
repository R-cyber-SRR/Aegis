Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Bank Fraud Detection Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements_bank_fraud.txt

Write-Host ""
Write-Host "Training the fraud detection model..." -ForegroundColor Yellow
python train_bank_model.py

Write-Host ""
Write-Host "Training complete!" -ForegroundColor Green
Read-Host "Press Enter to exit"
