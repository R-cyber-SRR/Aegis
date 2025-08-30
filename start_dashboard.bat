@echo off
echo ========================================
echo    Fraud Detection Dashboard Launcher
echo ========================================
echo.
echo Starting the fraud detection dashboard...
echo.
echo This will:
echo 1. Install required packages (if needed)
echo 2. Launch the Streamlit web interface
echo 3. Open your browser automatically
echo.
echo Press any key to continue...
pause >nul

python run_dashboard.py

echo.
echo Dashboard stopped. Press any key to exit...
pause >nul
