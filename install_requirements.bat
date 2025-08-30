@echo off
echo ========================================
echo    Fraud Detection System Installer
echo ========================================
echo.
echo This script will install all required packages
echo for the Fraud Detection System.
echo.
echo Press any key to continue...
pause >nul

python install_requirements.py

echo.
echo Installation complete! Press any key to exit...
pause >nul

