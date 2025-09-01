#!/usr/bin/env python3
"""
Installation script for Fraud Detection System
This script will install all required packages and check for any missing dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f" {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print(" Fraud Detection System - Package Installation")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(" Python 3.8+ is required. Current version:", sys.version)
        return
    
    print(f" Python version: {sys.version}")
    
    # Upgrade pip first
    print("\n Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    print("\n Installing required packages...")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        print("\n Failed to install requirements. Trying individual packages...")
        
        # Try installing packages individually
        packages = [
            "pandas>=2.1.0",
            "numpy>=1.24.0", 
            "scikit-learn>=1.3.0",
            "flask>=2.3.0",
            "plotly>=5.15.0",
            "streamlit>=1.28.0",
            "pyyaml>=6.0.0",
            "python-dotenv>=1.0.0",
            "joblib>=1.3.0",
            "uvicorn>=0.20.0",
            "orjson>=3.8.0"
        ]
        
        for package in packages:
            run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
    
    # Verify installations
    print("\n Verifying package installations...")
    required_packages = [
        "pandas", "numpy", "sklearn", "flask", "plotly", 
        "streamlit", "yaml", "dotenv", "joblib", "uvicorn"
    ]
    
    missing_packages = []
    for package in required_packages:
        if check_package(package):
            print(f"✅ {package} - OK")
        else:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them manually or check your internet connection.")
    else:
        print("\n🎉 All packages installed successfully!")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import streamlit as st
        print("✅ Streamlit import - OK")
    except ImportError:
        print("❌ Streamlit import - FAILED")
    
    try:
        import pandas as pd
        print("✅ Pandas import - OK")
    except ImportError:
        print("❌ Pandas import - FAILED")
    
    try:
        import plotly.express as px
        print("✅ Plotly import - OK")
    except ImportError:
        print("❌ Plotly import - FAILED")
    
    print("\n🚀 Installation complete!")
    print("\nTo run the dashboard:")
    print("  python run_dashboard.py")
    print("\nOr on Windows:")
    print("  start_dashboard.bat")

if __name__ == "__main__":
    main()

