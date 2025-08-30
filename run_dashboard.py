#!/usr/bin/env python3
"""
Launcher script for the Fraud Detection Dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found.")
        print("🔧 Installing required packages...")
        print("   This may take a few minutes...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Requirements installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements automatically.")
            print("🔧 Please run the installation script manually:")
            print("   python install_requirements.py")
            return
    
    # Get the path to the dashboard module
    dashboard_path = os.path.join("src", "fraud_detector", "dashboard.py")
    
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard not found at {dashboard_path}")
        return
    
    print("🚀 Launching Fraud Detection Dashboard...")
    print("📱 The dashboard will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the dashboard")
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")

if __name__ == "__main__":
    main()
