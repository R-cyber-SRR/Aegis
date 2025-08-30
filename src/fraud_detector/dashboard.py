try:
    import streamlit as st  # pyright: ignore[reportMissingImports]
except ImportError:
    st = None
    print("Warning: Streamlit not installed. Please run: pip install streamlit")

try:
    import pandas as pd
except ImportError:
    pd = None
    print("Warning: Pandas not installed. Please run: pip install pandas")

try:
    import plotly.express as px  # pyright: ignore[reportMissingImports]
    import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
except ImportError:
    px = None
    go = None
    print("Warning: Plotly not installed. Please run: pip install plotly")

from pathlib import Path
import tempfile
import os
import io
from datetime import datetime
import json

try:
    from .config import AppConfig
    from .data_ingestion import DataIngestion
    from .preprocessing import preprocess
    from .profiling import create_transaction_features, select_feature_matrix
    from .model import AnomalyModel
    from .reporting import generate_reasons, write_flags_csv
except ImportError as e:
    print(f"Warning: Could not import fraud detection modules: {e}")
    AppConfig = None
    DataIngestion = None
    preprocess = None
    create_transaction_features = None
    select_feature_matrix = None
    AnomalyModel = None
    generate_reasons = None
    write_flags_csv = None


def run_streamlit_dashboard():
    """Run the Streamlit dashboard for fraud detection"""
    
    # Check if all required packages are available
    if st is None:
        print("❌ Streamlit is not installed. Please run: pip install streamlit")
        return
    
    if pd is None:
        print("❌ Pandas is not installed. Please run: pip install pandas")
        return
    
    if px is None or go is None:
        print("❌ Plotly is not installed. Please run: pip install plotly")
        return
    
    # Check if fraud detection modules are available
    if any([AppConfig is None, DataIngestion is None, preprocess is None, 
            create_transaction_features is None, select_feature_matrix is None,
            AnomalyModel is None, generate_reasons is None, write_flags_csv is None]):
        print("❌ Some fraud detection modules are not available. Please check your installation.")
        return
    
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🕵️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🕵️ Fraud Detection System")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_dir = st.sidebar.text_input("Model Directory", value="models/")
    
    # Threshold adjustment
    threshold = st.sidebar.slider(
        "Fraud Detection Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.05
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Upload & Analysis", 
        "🔍 Fraud Detection", 
        "📈 Results Dashboard", 
        "⚙️ Model Training"
    ])
    
    with tab1:
        st.header("Data Upload & Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'], 
            help="Upload your transaction data in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display data
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Basic statistics
                st.subheader("Data Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Info:**")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                
                with col2:
                    st.write("**Numeric Columns Summary:**")
                    st.dataframe(df.describe())
                
                # Store data in session state
                st.session_state['uploaded_data'] = df
                st.session_state['filename'] = uploaded_file.name
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with tab2:
        st.header("Fraud Detection")
        
        if 'uploaded_data' not in st.session_state:
            st.warning("⚠️ Please upload data in the Data Upload tab first.")
        else:
            df = st.session_state['uploaded_data']
            
            # Configuration file
            config_file = st.file_uploader(
                "Upload Configuration File (optional)", 
                type=['yaml', 'yml']
            )
            
            if st.button("🚀 Run Fraud Detection", type="primary"):
                with st.spinner("Running fraud detection..."):
                    try:
                        # Load configuration
                        if config_file:
                            # Save uploaded config temporarily
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                                f.write(config_file.getvalue().decode())
                                config_path = f.name
                            
                            cfg = AppConfig.from_yaml(Path(config_path))
                            os.unlink(config_path)  # Clean up
                        else:
                            # Use default config
                            cfg = AppConfig.from_yaml(Path("config.yaml"))
                        
                        # Update threshold
                        cfg.model.threshold = threshold
                        
                        # Load model
                        model_path = Path(model_dir)
                        if not model_path.exists():
                            st.error(f"❌ Model directory not found: {model_dir}")
                            return
                        
                        model = AnomalyModel.load(model_path)
                        
                        # Preprocess data
                        df_processed = preprocess(df, cfg.features)
                        feat = create_transaction_features(df_processed, cfg.features)
                        X = select_feature_matrix(feat, cfg.features)
                        
                        # Score data
                        scores = model.score(X)
                        feat["suspicion_score"] = scores
                        
                        # Flag suspicious transactions
                        flags = feat[feat["suspicion_score"] >= threshold].copy()
                        flags["reasons"] = flags.apply(generate_reasons, axis=1)
                        
                        # Store results
                        st.session_state['fraud_results'] = {
                            'scores': scores,
                            'flags': flags,
                            'features': feat,
                            'timestamp': datetime.now()
                        }
                        
                        st.success(f"✅ Fraud detection completed! Found {len(flags)} suspicious transactions.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during fraud detection: {str(e)}")
                        st.exception(e)
    
    with tab3:
        st.header("Results Dashboard")
        
        if 'fraud_results' not in st.session_state:
            st.info("ℹ️ Run fraud detection first to see results.")
        else:
            results = st.session_state['fraud_results']
            flags = results['flags']
            scores = results['scores']
            features = results['features']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", len(features))
            
            with col2:
                st.metric("Suspicious Transactions", len(flags))
            
            with col3:
                suspicious_rate = len(flags) / len(features) * 100
                st.metric("Suspicious Rate", f"{suspicious_rate:.2f}%")
            
            with col4:
                avg_score = scores.mean()
                st.metric("Average Score", f"{avg_score:.3f}")
            
            # Score distribution
            st.subheader("Suspicion Score Distribution")
            fig_hist = px.histogram(
                x=scores, 
                nbins=50,
                title="Distribution of Suspicion Scores",
                labels={'x': 'Suspicion Score', 'y': 'Count'}
            )
            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", 
                              annotation_text=f"Threshold: {threshold}")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Suspicious transactions table
            if len(flags) > 0:
                st.subheader("Suspicious Transactions")
                st.dataframe(flags, use_container_width=True)
                
                # Download results
                csv = flags.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"fraud_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("🎉 No suspicious transactions found!")
    
    with tab4:
        st.header("Model Training")
        
        st.info("""
        **Model Training Instructions:**
        
        1. Prepare your training data in CSV format
        2. Ensure your configuration file is properly set up
        3. Use the CLI command: `python -m fraud_detector train --data your_data.csv --model_dir models/`
        
        **Required Data Format:**
        - CSV file with transaction data
        - Must include columns specified in your config.yaml
        - Numerical features for anomaly detection
        """)
        
        # Training data upload
        training_file = st.file_uploader(
            "Upload Training Data", 
            type=['csv'], 
            key="training_upload"
        )
        
        if training_file and st.button("🏋️ Train Model", type="primary"):
            st.info("Training models through the web interface is not yet implemented. Please use the CLI command above.")


def run_dashboard(flags_path: Path):
    """Legacy dashboard function for CLI compatibility"""
    st.warning("This is a legacy function. Please use the Streamlit interface instead.")
    run_streamlit_dashboard()


if __name__ == "__main__":
    run_streamlit_dashboard()
