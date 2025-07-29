import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Insurance Pricing Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class InsuranceDashboard:
    def __init__(self):
        """
        Initialize the dashboard with data sources.
        """
        self.data_file = "data/insurance_expanded.csv"
        self.comparison_file = "data/model_comparison.csv"
        self.individual_comparison_file = "data/individual_model_comparison.csv"
        
    def load_data(self):
        """
        Load all necessary data for the dashboard.
        """
        try:
            # Load main dataset
            self.df = pd.read_csv(self.data_file)
            
            # Load comparison results
            if os.path.exists(self.comparison_file):
                self.comparison_df = pd.read_csv(self.comparison_file)
            else:
                self.comparison_df = None
                
            if os.path.exists(self.individual_comparison_file):
                self.individual_comparison_df = pd.read_csv(self.individual_comparison_file)
            else:
                self.individual_comparison_df = None
                
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def show_dataset_overview(self):
        """
        Display dataset overview and basic statistics.
        """
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(self.df):,}")
        
        with col2:
            st.metric("Features", len(self.df.columns))
        
        with col3:
            st.metric("Memory Usage", f"{self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        with col4:
            missing_values = self.df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        # Show data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes,
            'Non-Null Count': self.df.count(),
            'Null Count': self.df.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Show basic statistics
        st.subheader("Numerical Features Statistics")
        st.dataframe(self.df.describe(), use_container_width=True)
    
    def show_target_analysis(self):
        """
        Display target variable analysis.
        """
        st.header("🎯 Target Variable Analysis")
        
        charges = self.df['charges']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Summary statistics
            st.subheader("Summary Statistics")
            stats_data = {
                'Metric': ['Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    f"${charges.mean():,.2f}",
                    f"${charges.median():,.2f}",
                    f"${charges.std():,.2f}",
                    f"${charges.min():,.2f}",
                    f"${charges.max():,.2f}",
                    f"{charges.skew():.3f}",
                    f"{charges.kurtosis():.3f}"
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Distribution plot
            st.subheader("Distribution of Insurance Charges")
            fig = px.histogram(
                self.df, 
                x='charges', 
                nbins=50,
                title="Distribution of Insurance Charges",
                labels={'charges': 'Charges ($)', 'count': 'Frequency'}
            )
            fig.add_vline(x=charges.mean(), line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: ${charges.mean():,.0f}")
            fig.add_vline(x=charges.median(), line_dash="dash", line_color="green", 
                         annotation_text=f"Median: ${charges.median():,.0f}")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_feature_analysis(self):
        """
        Display feature analysis and correlations.
        """
        st.header("🔍 Feature Analysis")
        
        # Numerical features correlation
        numerical_cols = ['age', 'bmi', 'charges']
        correlation_matrix = self.df[numerical_cols].corr()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Correlations")
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Correlation Values")
            st.dataframe(correlation_matrix.round(3), use_container_width=True)
        
        # Categorical features analysis
        st.subheader("Categorical Features Impact")
        
        categorical_features = ['sex', 'smoker', 'region']
        
        for feature in categorical_features:
            if feature in self.df.columns:
                fig = px.box(
                    self.df, 
                    x=feature, 
                    y='charges',
                    title=f"Charges by {feature.title()}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def show_model_comparison(self):
        """
        Display model comparison results.
        """
        st.header("🏆 Model Comparison")
        
        if self.comparison_df is not None:
            st.subheader("Experiment Tracker Results")
            
            # Sort by MAE (best first)
            comparison_sorted = self.comparison_df.sort_values('mae')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance metrics table
                st.dataframe(comparison_sorted, use_container_width=True)
            
            with col2:
                # Performance visualization
                fig = px.bar(
                    comparison_sorted,
                    x='model_name',
                    y='mae',
                    title="Model Performance (MAE)",
                    labels={'mae': 'Mean Absolute Error ($)', 'model_name': 'Model'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Best model highlight
            best_model = comparison_sorted.iloc[0]
            st.success(f"🏆 **Best Model**: {best_model['model_name']} with MAE: ${best_model['mae']:,.2f}")
        
        if self.individual_comparison_df is not None:
            st.subheader("Individual Model Training Results")
            
            # Sort by MAE (best first)
            individual_sorted = self.individual_comparison_df.sort_values('MAE')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(individual_sorted, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    individual_sorted,
                    x='Model',
                    y='MAE',
                    title="Individual Model Performance (MAE)",
                    labels={'MAE': 'Mean Absolute Error ($)', 'Model': 'Model'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    def show_mlflow_integration(self):
        """
        Display MLflow integration information.
        """
        st.header("🔬 MLflow Experiment Tracking")
        
        st.info("""
        **MLflow Integration Benefits:**
        - **Reproducibility**: All experiments logged with exact parameters
        - **Comparison**: Easy side-by-side model comparison
        - **Versioning**: Model artifacts are versioned and stored
        - **Collaboration**: Share results with team members
        - **Monitoring**: Track model performance over time
        """)
        
        st.subheader("Access MLflow UI")
        st.code("mlflow ui", language="bash")
        st.write("Then open your browser to: http://localhost:5000")
        
        # Show MLflow runs if available
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name("insurance_pricing_benchmark")
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.mae ASC"]
                )
                
                if runs:
                    st.subheader("Recent MLflow Runs")
                    
                    run_data = []
                    for run in runs[:10]:  # Show last 10 runs
                        run_data.append({
                            'Run Name': run.data.tags.get('mlflow.runName', 'Unknown'),
                            'MAE': f"${run.data.metrics.get('mae', 0):,.2f}",
                            'R²': f"{run.data.metrics.get('r2', 0):.4f}",
                            'Status': run.info.status
                        })
                    
                    if run_data:
                        runs_df = pd.DataFrame(run_data)
                        st.dataframe(runs_df, use_container_width=True)
                else:
                    st.info("No MLflow runs found. Run the experiment tracker to generate results.")
            else:
                st.info("No MLflow experiment found. Run the experiment tracker to create experiments.")
        except:
            st.info("MLflow client not available. Make sure MLflow is properly configured.")
    
    def show_predictions_interface(self):
        """
        Display a simple prediction interface.
        """
        st.header("🔮 Insurance Cost Predictor")
        
        st.write("Enter patient information to predict insurance costs:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 30)
            sex = st.selectbox("Sex", ["male", "female"])
            bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        
        with col2:
            children = st.slider("Number of Children", 0, 10, 0)
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
        
        # Create sample data for prediction
        sample_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })
        
        st.subheader("Sample Data for Prediction")
        st.dataframe(sample_data, use_container_width=True)
        
        st.info("""
        **Note**: This is a demonstration interface. To make actual predictions, 
        you would need to:
        1. Load a trained model
        2. Preprocess the input data
        3. Make predictions using the model
        """)
    
    def run_dashboard(self):
        """
        Run the complete dashboard.
        """
        # Header
        st.markdown('<h1 class="main-header">📊 Insurance Pricing Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Load data
        if not self.load_data():
            st.error("Failed to load data. Please check if the data files exist.")
            return
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a section:",
            ["📋 Dataset Overview", "🎯 Target Analysis", "🔍 Feature Analysis", 
             "🏆 Model Comparison", "🔬 MLflow Integration", "🔮 Predictions"]
        )
        
        # Display selected page
        if page == "📋 Dataset Overview":
            self.show_dataset_overview()
        
        elif page == "🎯 Target Analysis":
            self.show_target_analysis()
        
        elif page == "🔍 Feature Analysis":
            self.show_feature_analysis()
        
        elif page == "🏆 Model Comparison":
            self.show_model_comparison()
        
        elif page == "🔬 MLflow Integration":
            self.show_mlflow_integration()
        
        elif page == "🔮 Predictions":
            self.show_predictions_interface()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            Insurance Pricing Analysis Dashboard | Built with Streamlit and MLflow
        </div>
        """, unsafe_allow_html=True)

def main():
    """
    Main function to run the dashboard.
    """
    dashboard = InsuranceDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 