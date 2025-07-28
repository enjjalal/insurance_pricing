import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import os

# Set page config
st.set_page_config(
    page_title="Experiment Results Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Experiment Results Dashboard")
st.markdown("---")

@st.cache_data
def load_experiment_results():
    """
    Load experiment results from MLflow and comparison CSV.
    """
    try:
        # Load model comparison results
        comparison_df = pd.read_csv("data/model_comparison.csv")
        return comparison_df
    except FileNotFoundError:
        st.error("No experiment results found. Please run the experiment tracker first.")
        return None

@st.cache_data
def get_mlflow_runs():
    """
    Get MLflow runs for detailed analysis.
    """
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("insurance_pricing_benchmark")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.mae ASC"]
            )
            return runs
        return []
    except:
        return []

def main():
    # Load results
    comparison_df = load_experiment_results()
    
    if comparison_df is None:
        st.warning("Please run the experiment tracker first:")
        st.code("python src/experiment_tracker.py")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📈 Model Comparison", "🏆 Best Model Analysis", "📊 Performance Metrics", "🔍 MLflow Integration"]
    )
    
    if page == "📈 Model Comparison":
        st.header("📈 Model Performance Comparison")
        
        # Display comparison table
        st.subheader("Model Rankings (by MAE)")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("MAE Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = plt.bar(comparison_df['model_name'], comparison_df['mae'])
            plt.title("Mean Absolute Error by Model")
            plt.ylabel("MAE ($)")
            plt.xticks(rotation=45, ha='right')
            
            # Color bars based on performance
            colors = ['green' if x == comparison_df['mae'].min() else 'lightblue' for x in comparison_df['mae']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("R² Score Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = plt.bar(comparison_df['model_name'], comparison_df['r2'])
            plt.title("R² Score by Model")
            plt.ylabel("R² Score")
            plt.xticks(rotation=45, ha='right')
            
            # Color bars based on performance
            colors = ['green' if x == comparison_df['r2'].max() else 'lightblue' for x in comparison_df['r2']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Correlation heatmap of metrics
        st.subheader("Metric Correlations")
        metric_cols = ['mae', 'rmse', 'r2', 'mape']
        correlation_matrix = comparison_df[metric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title("Correlation Between Performance Metrics")
        st.pyplot(fig)
        plt.close()
    
    elif page == "🏆 Best Model Analysis":
        st.header("🏆 Best Model Analysis")
        
        # Get best model
        best_model = comparison_df.iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Best Performing Model")
            st.metric("Model", best_model['model_name'])
            st.metric("MAE", f"${best_model['mae']:,.2f}")
            st.metric("R² Score", f"{best_model['r2']:.4f}")
            st.metric("MAPE", f"{best_model['mape']:.2f}%")
        
        with col2:
            st.subheader("Performance Summary")
            st.write(f"**Best Model:** {best_model['model_name']}")
            st.write(f"**Mean Absolute Error:** ${best_model['mae']:,.2f}")
            st.write(f"**Root Mean Square Error:** ${best_model['rmse']:,.2f}")
            st.write(f"**R² Score:** {best_model['r2']:.4f}")
            st.write(f"**Mean Absolute Percentage Error:** {best_model['mape']:.2f}%")
        
        # Performance improvement analysis
        st.subheader("Performance Improvement Analysis")
        
        # Calculate improvements
        baseline_mae = comparison_df[comparison_df['model_name'] == 'Linear_Regression']['mae'].iloc[0]
        improvements = []
        
        for _, row in comparison_df.iterrows():
            improvement = ((baseline_mae - row['mae']) / baseline_mae) * 100
            improvements.append(improvement)
        
        comparison_df['improvement_%'] = improvements
        
        # Plot improvements
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = plt.bar(comparison_df['model_name'], comparison_df['improvement_%'])
        plt.title("Improvement over Linear Regression Baseline")
        plt.ylabel("Improvement (%)")
        plt.xticks(rotation=45, ha='right')
        
        # Color bars
        colors = ['green' if x > 0 else 'red' for x in comparison_df['improvement_%']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    elif page == "📊 Performance Metrics":
        st.header("📊 Detailed Performance Metrics")
        
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("MAE vs RMSE")
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(comparison_df['mae'], comparison_df['rmse'], s=100, alpha=0.7)
            
            # Add model labels
            for i, row in comparison_df.iterrows():
                plt.annotate(row['model_name'], 
                           (row['mae'], row['rmse']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
            
            plt.xlabel('MAE ($)')
            plt.ylabel('RMSE ($)')
            plt.title('MAE vs RMSE Comparison')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("R² vs MAPE")
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(comparison_df['r2'], comparison_df['mape'], s=100, alpha=0.7)
            
            # Add model labels
            for i, row in comparison_df.iterrows():
                plt.annotate(row['model_name'], 
                           (row['r2'], row['mape']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
            
            plt.xlabel('R² Score')
            plt.ylabel('MAPE (%)')
            plt.title('R² vs MAPE Comparison')
            st.pyplot(fig)
            plt.close()
        
        # Detailed metrics table
        st.subheader("Detailed Metrics Table")
        detailed_metrics = comparison_df.copy()
        detailed_metrics['mae'] = detailed_metrics['mae'].round(2)
        detailed_metrics['rmse'] = detailed_metrics['rmse'].round(2)
        detailed_metrics['r2'] = detailed_metrics['r2'].round(4)
        detailed_metrics['mape'] = detailed_metrics['mape'].round(2)
        
        st.dataframe(detailed_metrics, use_container_width=True)
    
    elif page == "🔍 MLflow Integration":
        st.header("🔍 MLflow Integration")
        
        st.subheader("MLflow UI Access")
        st.write("To view detailed MLflow results, run the following command in your terminal:")
        st.code("mlflow ui", language="bash")
        st.write("Then open your browser to: http://localhost:5000")
        
        # Show MLflow runs if available
        runs = get_mlflow_runs()
        if runs:
            st.subheader("Recent MLflow Runs")
            
            run_data = []
            for run in runs[:10]:  # Show last 10 runs
                run_data.append({
                    'Run Name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'MAE': run.data.metrics.get('mae', 0),
                    'R²': run.data.metrics.get('r2', 0),
                    'Status': run.info.status
                })
            
            if run_data:
                runs_df = pd.DataFrame(run_data)
                st.dataframe(runs_df, use_container_width=True)
        else:
            st.info("No MLflow runs found. Run the experiment tracker to generate results.")
        
        st.subheader("Experiment Tracking Benefits")
        st.write("""
        - **Reproducibility**: All experiments are logged with exact parameters
        - **Comparison**: Easy comparison of different model configurations
        - **Versioning**: Model artifacts are versioned and stored
        - **Collaboration**: Share results with team members
        - **Monitoring**: Track model performance over time
        """)

if __name__ == "__main__":
    main() 