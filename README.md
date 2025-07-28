# Insurance Pricing Project - Experiment Tracking & Benchmarking

This project demonstrates comprehensive machine learning model benchmarking and experiment tracking for insurance pricing prediction.

## Features

### 🧪 Experiment Tracking
- **MLflow Integration**: Complete experiment tracking with parameter logging, metric tracking, and model versioning
- **Reproducible Results**: All experiments are logged with exact parameters and data splits
- **Model Comparison**: Systematic benchmarking of 8 different algorithms

### 📊 Model Benchmarking
- **Linear Models**: Linear Regression, Ridge, Lasso
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Advanced Models**: XGBoost, LightGBM, Support Vector Regression
- **Hyperparameter Tuning**: Grid search optimization for best performing models

### 📈 Performance Metrics
- **MAE**: Mean Absolute Error (primary metric)
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Experiment Tracking
```bash
python src/experiment_tracker_simple.py
```

### 3. View Results
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

## Project Structure

```
insurance_pricing/
├── src/
│   ├── experiment_tracker_simple.py    # Main experiment tracking script
│   └── experiment_dashboard.py         # Streamlit dashboard (optional)
├── data/
│   ├── insurance_expanded.csv          # Dataset
│   └── model_comparison.csv           # Generated comparison results
├── mlruns/                            # MLflow experiment tracking (auto-generated)
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Results Summary

### Best Performing Models (by MAE):
1. **Gradient Boosting**: MAE = $1,334.15, R² = 0.9144
2. **LightGBM**: MAE = $1,357.73, R² = 0.9100
3. **XGBoost**: MAE = $1,405.86, R² = 0.9001

### Key Insights:
- Ensemble methods significantly outperform linear models
- Gradient Boosting shows the best performance for insurance pricing
- Hyperparameter tuning improves model performance by ~5-10%

## Experiment Tracking Benefits

- **Reproducibility**: All experiments logged with exact parameters
- **Comparison**: Easy side-by-side model comparison
- **Versioning**: Model artifacts are versioned and stored
- **Collaboration**: Share results with team members
- **Monitoring**: Track model performance over time

## Usage Examples

### Run Basic Experiments
```python
from src.experiment_tracker_simple import InsuranceExperimentTracker

tracker = InsuranceExperimentTracker()
tracker.run_all_experiments()
```

### Access MLflow Results
```python
import mlflow

# View all experiments
mlflow.search_experiments()

# Load a specific model
loaded_model = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

## Version History

- **v1.2.0**: Complete experiment tracking and benchmarking system
- **v1.1.0**: Basic ML pipeline with data preprocessing
- **v1.0.0**: Initial project setup

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your experiments
4. Submit a pull request

## License

This project is licensed under the MIT License. 