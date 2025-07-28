import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class InsuranceExperimentTracker:
    def __init__(self, data_file="data/insurance_expanded.csv"):
        """
        Initialize the experiment tracker with the expanded dataset.
        """
        self.data_file = data_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
        # Set up MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("insurance_pricing_benchmark")
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the expanded dataset.
        """
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.data_file)
        print(f"Dataset size: {len(df)} records")
        
        # Handle categorical features
        le = LabelEncoder()
        df["sex"] = le.fit_transform(df["sex"])
        df["smoker"] = le.fit_transform(df["smoker"])
        
        # Create dummy variables for region
        df = pd.get_dummies(df, columns=["region"], drop_first=True)
        
        # Separate features and target
        X = df.drop("charges", axis=1)
        y = df["charges"]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def evaluate_model(self, model, model_name, X_train, X_test, y_train, y_test):
        """
        Evaluate a model and log metrics to MLflow.
        """
        with mlflow.start_run(run_name=model_name):
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Log parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_name}_model")
            
            # Print results
            print(f"\n{model_name} Results:")
            print(f"MAE: ${mae:,.2f}")
            print(f"RMSE: ${rmse:,.2f}")
            print(f"R²: {r2:.4f}")
            print(f"MAPE: {mape:.2f}%")
            
            return {
                'model_name': model_name,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'model': model
            }
    
    def run_baseline_experiments(self):
        """
        Run baseline experiments with different algorithms.
        """
        print("\n" + "="*50)
        print("RUNNING BASELINE EXPERIMENTS")
        print("="*50)
        
        results = []
        
        # 1. Linear Regression
        print("\n1. Linear Regression")
        lr = LinearRegression()
        results.append(self.evaluate_model(lr, "Linear_Regression", 
                                        self.X_train, self.X_test, 
                                        self.y_train, self.y_test))
        
        # 2. Ridge Regression
        print("\n2. Ridge Regression")
        ridge = Ridge(alpha=1.0)
        results.append(self.evaluate_model(ridge, "Ridge_Regression", 
                                        self.X_train, self.X_test, 
                                        self.y_train, self.y_test))
        
        # 3. Lasso Regression
        print("\n3. Lasso Regression")
        lasso = Lasso(alpha=0.1)
        results.append(self.evaluate_model(lasso, "Lasso_Regression", 
                                        self.X_train, self.X_test, 
                                        self.y_train, self.y_test))
        
        # 4. Random Forest
        print("\n4. Random Forest")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        results.append(self.evaluate_model(rf, "Random_Forest", 
                                        self.X_train, self.X_test, 
                                        self.y_train, self.y_test))
        
        # 5. Gradient Boosting
        print("\n5. Gradient Boosting")
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        results.append(self.evaluate_model(gb, "Gradient_Boosting", 
                                        self.X_train, self.X_test, 
                                        self.y_train, self.y_test))
        
        # 6. SVR (with scaled data)
        print("\n6. Support Vector Regression")
        svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        results.append(self.evaluate_model(svr, "SVR", 
                                        self.X_train_scaled, self.X_test_scaled, 
                                        self.y_train, self.y_test))
        
        return results
    
    def run_hyperparameter_tuning(self):
        """
        Run hyperparameter tuning for the best performing models.
        """
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # 1. Random Forest Tuning
        print("\n1. Random Forest Hyperparameter Tuning")
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_tuned = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        with mlflow.start_run(run_name="Random_Forest_Tuned"):
            rf_tuned.fit(self.X_train, self.y_train)
            mlflow.log_params(rf_tuned.best_params_)
            
            y_pred = rf_tuned.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(rf_tuned.best_estimator_, "rf_tuned_model")
            
            print(f"Best parameters: {rf_tuned.best_params_}")
            print(f"MAE: ${mae:,.2f}")
            print(f"R²: {r2:.4f}")
        
        # 2. Gradient Boosting Tuning
        print("\n2. Gradient Boosting Hyperparameter Tuning")
        gb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb_tuned = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        with mlflow.start_run(run_name="Gradient_Boosting_Tuned"):
            gb_tuned.fit(self.X_train, self.y_train)
            mlflow.log_params(gb_tuned.best_params_)
            
            y_pred = gb_tuned.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(gb_tuned.best_estimator_, "gb_tuned_model")
            
            print(f"Best parameters: {gb_tuned.best_params_}")
            print(f"MAE: ${mae:,.2f}")
            print(f"R²: {r2:.4f}")
    
    def generate_comparison_report(self, results):
        """
        Generate a comparison report of all models.
        """
        print("\n" + "="*50)
        print("MODEL COMPARISON REPORT")
        print("="*50)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df[['model_name', 'mae', 'rmse', 'r2', 'mape']]
        
        # Sort by MAE (best first)
        comparison_df = comparison_df.sort_values('mae')
        
        print("\nModel Performance Ranking (by MAE):")
        print(comparison_df.to_string(index=False))
        
        # Save comparison to file
        comparison_df.to_csv("data/model_comparison.csv", index=False)
        print(f"\nComparison saved to: data/model_comparison.csv")
        
        return comparison_df
    
    def run_all_experiments(self):
        """
        Run all experiments and generate reports.
        """
        print("Starting comprehensive model benchmarking...")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Run baseline experiments
        baseline_results = self.run_baseline_experiments()
        
        # Generate comparison report
        comparison_df = self.generate_comparison_report(baseline_results)
        
        # Run hyperparameter tuning
        self.run_hyperparameter_tuning()
        
        print("\n" + "="*50)
        print("EXPERIMENT TRACKING COMPLETED")
        print("="*50)
        print("Check MLflow UI for detailed results:")
        print("mlflow ui")
        print("\nBest performing model:")
        best_model = comparison_df.iloc[0]
        print(f"{best_model['model_name']}: MAE=${best_model['mae']:,.2f}, R²={best_model['r2']:.4f}")

if __name__ == "__main__":
    # Initialize and run experiments
    tracker = InsuranceExperimentTracker()
    tracker.run_all_experiments() 