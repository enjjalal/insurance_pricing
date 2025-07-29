import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class InsuranceModelTrainer:
    def __init__(self, data_file="data/insurance_expanded.csv"):
        """
        Initialize the model trainer with the insurance dataset.
        """
        self.data_file = data_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the insurance dataset.
        """
        print("Loading and preprocessing data...")
        
        try:
            # Load data
            self.df = pd.read_csv(self.data_file)
            print(f"Dataset loaded: {len(self.df)} records")
            
            # Handle categorical features
            le = LabelEncoder()
            self.df["sex"] = le.fit_transform(self.df["sex"])
            self.df["smoker"] = le.fit_transform(self.df["smoker"])
            
            # Create dummy variables for region
            self.df = pd.get_dummies(self.df, columns=["region"], drop_first=True)
            
            # Separate features and target
            X = self.df.drop("charges", axis=1)
            y = self.df["charges"]
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print(f"Training set: {self.X_train.shape[0]} samples")
            print(f"Test set: {self.X_test.shape[0]} samples")
            print(f"Features: {self.X_train.shape[1]}")
            
            return True
            
        except Exception as e:
            print(f"Error in data preprocessing: {e}")
            return False
    
    def train_linear_regression(self):
        """
        Train a Linear Regression model.
        """
        print("\n" + "="*50)
        print("TRAINING LINEAR REGRESSION")
        print("="*50)
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        # Store results
        self.models['linear_regression'] = model
        self.results['linear_regression'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return model
    
    def train_ridge_regression(self, alpha=1.0):
        """
        Train a Ridge Regression model.
        """
        print("\n" + "="*50)
        print("TRAINING RIDGE REGRESSION")
        print("="*50)
        
        model = Ridge(alpha=alpha)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        # Store results
        self.models['ridge_regression'] = model
        self.results['ridge_regression'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'alpha': alpha
        }
        
        print(f"Alpha: {alpha}")
        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return model
    
    def train_random_forest(self, n_estimators=100, max_depth=None):
        """
        Train a Random Forest model.
        """
        print("\n" + "="*50)
        print("TRAINING RANDOM FOREST")
        print("="*50)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        # Store results
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        
        print(f"n_estimators: {n_estimators}")
        print(f"max_depth: {max_depth}")
        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return model
    
    def train_gradient_boosting(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Train a Gradient Boosting model.
        """
        print("\n" + "="*50)
        print("TRAINING GRADIENT BOOSTING")
        print("="*50)
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        # Store results
        self.models['gradient_boosting'] = model
        self.results['gradient_boosting'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth
        }
        
        print(f"n_estimators: {n_estimators}")
        print(f"learning_rate: {learning_rate}")
        print(f"max_depth: {max_depth}")
        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return model
    
    def hyperparameter_tuning(self, model_name='random_forest'):
        """
        Perform hyperparameter tuning for a specific model.
        """
        print(f"\n" + "="*50)
        print(f"HYPERPARAMETER TUNING - {model_name.upper()}")
        print("="*50)
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
            
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
            
        elif model_name == 'ridge':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
            model = Ridge()
            
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Evaluate best model
        y_pred = best_model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"Best parameters: {best_params}")
        print(f"Best MAE: ${mae:,.2f}")
        print(f"Best R²: {r2:.4f}")
        
        # Store tuned model
        self.models[f'{model_name}_tuned'] = best_model
        self.results[f'{model_name}_tuned'] = {
            'mae': mae,
            'r2': r2,
            'best_params': best_params
        }
        
        return best_model
    
    def plot_predictions(self, model_name):
        """
        Plot actual vs predicted values for a specific model.
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Charges ($)')
        plt.ylabel('Predicted Charges ($)')
        plt.title(f'Actual vs Predicted - {model_name.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(self.y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"data/{model_name}_predictions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, model_name):
        """
        Plot residuals for a specific model.
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        residuals = self.y_test - y_pred
        
        plt.figure(figsize=(15, 5))
        
        # Residuals vs Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Charges ($)')
        plt.ylabel('Residuals ($)')
        plt.title('Residuals vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Residuals histogram
        plt.subplot(1, 3, 2)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals ($)')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        # Q-Q plot of residuals
        plt.subplot(1, 3, 3)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.savefig(f"data/{model_name}_residuals.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_name, filepath=None):
        """
        Save a trained model to disk.
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return False
        
        if filepath is None:
            filepath = f"models/{model_name}_model.pkl"
        
        # Create models directory if it doesn't exist
        import os
        os.makedirs("models", exist_ok=True)
        
        # Save model
        joblib.dump(self.models[model_name], filepath)
        print(f"Model saved to: {filepath}")
        
        return True
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        """
        try:
            model = joblib.load(filepath)
            print(f"Model loaded from: {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def generate_comparison_report(self):
        """
        Generate a comparison report of all trained models.
        """
        if not self.results:
            print("No models have been trained yet.")
            return
        
        print("\n" + "="*50)
        print("MODEL COMPARISON REPORT")
        print("="*50)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'MAE': results['mae'],
                'RMSE': results.get('rmse', np.nan),
                'R²': results['r2'],
                'MAPE': results.get('mape', np.nan)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAE')
        
        print("\nModel Performance Ranking (by MAE):")
        print(comparison_df.to_string(index=False))
        
        # Save comparison to file
        comparison_df.to_csv("data/individual_model_comparison.csv", index=False)
        print(f"\nComparison saved to: data/individual_model_comparison.csv")
        
        return comparison_df
    
    def run_complete_training(self):
        """
        Run the complete model training pipeline.
        """
        print("="*50)
        print("INDIVIDUAL MODEL TRAINING")
        print("="*50)
        
        # Load and preprocess data
        if not self.load_and_preprocess_data():
            return False
        
        # Train different models
        self.train_linear_regression()
        self.train_ridge_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        
        # Hyperparameter tuning
        self.hyperparameter_tuning('random_forest')
        self.hyperparameter_tuning('gradient_boosting')
        
        # Generate comparison report
        comparison_df = self.generate_comparison_report()
        
        # Plot predictions for best model
        best_model = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
        self.plot_predictions(best_model)
        self.plot_residuals(best_model)
        
        # Save best model
        self.save_model(best_model)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        print(f"Best model: {comparison_df.iloc[0]['Model']}")
        print(f"Best MAE: ${comparison_df.iloc[0]['MAE']:,.2f}")
        print(f"Best R²: {comparison_df.iloc[0]['R²']:.4f}")
        
        return True

def main():
    """
    Main function to run the model training pipeline.
    """
    trainer = InsuranceModelTrainer()
    success = trainer.run_complete_training()
    
    if success:
        print("\nIndividual model training completed successfully!")
        print("You can now use the trained models for predictions.")
    else:
        print("\nModel training failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 