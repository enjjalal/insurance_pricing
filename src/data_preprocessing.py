import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class InsuranceDataPreprocessor:
    def __init__(self, data_file="data/insurance_expanded.csv"):
        """
        Initialize the data preprocessor with the insurance dataset.
        """
        self.data_file = data_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """
        Load the insurance dataset.
        """
        print("Loading insurance dataset...")
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Dataset loaded successfully: {len(self.df)} records")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.data_file} not found.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """
        Perform basic data exploration.
        """
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")
        
        print("\nNumerical Features Summary:")
        print(self.df.describe())
        
        print("\nCategorical Features:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"{col}: {self.df[col].nunique()} unique values")
            print(self.df[col].value_counts().head())
            print()
    
    def handle_categorical_features(self):
        """
        Encode categorical features.
        """
        print("Handling categorical features...")
        
        # Encode binary categorical features
        binary_categorical = ['sex', 'smoker']
        for col in binary_categorical:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"Encoded {col}: {le.classes_}")
        
        # Create dummy variables for multi-category features
        multi_categorical = ['region']
        for col in multi_categorical:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
                print(f"Created dummy variables for {col}")
        
        print("Categorical feature encoding completed.")
    
    def check_outliers(self, method='iqr'):
        """
        Check for outliers in numerical features.
        """
        print("\nChecking for outliers...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'charges']  # Exclude target
        
        outliers_info = {}
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outliers_info[col] = len(outliers)
            
            if len(outliers) > 0:
                print(f"{col}: {len(outliers)} outliers detected ({len(outliers)/len(self.df)*100:.2f}%)")
        
        return outliers_info
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        """
        print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
        
        # Separate features and target
        X = self.df.drop('charges', axis=1)
        y = self.df['charges']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"Test set: {self.X_test.shape[0]} samples, {self.X_test.shape[1]} features")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """
        Scale numerical features.
        """
        print("Scaling numerical features...")
        
        # Get numerical columns (exclude target)
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        
        # Scale training data
        self.X_train_scaled = self.X_train.copy()
        self.X_train_scaled[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
        
        # Scale test data
        self.X_test_scaled = self.X_test.copy()
        self.X_test_scaled[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        
        print("Feature scaling completed.")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def save_processed_data(self, output_dir="data/processed"):
        """
        Save processed data to files.
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training data
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        train_data.to_csv(f"{output_dir}/train_data.csv", index=False)
        
        # Save test data
        test_data = pd.concat([self.X_test, self.y_test], axis=1)
        test_data.to_csv(f"{output_dir}/test_data.csv", index=False)
        
        # Save scaled training data
        train_scaled_data = pd.concat([self.X_train_scaled, self.y_train], axis=1)
        train_scaled_data.to_csv(f"{output_dir}/train_data_scaled.csv", index=False)
        
        # Save scaled test data
        test_scaled_data = pd.concat([self.X_test_scaled, self.y_test], axis=1)
        test_scaled_data.to_csv(f"{output_dir}/test_data_scaled.csv", index=False)
        
        print(f"Processed data saved to {output_dir}/")
    
    def get_preprocessing_summary(self):
        """
        Get a summary of the preprocessing steps.
        """
        summary = {
            'original_shape': self.df.shape,
            'features_after_encoding': self.X_train.shape[1],
            'training_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0],
            'categorical_features_encoded': list(self.label_encoders.keys()),
            'numerical_features_scaled': self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        return summary
    
    def run_complete_preprocessing(self):
        """
        Run the complete preprocessing pipeline.
        """
        print("="*50)
        print("DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        # Load data
        if not self.load_data():
            return False
        
        # Explore data
        self.explore_data()
        
        # Handle categorical features
        self.handle_categorical_features()
        
        # Check outliers
        outliers_info = self.check_outliers()
        
        # Split data
        self.split_data()
        
        # Scale features
        self.scale_features()
        
        # Save processed data
        self.save_processed_data()
        
        # Print summary
        summary = self.get_preprocessing_summary()
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("\nPreprocessing completed successfully!")
        return True

def main():
    """
    Main function to run the preprocessing pipeline.
    """
    preprocessor = InsuranceDataPreprocessor()
    success = preprocessor.run_complete_preprocessing()
    
    if success:
        print("\nData preprocessing completed successfully!")
        print("You can now run the experiment tracker or individual model training.")
    else:
        print("\nData preprocessing failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 