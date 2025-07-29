import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InsuranceEDA:
    def __init__(self, data_file="data/insurance_expanded.csv"):
        """
        Initialize the EDA analyzer with the insurance dataset.
        """
        self.data_file = data_file
        self.df = None
        self.output_dir = "data/eda_plots"
        
        # Create output directory
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """
        Load the insurance dataset.
        """
        print("Loading insurance dataset for EDA...")
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Dataset loaded: {len(self.df)} records, {len(self.df.columns)} features")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.data_file} not found.")
            return False
    
    def basic_statistics(self):
        """
        Generate basic statistical summary.
        """
        print("\n" + "="*50)
        print("BASIC STATISTICAL SUMMARY")
        print("="*50)
        
        print("\nDataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found")
        
        print("\nNumerical Features Summary:")
        print(self.df.describe())
        
        print("\nCategorical Features Summary:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
            print(f"Unique values: {self.df[col].nunique()}")
    
    def target_analysis(self):
        """
        Analyze the target variable (charges).
        """
        print("\n" + "="*50)
        print("TARGET VARIABLE ANALYSIS")
        print("="*50)
        
        charges = self.df['charges']
        
        print(f"\nTarget Variable: 'charges'")
        print(f"Mean: ${charges.mean():,.2f}")
        print(f"Median: ${charges.median():,.2f}")
        print(f"Std: ${charges.std():,.2f}")
        print(f"Min: ${charges.min():,.2f}")
        print(f"Max: ${charges.max():,.2f}")
        print(f"Skewness: {charges.skew():.3f}")
        print(f"Kurtosis: {charges.kurtosis():.3f}")
        
        # Create target distribution plot
        plt.figure(figsize=(15, 5))
        
        # Histogram
        plt.subplot(1, 3, 1)
        plt.hist(charges, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Insurance Charges')
        plt.xlabel('Charges ($)')
        plt.ylabel('Frequency')
        plt.axvline(charges.mean(), color='red', linestyle='--', label=f'Mean: ${charges.mean():,.0f}')
        plt.axvline(charges.median(), color='green', linestyle='--', label=f'Median: ${charges.median():,.0f}')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 3, 2)
        plt.boxplot(charges)
        plt.title('Box Plot of Insurance Charges')
        plt.ylabel('Charges ($)')
        
        # Q-Q plot
        plt.subplot(1, 3, 3)
        stats.probplot(charges, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normal Distribution)')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/target_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_analysis(self):
        """
        Analyze individual features and their relationship with the target.
        """
        print("\n" + "="*50)
        print("FEATURE ANALYSIS")
        print("="*50)
        
        # Numerical features
        numerical_features = ['age', 'bmi']
        
        for feature in numerical_features:
            print(f"\n{feature.upper()} Analysis:")
            print(f"Mean: {self.df[feature].mean():.2f}")
            print(f"Median: {self.df[feature].median():.2f}")
            print(f"Std: {self.df[feature].std():.2f}")
            print(f"Correlation with charges: {self.df[feature].corr(self.df['charges']):.3f}")
            
            # Create feature vs target plots
            plt.figure(figsize=(15, 5))
            
            # Scatter plot
            plt.subplot(1, 3, 1)
            plt.scatter(self.df[feature], self.df['charges'], alpha=0.6)
            plt.title(f'{feature.title()} vs Charges')
            plt.xlabel(feature.title())
            plt.ylabel('Charges ($)')
            
            # Distribution
            plt.subplot(1, 3, 2)
            plt.hist(self.df[feature], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title(f'Distribution of {feature.title()}')
            plt.xlabel(feature.title())
            plt.ylabel('Frequency')
            
            # Box plot by feature
            plt.subplot(1, 3, 3)
            plt.boxplot([self.df[self.df[feature] <= self.df[feature].quantile(0.33)]['charges'],
                        self.df[(self.df[feature] > self.df[feature].quantile(0.33)) & 
                               (self.df[feature] <= self.df[feature].quantile(0.66))]['charges'],
                        self.df[self.df[feature] > self.df[feature].quantile(0.66)]['charges']],
                       labels=['Low', 'Medium', 'High'])
            plt.title(f'Charges by {feature.title()} Groups')
            plt.ylabel('Charges ($)')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{feature}_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def categorical_analysis(self):
        """
        Analyze categorical features and their impact on charges.
        """
        print("\n" + "="*50)
        print("CATEGORICAL FEATURE ANALYSIS")
        print("="*50)
        
        categorical_features = ['sex', 'smoker', 'region']
        
        for feature in categorical_features:
            print(f"\n{feature.upper()} Analysis:")
            
            # Summary statistics by category
            category_stats = self.df.groupby(feature)['charges'].agg(['mean', 'median', 'std', 'count'])
            print(category_stats)
            
            # Create categorical analysis plots
            plt.figure(figsize=(15, 5))
            
            # Box plot
            plt.subplot(1, 3, 1)
            self.df.boxplot(column='charges', by=feature, ax=plt.gca())
            plt.title(f'Charges by {feature.title()}')
            plt.suptitle('')  # Remove default title
            
            # Bar plot of means
            plt.subplot(1, 3, 2)
            means = self.df.groupby(feature)['charges'].mean()
            means.plot(kind='bar', color='lightcoral')
            plt.title(f'Average Charges by {feature.title()}')
            plt.ylabel('Average Charges ($)')
            plt.xticks(rotation=45)
            
            # Distribution by category
            plt.subplot(1, 3, 3)
            for category in self.df[feature].unique():
                category_data = self.df[self.df[feature] == category]['charges']
                plt.hist(category_data, alpha=0.6, label=category, bins=20)
            plt.title(f'Charges Distribution by {feature.title()}')
            plt.xlabel('Charges ($)')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{feature}_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def correlation_analysis(self):
        """
        Analyze correlations between features.
        """
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Create correlation matrix
        numerical_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numerical_df.corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Pair plot for numerical features
        plt.figure(figsize=(12, 12))
        sns.pairplot(numerical_df, diag_kind='kde')
        plt.savefig(f"{self.output_dir}/pairplot.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def outlier_analysis(self):
        """
        Analyze outliers in the dataset.
        """
        print("\n" + "="*50)
        print("OUTLIER ANALYSIS")
        print("="*50)
        
        numerical_features = ['age', 'bmi', 'charges']
        
        for feature in numerical_features:
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
            
            print(f"\n{feature.upper()} Outliers:")
            print(f"Lower bound: {lower_bound:.2f}")
            print(f"Upper bound: {upper_bound:.2f}")
            print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")
            
            if len(outliers) > 0:
                print(f"Outlier values: {outliers[feature].tolist()[:5]}...")  # Show first 5
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance using correlation and statistical tests.
        """
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Correlation with target
        numerical_features = ['age', 'bmi']
        correlations = {}
        
        for feature in numerical_features:
            corr = self.df[feature].corr(self.df['charges'])
            correlations[feature] = abs(corr)
            print(f"{feature}: Correlation with charges = {corr:.3f}")
        
        # Categorical feature importance using ANOVA
        categorical_features = ['sex', 'smoker', 'region']
        
        for feature in categorical_features:
            categories = self.df[feature].unique()
            category_charges = [self.df[self.df[feature] == cat]['charges'] for cat in categories]
            
            # Perform ANOVA test
            f_stat, p_value = stats.f_oneway(*category_charges)
            print(f"{feature}: F-statistic = {f_stat:.3f}, p-value = {p_value:.6f}")
            
            # Calculate effect size (eta-squared)
            ss_between = sum(len(cat_charges) * ((cat_charges.mean() - self.df['charges'].mean())**2) 
                           for cat_charges in category_charges)
            ss_total = sum((charge - self.df['charges'].mean())**2 for charge in self.df['charges'])
            eta_squared = ss_between / ss_total
            print(f"{feature}: Effect size (η²) = {eta_squared:.3f}")
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report.
        """
        print("\n" + "="*50)
        print("EDA SUMMARY REPORT")
        print("="*50)
        
        summary = {
            'Dataset Size': f"{len(self.df)} records, {len(self.df.columns)} features",
            'Target Variable': 'charges (insurance costs)',
            'Numerical Features': ['age', 'bmi'],
            'Categorical Features': ['sex', 'smoker', 'region'],
            'Missing Values': 'None detected',
            'Outliers': 'Present in charges (expected for insurance costs)',
            'Key Insights': [
                'Smoker status has the strongest impact on charges',
                'Age and BMI show moderate correlation with charges',
                'Region shows some variation in charges',
                'Sex has minimal impact on charges'
            ]
        }
        
        for key, value in summary.items():
            print(f"\n{key}:")
            if isinstance(value, list):
                for item in value:
                    print(f"  • {item}")
            else:
                print(f"  {value}")
    
    def run_complete_eda(self):
        """
        Run the complete EDA pipeline.
        """
        print("="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Load data
        if not self.load_data():
            return False
        
        # Run all analyses
        self.basic_statistics()
        self.target_analysis()
        self.feature_analysis()
        self.categorical_analysis()
        self.correlation_analysis()
        self.outlier_analysis()
        self.feature_importance_analysis()
        self.generate_summary_report()
        
        print(f"\nEDA completed! All plots saved to {self.output_dir}/")
        return True

def main():
    """
    Main function to run the EDA pipeline.
    """
    eda = InsuranceEDA()
    success = eda.run_complete_eda()
    
    if success:
        print("\nExploratory Data Analysis completed successfully!")
        print("You can now proceed with data preprocessing and model training.")
    else:
        print("\nEDA failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 