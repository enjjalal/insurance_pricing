
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda_and_visualize(df):
    """
    Performs basic EDA and generates visualizations.
    """
    print("\n--- Exploratory Data Analysis ---")
    print("Dataset Info:")
    df.info()

    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nCorrelation Matrix:")
    print(df.corr(numeric_only=True))

    # Visualizations
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(df["charges"], kde=True)
    plt.title("Distribution of Charges")

    plt.subplot(1, 3, 2)
    sns.scatterplot(x="age", y="charges", hue="smoker", data=df)
    plt.title("Age vs. Charges (by Smoker Status)")

    plt.subplot(1, 3, 3)
    sns.boxplot(x="smoker", y="charges", data=df)
    plt.title("Charges by Smoker Status")

    plt.tight_layout()
    plt.savefig("data/eda_plots.png")
    print("EDA plots saved to data/eda_plots.png")
    # plt.show() # In a headless environment, plt.show() might not work directly

if __name__ == "__main__":
    # This part is for demonstration and assumes data_preprocessing.py has been run
    # and the 'insurance.csv' is available.
    from data_preprocessing import load_and_preprocess_data

    # Load the dataset (assuming it's in the same directory)
    X_train, X_test, y_train, y_test, df = load_and_preprocess_data("data/insurance.csv")

    if df is not None:
        perform_eda_and_visualize(df)




