# Insurance Pricing Project (One-Day Scope)

This project demonstrates a basic end-to-end workflow for predicting medical insurance costs using a simple linear regression model. It is designed to be completed within a single day, focusing on core data analysis and machine learning concepts.

## Project Structure

- `src/` : All Python scripts (code files)
- `data/` : Input and output files (e.g., insurance.csv, eda_plots.png)
- `env_ins/` : Python virtual environment (do not edit)
- `__pycache__/` : Python cache files (auto-generated)

## How to Run

1.  **Clone the repository (or create the files manually):**
    ```bash
    # If you have git configured
    # git clone <repository_url>
    # cd insurance_pricing_project
    ```

2.  **Ensure you have the dataset:**
    The `insurance.csv` file should be in the `insurance_pricing_project` directory. If not, you can download it manually from the Kaggle link provided above or use the `wget` command:
    ```bash
    wget -O insurance.csv "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

4.  **Run the scripts in order:**

    *   **Data Preprocessing:**
        ```bash
        python3.11 data_preprocessing.py
        ```
        This script will load the data, preprocess it, and print the shapes of the training/testing sets and the head of the preprocessed DataFrame.

    *   **EDA and Visualization:**
        ```bash
        python3.11 eda_visualization.py
        ```
        This script will perform EDA, print descriptive statistics and correlation matrix, and save the generated plots to `eda_plots.png`.

    *   **Model Training and Evaluation:**
        ```bash
        python3.11 model_training.py
        ```
        This script will train a Linear Regression model, make predictions, and print the Mean Absolute Error (MAE) and R-squared (R2) score.

## Results

After running `model_training.py`, you should see output similar to this:

```
--- Model Training and Evaluation ---
Mean Absolute Error (MAE): [some_value]
R-squared (R2): [some_value]
Model training and evaluation complete.
```

The `eda_plots.png` file will contain visualizations showing the distribution of charges, age vs. charges by smoker status, and charges by smoker status.

## Limitations of One-Day Scope

This project is a simplified demonstration. In a real-world scenario, further steps would include:

-   More robust data cleaning and handling of outliers.
-   Advanced feature engineering.
-   Exploration of multiple machine learning models and hyperparameter tuning.
-   Cross-validation for more reliable model evaluation.
-   Deployment of the model for inference.
-   More in-depth business insights and reporting.

This project serves as a foundational piece to showcase core data analysis and machine learning skills within a constrained timeframe.

