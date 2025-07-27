
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from data_preprocessing import load_and_preprocess_data

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Trains a Linear Regression model and evaluates its performance.
    """
    print("\n--- Model Training and Evaluation ---")

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    return model, mae, r2

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data("data/insurance.csv")

    if X_train is not None:
        # Train and evaluate the model
        model, mae, r2 = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        print("Model training and evaluation complete.")



