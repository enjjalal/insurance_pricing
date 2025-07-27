
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import shutil

def load_and_preprocess_data(file_path):
    """
    Loads the insurance dataset and performs basic preprocessing.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None

    # Handle categorical features
    # Sex: male/female
    le = LabelEncoder()
    df["sex"] = le.fit_transform(df["sex"])

    # Smoker: yes/no
    df["smoker"] = le.fit_transform(df["smoker"])

    # Region: northeast, northwest, southeast, southwest
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # Define features (X) and target (y)
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, df

def process_incoming_data():
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from eda_visualization import perform_eda_and_visualize
    from model_training import train_and_evaluate_model

    incoming_dir = "data/incoming/"
    processed_dir = "data/processed/"
    for filename in os.listdir(incoming_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(incoming_dir, filename)
            print(f"Processing {file_path}...")
            X_train, X_test, y_train, y_test, df = load_and_preprocess_data(file_path)
            if X_train is not None and df is not None:
                print(f"Processed {filename}: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
                # Perform EDA and visualization
                perform_eda_and_visualize(df)
                # Train and evaluate the model
                train_and_evaluate_model(X_train, X_test, y_train, y_test)
            # Move file to processed folder
            shutil.move(file_path, os.path.join(processed_dir, filename))
            print(f"Moved {filename} to {processed_dir}")

if __name__ == "__main__":
    # This part is for demonstration and can be removed in a modular setup
    # In a real scenario, the dataset would be downloaded first.
    # For this one-day scope, we assume the CSV is available.
    # You would typically download it from Kaggle and place it in the project directory.
    # Example: https://www.kaggle.com/datasets/mirichoi0218/insurance
    
    # Placeholder for file path - user needs to download the CSV
    # For demonstration, let's create a dummy CSV if it doesn't exist
    try:
        pd.read_csv("data/insurance.csv")
    except FileNotFoundError:
        print("Creating a dummy data/insurance.csv for demonstration. Please replace with actual data.")
        dummy_data = {
            'age': [19, 18, 28, 33, 32],
            'sex': ['female', 'male', 'male', 'male', 'female'],
            'bmi': [27.9, 33.77, 33.0, 22.705, 28.88],
            'children': [0, 1, 3, 0, 0],
            'smoker': ['yes', 'no', 'no', 'no', 'no'],
            'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest'],
            'charges': [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv("data/insurance.csv", index=False)

    X_train, X_test, y_train, y_test, df = load_and_preprocess_data("data/insurance.csv")

    if X_train is not None:
        print("Data loaded and preprocessed successfully!")
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("First 5 rows of preprocessed data:")
        print(df.head())

    print("\n--- Automated Pipeline: Processing incoming data files ---")
    process_incoming_data()




