import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_and_preprocess_data

# Set page config
st.set_page_config(
    page_title="Insurance Pricing Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Insurance Pricing Analysis Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        X_train, X_test, y_train, y_test, df = load_and_preprocess_data("data/insurance.csv")
        return X_train, X_test, y_train, y_test, df
    except:
        st.error("Error loading data. Please ensure 'data/insurance.csv' exists.")
        return None, None, None, None, None

# Load the data
X_train, X_test, y_train, y_test, df = load_data()

if df is not None:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Data Overview", "📈 Exploratory Data Analysis", "🤖 Model Training", "💰 Price Predictor"]
    )

    if page == "📊 Data Overview":
        st.header("📊 Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Total Records:** {len(df)}")
            st.write(f"**Features:** {len(df.columns)}")
            st.write("**Columns:**")
            for col in df.columns:
                st.write(f"- {col}")
        
        with col2:
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
        
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

    elif page == "📈 Exploratory Data Analysis":
        st.header("📈 Exploratory Data Analysis")
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['age'], kde=True, ax=ax)
            plt.title("Age Distribution")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("BMI Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['bmi'], kde=True, ax=ax)
            plt.title("BMI Distribution")
            st.pyplot(fig)
            plt.close()
        
        # Charges analysis
        st.subheader("Insurance Charges Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df['charges'], kde=True, ax=ax)
            plt.title("Charges Distribution")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='smoker', y='charges', ax=ax)
            plt.title("Charges by Smoker Status")
            st.pyplot(fig)
            plt.close()
        
        # Correlation heatmap
        st.subheader("Feature Correlation")
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close()
        
        # Age vs Charges by Smoker
        st.subheader("Age vs Charges (by Smoker Status)")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.6, ax=ax)
        plt.title("Age vs Charges")
        st.pyplot(fig)
        plt.close()

    elif page == "🤖 Model Training":
        st.header("🤖 Model Training & Evaluation")
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Train the model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"${mae:.2f}")
                with col2:
                    st.metric("R² Score", f"{r2:.3f}")
                with col3:
                    st.metric("RMSE", f"${rmse:.2f}")
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                fig, ax = plt.subplots()
                sns.barplot(data=feature_importance, x='Coefficient', y='Feature', ax=ax)
                plt.title("Feature Importance (Linear Regression Coefficients)")
                st.pyplot(fig)
                plt.close()
                
                # Actual vs Predicted
                st.subheader("Actual vs Predicted Values")
                fig, ax = plt.subplots()
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual Charges')
                plt.ylabel('Predicted Charges')
                plt.title('Actual vs Predicted Insurance Charges')
                st.pyplot(fig)
                plt.close()
                
                st.success("Model training completed!")

    elif page == "💰 Price Predictor":
        st.header("💰 Insurance Price Predictor")
        st.write("Enter customer information to predict insurance charges:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 30)
            sex = st.selectbox("Sex", ["male", "female"])
            bmi = st.slider("BMI", 15.0, 50.0, 25.0)
            children = st.slider("Number of Children", 0, 10, 0)
        
        with col2:
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        
        if st.button("Predict Insurance Charges"):
            # Create input data
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
                'region': [region]
            })
            
            # Preprocess input data (same as training data)
            # Sex: male/female -> 0/1
            input_data['sex'] = input_data['sex'].map({'female': 0, 'male': 1})
            
            # Smoker: yes/no -> 1/0 (yes=1, no=0)
            input_data['smoker'] = input_data['smoker'].map({'yes': 1, 'no': 0})
            
            # Region: create dummy variables (drop_first=True)
            input_data = pd.get_dummies(input_data, columns=['region'], drop_first=True)
            
            # Manually add missing region columns
            if 'region_northwest' not in input_data.columns:
                input_data['region_northwest'] = 0
            if 'region_southeast' not in input_data.columns:
                input_data['region_southeast'] = 0
            if 'region_southwest' not in input_data.columns:
                input_data['region_southwest'] = 0
            
            # Set the correct region column to 1 based on selection
            if region == 'northwest':
                input_data['region_northwest'] = 1
            elif region == 'southeast':
                input_data['region_southeast'] = 1
            elif region == 'southwest':
                input_data['region_southwest'] = 1
            # northeast is the reference category (all 0s)
            
            # Ensure all columns match training data
            for col in X_train.columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[X_train.columns]
            
            # Train model and predict
            model = LinearRegression()
            model.fit(X_train, y_train)
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f"**Predicted Insurance Charges: ${prediction:,.2f}**")
            
            # Show confidence interval (simple approximation)
            confidence_range = prediction * 0.2  # 20% margin
            st.info(f"Estimated range: ${prediction - confidence_range:,.2f} - ${prediction + confidence_range:,.2f}")

else:
    st.error("Unable to load data. Please check if 'data/insurance.csv' exists in the data folder.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Insurance Pricing Analysis Dashboard") 