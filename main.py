import numpy as np
import os
import joblib
from typing import Tuple, List, Any, Dict
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from data_preparation import generate_synthetic_data, explore_data, prepare_data
from model_training import train_model, display_model_coefficients
from model_evaluation import evaluate_model
from prediction import predict_new_data


def main() -> None:
    # Create output directory for model artifacts
    os.makedirs("model_artifacts", exist_ok=True)

    # Interview Questions - Data Generation & Exploration:
    # Q1: How would you ensure data quality and handle missing values in a production environment?
    # Q2: What monitoring systems would you put in place to detect data drift over time?
    print("1. Data Generation & Exploration")
    # Generate synthetic data
    df: pd.DataFrame = generate_synthetic_data()

    # Explore data
    explore_data(df)

    # Interview Questions - Data Preparation:
    # Q1: How would you ensure consistent data preprocessing across training and inference pipelines?
    # Q2: What strategies would you employ to handle outliers or unexpected data distributions in production?
    print("\n2. Data Preparation")
    # Prepare data for modeling
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)

    # Interview Questions - Model Training:
    # Q1: How would you implement versioning for your models to track changes and enable rollbacks if needed?
    # Q2: What considerations would you make for retraining frequency and triggering model updates in production?
    print("\n3. Model Training")
    # Train the model
    model: BaseEstimator = train_model(X_train_scaled, y_train)

    # Display model coefficients
    display_model_coefficients(model, ["feature_1", "feature_2"])

    # Interview Questions - Model Evaluation:
    # Q1: What metrics beyond accuracy would you monitor in production to ensure model performance?
    # Q2: How would you implement A/B testing to safely deploy new model versions?
    print("\n4. Model Evaluation")
    # Evaluate model performance
    y_pred: np.ndarray
    accuracy: float
    y_pred, accuracy = evaluate_model(model, X_test_scaled, y_test)

    # Interview Questions - Making Predictions:
    # Q1: How would you design your prediction service to handle varying loads and ensure high availability?
    # Q2: What logging and observability practices would you implement for production inference?
    print("\n5. Making Predictions")
    # Create new data points for demonstration
    new_data: np.ndarray = np.array([[0.5, 1.0], [-1.0, 2.5], [0.0, 0.0]])

    # Make predictions
    results: Dict[str, Any] = predict_new_data(model, scaler, new_data)
    print("Predictions for new data points:")
    print(results)

    # Interview Questions - Model Persistence:
    # Q1: What security considerations should be addressed when storing and loading models in production?
    # Q2: How would you implement model artifact management to ensure reproducibility and traceability?
    # Save model and scaler for future use
    joblib.dump(model, "model_artifacts/logistic_regression_model.pkl")
    joblib.dump(scaler, "model_artifacts/feature_scaler.pkl")
    print("\nModel and scaler saved to 'model_artifacts' directory")


if __name__ == "__main__":
    main()
