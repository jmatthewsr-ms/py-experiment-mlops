import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(n_samples=500, random_seed=42):
    """Generate synthetic dataset for demonstration"""
    np.random.seed(random_seed)
    df = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(2, 1.5, n_samples),
            "target": np.random.binomial(1, 0.3, n_samples),
        }
    )
    return df


def explore_data(df):
    """Print data exploration summary"""
    print(f"Dataset shape: {df.shape}")
    print("\nData summary:")
    print(df.describe())
    print("\nSample data:")
    print(df.head())


def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare features and target, split data and scale features"""
    X = df[["feature_1", "feature_2"]]
    y = df["target"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
