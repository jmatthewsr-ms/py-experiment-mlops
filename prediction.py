import pandas as pd


def predict_new_data(model, scaler, new_data):
    """Make predictions on new data points"""
    new_data_scaled = scaler.transform(new_data)

    # Predict class probabilities
    probabilities = model.predict_proba(new_data_scaled)

    # Create a DataFrame for clearer output
    results = pd.DataFrame(
        {
            "feature_1": new_data[:, 0],
            "feature_2": new_data[:, 1],
            "probability_class_1": probabilities[:, 1],
            "predicted_class": model.predict(new_data_scaled),
        }
    )

    return results
