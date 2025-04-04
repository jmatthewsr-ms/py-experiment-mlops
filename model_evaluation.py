from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print classification report with zero_division parameter to handle the warning
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return y_pred, accuracy
