from sklearn.linear_model import LogisticRegression


def train_model(X_train_scaled, y_train, random_state=42, **model_params):
    """Train logistic regression model"""
    model = LogisticRegression(random_state=random_state, **model_params)
    model.fit(X_train_scaled, y_train)
    return model


def display_model_coefficients(model, feature_names):
    """Display model coefficients"""
    print("Model coefficients:")
    for feature, coef in zip(feature_names, model.coef_[0]):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")
