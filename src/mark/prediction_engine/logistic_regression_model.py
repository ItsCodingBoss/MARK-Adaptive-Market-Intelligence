from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics import accuracy_score


from mark.feature_engine.feature_engineering import (
    feature_columns,
    prepare_model_data,
)
def train_logistic_regression(train_data, val_data):
    x_train = train_data[feature_columns]
    y_train = train_data["target_up"]
    x_val = val_data[feature_columns]
    y_val = val_data["target_up"]

    model = Pipeline([
        ("scalar", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
    model.fit(x_train, y_train)
    prediction = model.predict(x_val)
    accuracy = accuracy_score(y_val, prediction)
    return {
        "model_name": "logistic_regression",
        "model": model,
        "validation_accuracy": float(accuracy),
        "predictions": prediction,
        "actual": y_val.to_numpy(),
        "dates": val_data["Date"].to_numpy(),
    }
if __name__ == "__main__":
    train_data, val_data, test_data = prepare_model_data()
    results = train_logistic_regression(train_data, val_data)
    print("\n--- Logistic Regression Model ---")
    print(f"Validation accuracy: {results['validation_accuracy']:.2%}")
