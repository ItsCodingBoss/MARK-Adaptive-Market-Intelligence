import numpy as np
from mark.prediction_engine.model_evaluation import evaluate_classifier


def evaluate_thresholds(model, X, actual, thresholds=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)):
    """
    Evaluate the model's performance at different thresholds.

    Parameters:
    - model: The trained model to evaluate.
    - X: The input features for prediction.
    - actual: The actual target values (binary).
    - thresholds: A tuple of thresholds to evaluate.

    Returns:
    A dictionary containing evaluation metrics for each threshold.
    """
    probabilities = model.predict_proba(X)
    classes =  model.named_steps['classifier'].classes_

    positive_class_index = np.flatnonzero(classes == 1)

    if len(positive_class_index) != 1:
        raise ValueError("Model must support binary classification with class 1.")

    probabilities_up = probabilities[:,positive_class_index[0]]
    results = []
    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise ValueError("Thresholds must be between 0 and 1.")
        predictions  = (probabilities_up >= threshold).astype(int)
        metrics = evaluate_classifier(actual, predictions)
        results.append({
                "threshold": float(threshold),
                "predicted_up_pct": float(predictions.mean() * 100),
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": metrics["confusion_matrix"],
        })
    return results
