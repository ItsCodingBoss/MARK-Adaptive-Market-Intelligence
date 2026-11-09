from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics import accuracy_score
from mark.prediction_engine.model_evaluation import evaluate_classifier, make_class_frequency_baseline,evaluate_probability_predictions
from mark.prediction_engine.threshold_analysis import evaluate_thresholds


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
    metrics = evaluate_classifier(
    results["actual"],
    results["predictions"],
)

    print("\n--- Logistic Regression Evaluation ---")

    for metric, value in metrics.items():
        if metric == "confusion_matrix":
            print(f"{metric}:")
            for label, count in value.items():
                print(f"  {label}: {count}")
        elif metric in {"samples"}:
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.2%}")
    X_val = val_data[feature_columns]

    threshold_results = evaluate_thresholds(
        model=results["model"],
        X=X_val,
        actual=results["actual"],
    )

    print("\n--- Threshold Experiment ---")
    print(
        f"{'Threshold':>9} {'Accuracy':>10} {'Balanced':>10} "
        f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'Predicted Up':>13}"
    )

    for result in threshold_results:
        print(
            f"{result['threshold']:>9.2f} "
            f"{result['accuracy']:>9.2%} "
            f"{result['balanced_accuracy']:>9.2%} "
            f"{result['precision']:>9.2%} "
            f"{result['recall']:>9.2%} "
            f"{result['f1_score']:>9.2%} "
            f"{result['predicted_up_pct']:>12.2f}%"
        )  
    
        import numpy as np

        probabilities_up = results["model"].predict_proba(
            val_data[feature_columns]
        )[:, 1]

        print("\n--- Predicted Probability Distribution ---")
        print(f"Minimum: {probabilities_up.min():.4f}")
        print(f"25th percentile: {np.percentile(probabilities_up, 25):.4f}")
        print(f"Median: {np.median(probabilities_up):.4f}")
        print(f"75th percentile: {np.percentile(probabilities_up, 75):.4f}")
        print(f"Maximum: {probabilities_up.max():.4f}")

        print(
            "Between 0.50 and 0.55:",
            int(((probabilities_up >= 0.50) & (probabilities_up < 0.55)).sum())
        )

        # --- Probability Analysis by Actual Market Outcome ---


        # Use the already-trained model and validation features.
        X_val = val_data[feature_columns]
        model = results["model"]

        # Get the probability predictions without retraining the model.
        probabilities = model.predict_proba(X_val)

        # Find the column corresponding to class 1 (UP).
        # This avoids assuming that the UP class is always in column 1.
        classes = model.named_steps["classifier"].classes_
        positive_index = np.flatnonzero(classes == 1)

        if len(positive_index) != 1:
            raise ValueError("Could not uniquely identify the upward class.")

        probabilities_up = probabilities[:, positive_index[0]]

        # Actual validation outcomes: 0 = DOWN/FLAT, 1 = UP.
        actual = np.asarray(results["actual"], dtype=int)

        if len(probabilities_up) != len(actual):
            raise ValueError(
                "The number of probabilities and actual outcomes does not match."
            )

        print("\n--- Probabilities by Actual Outcome ---")

        for label, description in [
            (0, "Actual DOWN / FLAT"),
            (1, "Actual UP"),
        ]:
            # Select predicted UP probabilities for this actual outcome.
            group = probabilities_up[actual == label]

            if len(group) == 0:
                print(f"\n{description}: No observations found.")
                continue

            print(f"\n{description}")
            print("Observations:", len(group))
            print(f"Mean probability:   {group.mean():.4f}")
            print(f"Median probability: {np.median(group):.4f}")
            print(f"25th percentile:    {np.percentile(group, 25):.4f}")
            print(f"75th percentile:    {np.percentile(group, 75):.4f}")

        y_val = results["actual"]

        model = results["model"]
        X_val = val_data[feature_columns]

        model_probabilities = model.predict_proba(X_val)
        classes = model.classes_
        training_targets = train_data["target_up"].astype(int).to_numpy()
        baseline_probabilities = make_class_frequency_baseline(
            training_targets = training_targets,
            n_predictions = len(y_val),
            classes = classes)

        model_metrics = evaluate_probability_predictions(
            actual=y_val,
            probabilities=model_probabilities,
            classes=classes,
            positive_label=1,
            )
        baseline_metrics = evaluate_probability_predictions(
            actual=y_val,
            probabilities=baseline_probabilities,
            classes=classes,
            positive_label=1,
            )
        print("\n--- Probability Quality Evaluation ---")

        print(
            f"{'Metric':<18}"
            f"{'Logistic Regression':>23}"
            f"{'Probability Baseline':>23}"
        )

        print(
            f"{'Brier Score':<18}"
            f"{model_metrics['brier_score']:>23.5f}"
            f"{baseline_metrics['brier_score']:>23.5f}"
        )

        print(
            f"{'Log Loss':<18}"
            f"{model_metrics['log_loss']:>23.5f}"
            f"{baseline_metrics['log_loss']:>23.5f}"
        )

        print(f"\nValidation samples: {model_metrics['samples']}")
        print(
            f"Training UP frequency: "
            f"{np.mean(training_targets == 1):.2%}"
        )


