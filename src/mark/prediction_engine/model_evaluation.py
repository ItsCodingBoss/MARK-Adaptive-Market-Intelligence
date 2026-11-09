import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix,balanced_accuracy_score,brier_score_loss,log_loss


def evaluate_classifier(actual, predicted):
    actual = np.asarray(actual,dtype = int)
    predicted = np.asarray(predicted,dtype = int)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Both actual and predicted arrays must be 1-dimensional")
    #scikit-learn's classification metrics generally expect a one-dimensional target array for ordinary binary classification
    if  not np.isin(actual,[0,1]).all():
        raise ValueError("Actual values must be binary (0 or 1)")
    if  not np.isin(predicted,[0,1]).all():
        raise ValueError("Predicted values must be binary (0 or 1)")
    
    tn, fp, fn, tp = confusion_matrix(
    actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1_score": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "samples": int(actual.size),
    }
def make_class_frequency_baseline(training_targets, n_predictions, classes):

    """
    Create a probability baseline using class frequencies
    from training data.

    The same baseline can be reused for different assets
    and binary classification models.
    """


    training_targets = np.asarray(training_targets)
    classes = np.asarray(classes)
    if training_targets.ndim != 1:
        raise ValueError("Training targets must be one-dimensional.")
    if training_targets.size == 0:
        raise ValueError("Training targets cannot be empty.")
    if classes.ndim != 1 or classes.size != 2:
        raise ValueError("Exactly two classes are required.")
    if len(set(classes.tolist()))!= 2:
        raise ValueError("Classes must be unique")
    if not np.isin(training_targets, classes).all():
        raise ValueError("Training targets must only contain the specified classes")
    
    class_frequencies = np.array([
    np.mean(training_targets == label)
    for label in classes
    ])
    baseline_probabilities = np.tile(class_frequencies, (n_predictions, 1))
    return baseline_probabilities
def evaluate_probability_predictions( actual,probabilities,classes,positive_label=1,):
    if probabilities.shape != (actual.size, classes.size):
        raise ValueError(
            "Probability dimensions do not match actual targets and classes."
        )
    positive_index = np.flatnonzero(classes == positive_label)
    probabilities_positive = probabilities[:, positive_index[0]]
    actual_binary = (actual == positive_label).astype(int)
    brier = brier_score_loss(actual_binary, probabilities_positive)
    loss = log_loss(actual,probabilities,labels=classes.tolist(),)
    return {
    "brier_score": float(brier),
    "log_loss": float(loss),
    "samples": int(actual.size),
    }
