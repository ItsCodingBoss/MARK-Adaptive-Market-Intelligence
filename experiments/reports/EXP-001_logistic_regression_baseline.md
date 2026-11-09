# EXP-001 — Logistic Regression Baseline Evaluation

## 1. Experiment information

- Experiment ID: EXP-001
- Project: MARK — Adaptive Market Intelligence & Research System
- Asset: NIFTY 50 (^NSEI)
- Prediction horizon: Next trading day
- Model: Logistic Regression
- Experiment status: Completed — preliminary evaluation
- Decision: Retain as reference model; do not promote.

## 2. Research question

Does Logistic Regression provide useful next-trading-day directional predictions compared with a simple baseline?

Additional investigation:
- Why does the model have weak directional performance?
- Do its probabilities distinguish actual UP days from DOWN/FLAT days?
- Does it improve probability quality over a class-frequency baseline?

## 3. Dataset and target

- Dataset: nifty50_features.csv
- Source: Yahoo Finance (^NSEI)
- Target: target_up
- Class 1: Next trading day's close is higher than today's close.
- Class 0: Next trading day's close is equal to or lower than today's close.

Features used:
- daily_return_pct
- return_5d_pct
- return_10d_pct
- sma_10
- sma_20
- volatility_10d
- volatility_20d
- rsi_14
- price_vs_sma10_pct
- price_vs_sma20_pct

## 4. Validation design

Chronological train/validation/test split.

Training:
- Rows: 2,563
- Period: 2011-11-03 to 2022-04-27
- Training UP frequency: 53.65%

Validation:
- Rows: 548
- Period: 2022-04-28 to 2024-07-19

Test:
- Rows: 550
- Period: 2024-07-22 to 2026-10-08
- Test results: Not included in this experiment.

## 5. Model evaluation

| Metric | Majority baseline | Logistic Regression |
|---|---:|---:|
| Accuracy | 56.20% | 57.30% |
| Balanced accuracy | Not recorded | 52.26% |
| Precision | Not recorded | 57.43% |
| Recall | Not recorded | 92.86% |
| F1 score | Not recorded | 70.97% |

Logistic Regression confusion matrix:

- True Negative: 28
- False Positive: 212
- False Negative: 22
- True Positive: 286

Percentage of validation observations predicted UP: 90.88%.

## 6. Threshold experiment

Thresholds tested: 0.35, 0.40, 0.45, 0.50, 0.55, 0.60 and 0.65.

Best balanced accuracy among tested thresholds:

- Threshold: 0.55
- Balanced accuracy: 55.68%
- Accuracy: 52.92%
- Precision: 66.03%
- Recall: 33.44%
- Predicted UP: 28.47%

Decision: No permanent threshold selected. Results are from one validation period and may be affected by threshold overfitting.

## 7. Probability distribution investigation

Overall predicted UP probability:

- Minimum: 0.3925
- 25th percentile: 0.5168
- Median: 0.5349
- 75th percentile: 0.5532
- Maximum: 0.6898
- Observations between 0.50 and 0.55: 342

Probability by actual outcome:

| Statistic | Actual DOWN/FLAT | Actual UP |
|---|---:|---:|
| Observations | 240 | 308 |
| Mean probability of UP | 53.27% | 53.74% |
| Median probability of UP | 53.00% | 53.80% |
| 25th percentile | 51.60% | 51.90% |
| 75th percentile | 54.70% | 55.57% |

Finding: The probability distributions overlap substantially. The model shows weak separation between actual outcomes.

## 8. Probability quality evaluation

| Metric | Logistic Regression | Probability baseline |
|---|---:|---:|
| Brier score | 0.24545 | 0.24680 |
| Log loss | 0.68405 | 0.68675 |

Relative improvement over probability baseline:

- Brier score: approximately 0.55%
- Log loss: approximately 0.39%

Finding: Logistic Regression provides a small improvement in probability quality. The improvement is not sufficient evidence of a robust forecasting advantage.

## 9. Experiment conclusion

The model performs only slightly better than the simple baselines.

Its high UP recall is accompanied by many false-positive predictions. Its predicted probabilities show only weak separation between the actual outcome groups.

The experiment does not demonstrate that Logistic Regression provides a sufficiently strong or reliable trading signal.

This does not prove that Logistic Regression is universally unsuitable.

## 10. Decision

- Preserve the model and evaluation results as a reference experiment.
- Do not promote it to production.
- Do not continue repeatedly tuning thresholds to chase 80% accuracy.
- Investigate data quality, target alignment, feature usefulness and market regimes.
- Compare additional algorithms using the same dataset split and evaluation criteria.

## 11. Next actions

1. Audit target alignment and possible data leakage.
2. Complete exploratory data analysis and feature investigation.
3. Investigate market-regime differences.
4. Train and evaluate Random Forest as the next candidate.
5. Compare it against the same baseline and Logistic Regression.
6. Keep the final test period protected until model selection is complete.

## 12. Reuse and reproducibility

Reusable evaluation functions are located in:

src/mark/prediction_engine/model_evaluation.py

These include:
- evaluate_classifier()
- make_class_frequency_baseline()
- evaluate_probability_predictions()

Do not rewrite these calculations for another asset or compatible binary classification model.

For future experiments, record the dataset version, target definition, features, chronological split, configuration, metrics and decision.

**Final experiment status: Completed as a preliminary reference-model evaluation.**