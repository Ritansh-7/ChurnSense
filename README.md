# ChurnSense | Customer Churn Prediction

**ChurnSense** is a machine learning project focused on predicting customer churn and translating model predictions into actionable retention insights. It combines exploratory data analysis, model comparison, churn-risk evaluation, and an API-based prediction interface.

The project explores how customer data can support churn identification, risk-based prioritization, and retention-oriented business decisions.

## Features

- **Exploratory Data Analysis:** Analyze customer characteristics and patterns associated with churn.
- **Machine Learning:** Train and compare classification models for customer churn prediction.
- **Model Evaluation:** Evaluate predictions using classification metrics, ROC-AUC, PR-AUC, confusion matrices, and probability-based metrics.
- **Churn Risk Scoring:** Estimate customer churn probabilities and apply a decision threshold to classify risk.
- **Retention Analytics:** Incorporate retention costs, expected value, and customer prioritization into business decisions.
- **Prediction API:** Serve model predictions through a FastAPI application.

## Model Performance

Reported test-set results:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.8406 |
| PR-AUC | 0.6472 |
| Accuracy | 77.71% |
| Precision | 57.11% |
| Recall | 64.44% |
| Brier Score | 0.1606 |

**Decision threshold:** 0.618

These metrics summarize the evaluated model's performance on the test set. The appropriate operating threshold depends on the relative costs of missed churners and unnecessary retention interventions.

## Project Structure

```text
ChurnSense/
├── api.py
├── ChurnSENSE.ipynb
└── README.md
```

### Files

- **`ChurnSENSE.ipynb`** — Notebook containing the churn prediction workflow, data analysis, model training, and evaluation.
- **`api.py`** — API application for serving churn predictions.
- **`README.md`** — Project documentation, model results, and usage instructions.

## Machine Learning Workflow

1. Explore customer data and analyze churn patterns.
2. Prepare features and the target variable for modeling.
3. Train and compare classification models.
4. Evaluate model performance using appropriate validation metrics.
5. Select a decision threshold for churn-risk classification.
6. Use predictions to support customer prioritization and retention decisions.
7. Expose predictions through the API.

## Business Application

ChurnSense connects churn prediction with retention-oriented decision-making. Customer risk scores can help identify customers who may warrant further attention, while retention-cost and expected-value calculations provide a framework for prioritizing interventions.

A high churn probability does not guarantee that a customer will leave, nor does it establish that a retention offer will change the outcome. The effectiveness of retention strategies requires separate validation.

## Limitations

- Predictions depend on the quality and representativeness of the available data.
- Model explanations and feature relationships do not establish causation.
- Threshold selection involves trade-offs between precision and recall.
- Predicted churn risk alone does not measure customer responsiveness to retention offers.
- Business impact must be evaluated independently of predictive performance.

## Future Improvements

- Customer-level feature explanations.
- Further calibration and segment-level error analysis.
- Automated testing and reproducible model training.
- Model versioning and performance monitoring.
- Controlled experiments to measure incremental retention impact.

## Objective

ChurnSense demonstrates an end-to-end approach to customer churn prediction, combining machine learning evaluation with API-based inference and retention-focused business analysis.
