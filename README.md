# Customer Churn Prediction

Machine-learning portfolio project demonstrating a reproducible classification pipeline for identifying customers at risk of churn.

## Workflow
`CSV -> Train/Test Split -> Missing-value Imputation -> Encoding/Scaling -> Logistic Regression -> Evaluation`

## Tech Stack
Python, Pandas, scikit-learn

## Structure
```text
src/model_pipeline.py   # preprocessing, model construction and metrics
src/train.py            # command-line training entry point
```

## Modeling Design
- Numeric features: median imputation + standard scaling
- Categorical features: most-frequent imputation + one-hot encoding
- Model: class-balanced logistic regression
- Metrics: accuracy, precision, recall, ROC-AUC
- Reproducible train/test split using a fixed random seed

## Run
```bash
python src/train.py data/customers.csv --target churn
```

The training command prints evaluation metrics as JSON. No performance score is claimed in this repository until a documented dataset is evaluated.

## Why Logistic Regression?
It provides an interpretable baseline and a useful benchmark before testing more complex models such as Random Forest, XGBoost, or gradient boosting.

## Next Steps
Add feature importance/explainability, cross-validation, model persistence, threshold tuning, tests, and a small demo application.