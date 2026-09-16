"""Starter customer churn classification pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_model(numeric_features, categorical_features):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preprocessing = ColumnTransformer([("num", numeric, numeric_features), ("cat", categorical, categorical_features)])
    return Pipeline([("preprocess", preprocessing), ("model", LogisticRegression(max_iter=1000))])

def train(df: pd.DataFrame, target: str = "churn"):
    X = df.drop(columns=[target])
    y = df[target]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    model = build_model(numeric, categorical)
    return model.fit(X, y)
