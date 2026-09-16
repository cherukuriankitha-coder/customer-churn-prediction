"""CLI training entry point for the churn project."""
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from model_pipeline import build_pipeline, evaluate


def main(path: str, target: str) -> None:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found")
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    print(json.dumps(evaluate(model, X_test, y_test), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--target", default="churn")
    args = parser.parse_args()
    main(args.data, args.target)
