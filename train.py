"""
Train a diabetes classifier and write deployable artifacts.

Artifacts written to ./artifacts:
- diabetes_model.joblib
- model_meta.json (feature order, training params, etc.)
- metrics.json (basic evaluation metrics)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "diabetes_model.joblib"
MODEL_META_PATH = ARTIFACTS_DIR / "model_meta.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Source dataset (kept simple for a demo project).
DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Keep feature order explicit and consistent between training and inference.
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
TARGET = "Outcome"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DEFAULT_DATASET_URL)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found: {df.columns.tolist()}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "dataset_url": DEFAULT_DATASET_URL,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    meta = {
        "features": FEATURES,
        "target": TARGET,
        "model_type": "RandomForestClassifier",
        "sklearn_params": model.get_params(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(model, MODEL_PATH)
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"OK: Model saved: {MODEL_PATH}")
    print(f"OK: Metadata saved: {MODEL_META_PATH}")
    print(f"OK: Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
