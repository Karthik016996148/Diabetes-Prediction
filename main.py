from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "diabetes_model.joblib"
MODEL_META_PATH = ARTIFACTS_DIR / "model_meta.json"

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

_model: Any | None = None
_features: list[str] = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(ge=0, le=20)
    Glucose: float = Field(ge=0, le=500)
    BloodPressure: float = Field(ge=0, le=300)
    BMI: float = Field(ge=0, le=100)
    Age: int = Field(ge=0, le=120)


@app.on_event("startup")
def _startup_load_model() -> None:
    global _model, _features
    if MODEL_META_PATH.exists():
        try:
            meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
            features = meta.get("features")
            if isinstance(features, list) and all(isinstance(x, str) for x in features):
                _features = features
        except Exception:
            # Don't crash on startup if metadata is malformed; API can still run.
            pass

    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)


@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live", "model_loaded": _model is not None}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
def predict(data: DiabetesInput):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found. Train it first: `python train.py` (expected: {MODEL_PATH.as_posix()})",
        )

    # Strict feature order matching training.
    row = [getattr(data, f) for f in _features]
    input_data = np.array([row], dtype=float)

    pred = int(_model.predict(input_data)[0])

    response = {"diabetic": bool(pred)}
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(input_data)[0]
        # proba[1] should correspond to Outcome=1 for standard sklearn classifiers
        response["probability_diabetic"] = float(proba[1])
    return response
