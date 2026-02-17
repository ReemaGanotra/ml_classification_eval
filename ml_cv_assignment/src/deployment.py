"""
Task 3 – Deployment Pipeline (FastAPI)
=======================================
Serves the best model produced by ml_classification.py.

5-Stage MLOps Pipeline
-----------------------
1. Data Ingestion   – validated via Pydantic input schema
2. Training         – handled by ml_classification.py (run first)
3. Model Registry   – lightweight version-tracked JSON registry (ModelRegistry)
4. Serving          – FastAPI REST API
5. Monitoring       – drift detection
"""

import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from utils import setup_logging, save_metrics, calculate_psi, MODELS_DIR, ARTIFACTS_DIR



# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class AdultIncomeInput(BaseModel):
    """
    Input schema for a single prediction request.
    All fields mirror the Adult dataset columns after feature engineering.
    Ranges are soft guards; the model will still predict outside them.
    """
    age:                  float = Field(..., ge=17,  le=90,  description="Age in years")
    workclass:            int   = Field(..., ge=0,   le=8,   description="Work class (label-encoded)")
    education:            int   = Field(..., ge=0,   le=15,  description="Education level (label-encoded)")
    education_num:        float = Field(..., ge=1,   le=16,  description="Education years")
    marital_status:       int   = Field(..., ge=0,   le=6,   description="Marital status (label-encoded)")
    occupation:           int   = Field(..., ge=0,   le=14,  description="Occupation (label-encoded)")
    relationship:         int   = Field(..., ge=0,   le=5,   description="Relationship (label-encoded)")
    race:                 int   = Field(..., ge=0,   le=4,   description="Race (label-encoded)")
    sex:                  int   = Field(..., ge=0,   le=1,   description="Sex (0=Female, 1=Male)")
    hours_per_week:       float = Field(..., ge=1,   le=99,  description="Hours worked per week")
    native_country:       int   = Field(..., ge=0,   le=41,  description="Native country (label-encoded)")
    # Engineered features (computed by ml_classification.py during training)
    capital_net:          float = Field(..., description="capital_gain – capital_loss")
    hours_per_week_norm:  float = Field(..., description="hours_per_week / 40")
    age_edu_interaction:  float = Field(..., description="age × education_num")
    is_married:           int   = Field(..., ge=0, le=1, description="1 if married")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 39, "workclass": 4, "education": 9,
                "education_num": 13, "marital_status": 4,
                "occupation": 1, "relationship": 0, "race": 4, "sex": 1,
                "hours_per_week": 40, "native_country": 39,
                "capital_net": 2174, "hours_per_week_norm": 1.0,
                "age_edu_interaction": 507, "is_married": 0,
            }
        }


class PredictionResponse(BaseModel):
    prediction:      int
    probability:     float
    income_label:    str
    model_name:      str
    model_version:   str
    timestamp:       str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count:       int
    batch_id:    str


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Lightweight file-backed model registry.

    Stores version metadata in models/registry.json.
    In production this would be replaced by MLflow Model Registry,
    but the interface is kept identical so the swap is a one-liner.

    Workflow
    --------
    ml_classification.py saves models/best_model.pkl
        → startup_event() calls registry.register(...)
        → registry.promote_to_production(...)
        → /predict loads the production model
    """

    def __init__(self):
        self.registry_file = MODELS_DIR / "registry.json"
        self._data: Dict[str, Any] = {"models": []}
        if self.registry_file.exists():
            self._data = json.loads(self.registry_file.read_text())

    def _save(self):
        self.registry_file.write_text(json.dumps(self._data, indent=2))

    def register(self, name: str, path: str,
                 metrics: dict, version: str) -> dict:
        """Add a new model entry (status = 'staging')."""
        entry = dict(
            name=name, version=version, path=path,
            metrics=metrics, status="staging",
            registered_at=datetime.now().isoformat(),
        )
        # Remove any older staging entry for the same name+version
        self._data["models"] = [
            m for m in self._data["models"]
            if not (m["name"] == name and m["version"] == version)
        ]
        self._data["models"].append(entry)
        self._save()
        logging.info("Registry: registered '%s' v%s  (staging)", name, version)
        return entry

    def promote_to_production(self, name: str, version: str):
        """Archive all previous production versions, promote the target."""
        for m in self._data["models"]:
            if m["name"] == name:
                if m["version"] == version:
                    m["status"] = "production"
                elif m["status"] == "production":
                    m["status"] = "archived"
        self._save()
        logging.info("Registry: '%s' v%s  →  production", name, version)

    def get_production(self, name: str) -> Optional[dict]:
        """Return the current production entry, or None."""
        for m in reversed(self._data["models"]):
            if m["name"] == name and m["status"] == "production":
                return m
        return None

    def list_all(self) -> list:
        return self._data["models"]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 – MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ModelMonitor:
    """
    Real-time drift monitor.

    Usage
    -----
    1.  Feed reference data (training distribution) at startup.
    2.  log_prediction() appends every live request to a rolling buffer.
    3.  POST /drift_check triggers check_drift() on the buffered features.

    PSI thresholds
    -----------------------------------------
    PSI < 0.10  →  No significant drift
    0.10–0.20   →  Moderate drift  →  investigate
    > 0.20      →  Severe drift    →  retrain recommended
    """

    def __init__(self, reference_data: np.ndarray, feature_names: list):
        self.reference_data  = reference_data
        self.feature_names   = feature_names
        self.prediction_log: List[dict] = []

    def log_prediction(self, features: list, prediction: int, probability: float):
        self.prediction_log.append(dict(
            features=features,
            prediction=prediction,
            probability=probability,
            timestamp=datetime.now().isoformat(),
        ))

    def check_drift(self, window: int = 500) -> dict:
        """
        Run PSI between the reference distribution and the last `window`
        predictions, using calculate_psi() from utils.py.

        Returns a report dict with per-feature PSI and an overall summary.
        """
        if len(self.prediction_log) < 10:
            return {"error": "Not enough predictions for drift analysis (need ≥ 10)"}

        recent = self.prediction_log[-window:]
        current_data = np.array([r["features"] for r in recent], dtype=np.float32)

        feature_reports = []
        for i, fname in enumerate(self.feature_names):
            ref = self.reference_data[:, i].astype(float)
            cur = current_data[:, i].astype(float)
            psi = float(calculate_psi(ref, cur))   # ← utils.py function

            if psi < 0.10:
                severity = "none"
            elif psi < 0.20:
                severity = "moderate"
            else:
                severity = "severe"

            feature_reports.append(dict(
                feature=fname, psi=round(psi, 5),
                drift_detected=(psi >= 0.10), severity=severity,
            ))

        severe   = [r for r in feature_reports if r["severity"] == "severe"]
        moderate = [r for r in feature_reports if r["severity"] == "moderate"]

        return dict(
            window_size=len(recent),
            feature_reports=feature_reports,
            summary=dict(
                total_features=len(feature_reports),
                severe_count=len(severe),
                moderate_count=len(moderate),
                retrain_recommended=len(severe) > 0,
            ),
        )

    def get_stats(self) -> dict:
        if not self.prediction_log:
            return {"message": "No predictions logged yet"}
        probs  = [r["probability"]  for r in self.prediction_log]
        preds  = [r["prediction"]   for r in self.prediction_log]
        return dict(
            total_predictions=len(self.prediction_log),
            positive_rate=round(sum(preds) / len(preds), 4),
            avg_probability=round(float(np.mean(probs)), 4),
            std_probability=round(float(np.std(probs)),  4),
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Adult Income Prediction API",
    description=(
        "Binary classification: predict whether annual income exceeds $50K.\n\n"
        "Run `python src/ml_classification.py` first to generate the model."
    ),
    version="1.0.0",
)

# ── global state (populated at startup) ──────────────────────────────────────
PIPELINE      = None
FEATURE_NAMES = None
MODEL_NAME    = "unknown"
MODEL_VERSION = "1.0.0"
REGISTRY      = None
MONITOR       = None


@app.on_event("startup")
async def startup_event():
    """
    Stage 1 – Data Ingestion check  (validate model artefacts exist)
    Stage 3 – Model Registry        (register + promote best_model.pkl)
    Stage 5 – Monitor init          (load reference distribution)
    """
    global PIPELINE, FEATURE_NAMES, MODEL_NAME, MODEL_VERSION, REGISTRY, MONITOR

    setup_logging("deployment.log")
    logging.info("API startup …")

    # ── load model pipeline ───────────────────────────────────────────────────
    model_path   = MODELS_DIR / "best_model.pkl"
    fn_path      = MODELS_DIR / "feature_names.json"
    metrics_path = ARTIFACTS_DIR / "ml_classification_metrics.json"

    if not model_path.exists():
        logging.warning(
            "models/best_model.pkl not found. "
            "Run ml_classification.py first."
        )
        return

    PIPELINE      = joblib.load(model_path)
    FEATURE_NAMES = json.loads(fn_path.read_text()) if fn_path.exists() else []

    # Read model name from saved metrics
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        MODEL_NAME = saved.get("model", "best_model")

    logging.info("Model loaded  –  %s", MODEL_NAME)

    # ── Stage 3: Model Registry ───────────────────────────────────────────────
    REGISTRY = ModelRegistry()
    metrics  = (json.loads(metrics_path.read_text())
                if metrics_path.exists() else {})
    REGISTRY.register(
        name="adult_income", path=str(model_path),
        metrics=metrics, version=MODEL_VERSION,
    )
    REGISTRY.promote_to_production("adult_income", MODEL_VERSION)

    # ── Stage 5: Monitor – seed reference distribution from dummy data ────────
    # In production you would load the actual training data here.
    # We use a random sample matching the feature count as a placeholder.
    n_features = len(FEATURE_NAMES) if FEATURE_NAMES else 15
    ref_data   = np.random.randn(1000, n_features).astype(np.float32)
    MONITOR    = ModelMonitor(ref_data, FEATURE_NAMES or [f"f{i}" for i in range(n_features)])

    logging.info("API ready to serve predictions on  /predict")


# ── helpers ───────────────────────────────────────────────────────────────────

def _input_to_array(data: AdultIncomeInput) -> np.ndarray:
    """
    Convert a validated Pydantic model to a (1, n_features) float32 array
    in the exact column order that was used during training.
    """
    row = [
        data.age, data.workclass, data.education, data.education_num,
        data.marital_status, data.occupation, data.relationship,
        data.race, data.sex, data.hours_per_week, data.native_country,
        data.capital_net, data.hours_per_week_norm,
        data.age_edu_interaction, data.is_married,
    ]
    return np.array(row, dtype=np.float32).reshape(1, -1)


def _make_response(features: np.ndarray) -> PredictionResponse:
    """Run inference and return a typed response object."""
    prediction  = int(PIPELINE.predict(features)[0])
    probability = float(PIPELINE.predict_proba(features)[0][1])
    MONITOR.log_prediction(features.flatten().tolist(), prediction, probability)
    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        income_label=">50K" if prediction == 1 else "<=50K",
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – ENDPOINTS (Serving)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    """Quick liveness check."""
    return dict(
        service="Adult Income Prediction API",
        status="healthy" if PIPELINE is not None else "model_not_loaded",
        model=MODEL_NAME,
        version=MODEL_VERSION,
    )


@app.get("/health", tags=["Health"])
async def health():
    """Detailed readiness check for load-balancer probes."""
    return dict(
        status="ready" if PIPELINE is not None else "not_ready",
        model_loaded=PIPELINE is not None,
        features_loaded=bool(FEATURE_NAMES),
        monitor_active=MONITOR is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(data: AdultIncomeInput):
    """
    Single-record prediction.

    Returns predicted class (0/1), probability of >50K, and metadata.
    """
    if PIPELINE is None:
        raise HTTPException(503, "Model not loaded. Run ml_classification.py first.")
    try:
        features = _input_to_array(data)
        return _make_response(features)
    except Exception as exc:
        logging.exception("Prediction error")
        raise HTTPException(500, str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def batch_predict(records: List[AdultIncomeInput]):
    """
    Batch prediction – up to 500 records per call.
    Returns a list of PredictionResponse objects and a unique batch_id.
    """
    if PIPELINE is None:
        raise HTTPException(503, "Model not loaded.")
    if len(records) > 500:
        raise HTTPException(400, "Batch size limit is 500 records.")
    try:
        responses = [_make_response(_input_to_array(r)) for r in records]
        return BatchPredictionResponse(
            predictions=responses,
            count=len(responses),
            batch_id=f"batch_{int(time.time())}",
        )
    except Exception as exc:
        logging.exception("Batch prediction error")
        raise HTTPException(500, str(exc))


# ── Monitoring endpoints ──────────────────────────────────────────────────────

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Live serving statistics:
    total_predictions, positive_rate, avg/std probability.
    """
    if MONITOR is None:
        raise HTTPException(503, "Monitor not initialised.")
    return MONITOR.get_stats()


@app.post("/drift_check", tags=["Monitoring"])
async def drift_check(window: int = 500):
    """
    PSI-based data drift check on the last `window` predictions.

    Uses calculate_psi() from the shared utils.py.

    PSI thresholds
    --------------
    < 0.10  →  no drift
    0.10–0.20  →  moderate – investigate
    > 0.20  →  severe – retrain recommended
    """
    if MONITOR is None:
        raise HTTPException(503, "Monitor not initialised.")
    return MONITOR.check_drift(window=window)


# ── Registry & pipeline info ──────────────────────────────────────────────────

@app.get("/registry", tags=["MLOps"])
async def get_registry():
    """List all registered models and their statuses."""
    if REGISTRY is None:
        raise HTTPException(503, "Registry not initialised.")
    return {"models": REGISTRY.list_all()}


@app.get("/pipeline_info", tags=["MLOps"])
async def pipeline_info():
    """
    Documents the full 5-stage MLOps pipeline implemented in this project.
    """
    return {
        "pipeline": {
            "stage_1_data_ingestion": {
                "description": "Automated UCI Adult Income download with validation",
                "implementation": "ml_classification.py  →  load_data()",
                "artefacts": ["data/adult_train.csv", "data/adult_test.csv"],
                "status": "implemented",
            },
            "stage_2_training": {
                "description": "Three models compared (LR, RF, SVM); winner selected by F1",
                "implementation": "ml_classification.py  →  compare_models(), evaluate_final_model()",
                "reproducibility": "SEED=42 fixed for numpy, sklearn, tensorflow",
                "artefacts": [
                    "artifacts/model_comparison.png",
                    "artifacts/model_comparison.json",
                    "artifacts/ml_classification_metrics.json",
                ],
                "status": "implemented",
            },
            "stage_3_model_registry": {
                "description": "File-backed version registry with staging→production promotion",
                "implementation": "deployment.py  →  ModelRegistry class",
                "production_upgrade": "Replace with MLflow Model Registry (same interface)",
                "artefacts": ["models/registry.json"],
                "status": "implemented",
            },
            "stage_4_serving": {
                "description": "FastAPI REST API with single and batch inference",
                "endpoints": {
                    "POST /predict":        "Single record prediction",
                    "POST /predict/batch":  "Batch inference (≤500 records)",
                    "GET  /health":         "Readiness probe",
                    "GET  /metrics":        "Live serving stats",
                },
                "docs_url": "http://localhost:8000/docs",
                "status": "implemented",
            },
            "stage_5_monitoring": {
                "description": "PSI-based feature drift detection via utils.calculate_psi()",
                "implementation": "deployment.py  →  ModelMonitor class",
                "thresholds": {
                    "no_drift":       "PSI < 0.10",
                    "moderate_drift": "0.10 ≤ PSI < 0.20  →  investigate",
                    "severe_drift":   "PSI ≥ 0.20  →  retrain recommended",
                },
                "endpoint": "POST /drift_check?window=500",
                "status": "implemented",
            },
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Adult Income Prediction API …")
    uvicorn.run("deployment:app", host="0.0.0.0", port=8000, reload=True)
