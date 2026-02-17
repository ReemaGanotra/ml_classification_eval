# ML/CV Assignment: Comprehensive Pipeline


## 🏗️ Project Structure

```
ml_cv_assignment/
├── data/                  # Datasets (gitignored, auto-downloaded)
├── src/
│   ├── ml_classification.py    # Task 1: ML model
│   ├── edge_detection.py        # Task 2: CV edge detection
│   ├── deployment.py            # Task 3: FastAPI deployment
│   └── utils.py                 # Shared utilities
├── models/                # Saved models 
├── artifacts/             # Plots, metrics, logs
├── README.md
├── SUMMARY_REPORT.md    
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/ReemaGanotra/ml_classification_eval.git
cd ml_cv_assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run All Tasks

```bash
# Task 1: ML Classification
python src/ml_classification.py

# Task 2: Edge Detection
python src/edge_detection.py

# Task 3: Deployment (FastAPI)
python src/deployment.py
```

### Datasets and Metrics:
Please refer to SUMMARY_REPORT.md for information on datasets used and performance metrics.

## FastAPI Deployment Pipeline

### 5-Stage MLOps Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  1. Data     │──▶│  2. Training │──▶│  3. Model    │──▶│  4. Serving  │──▶│  5. Monitor  │
│  Ingestion   │   │  & Compare   │   │  Registry    │   │  (FastAPI)   │   │  (PSI drift) │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
ml_classification.py                  deployment.py       deployment.py      deployment.py
```

| Stage | Implementation |
|---|---|
| **Data Ingestion** | `load_data()` – auto-downloads, validates, cleans |
| **Training** | `compare_models()` + `evaluate_final_model()` – LR / RF / SVM |
| **Model Registry** | `ModelRegistry` class – file-backed, staging→production promotion |
| **Serving** | FastAPI + Pydantic – single & batch endpoints |
| **Monitoring** | `ModelMonitor` + `utils.calculate_psi()` – PSI drift detection |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Liveness check |
| GET | `/health` | Readiness probe (for load balancers) |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction (≤ 500 records) |
| GET | `/metrics` | Live serving statistics |
| POST | `/drift_check` | PSI-based feature drift report |
| GET | `/registry` | All registered model versions |
| GET | `/pipeline_info` | Full 5-stage pipeline documentation |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39, "workclass": 4, "education": 9,
    "education_num": 13, "marital_status": 4,
    "occupation": 1, "relationship": 0, "race": 4, "sex": 1,
    "hours_per_week": 40, "native_country": 39,
    "capital_net": 2174, "hours_per_week_norm": 1.0,
    "age_edu_interaction": 507, "is_married": 0
  }'
```

```json
{
  "prediction": 0,
  "probability": 0.2341,
  "income_label": "<=50K",
  "model_name": "Random Forest",
  "model_version": "1.0.0",
  "timestamp": "2026-02-17T10:30:00"
}
```

## Drift monitoring

```bash
# After making some predictions
curl -X POST "http://localhost:8000/drift_check?window=200"
```

Uses `calculate_psi()` from the shared `utils.py`.
PSI > 0.20 on any feature triggers `"retrain_recommended": true`.

---

## Reproducibility

```python
SEED = 42   # set in ml_classification.py, passed to set_seed() from utils.py
```

`set_seed()` fixes `random`, `numpy`, and `tensorflow` seeds.
All sklearn estimators receive `random_state=SEED`.

---




## Optimization

**ONNX Export:**
```python
# Tensorflow → ONNX
tf2onnx.convert.from_keras(
                edge_cnn,
                input_signature=spec,
                opset=13,
                output_path=str(onnx_path)
            )

```

## Latency Benchmarking (CPU):
| Model | Format     | Avg Latency (ms) | Speedup |
|-------|------------|-----------------|---------|
| CNN   | Tensorflow | 87.0322         | 1.0x    |
| CNN   | ONNX       | 37.2428         | 2.33x   |


**Optimization Explained: Quantization**
- Technique: Convert FP32 weights → INT8, 2-3x inference speedup
- Trade-off: <2% accuracy loss (acceptable for most applications)
- Implementation: `tf.lite.TFLiteConverter`

### Task 5: Drift Monitoring Plan

**Statistical Drift Detection:**
- **Population Stability Index (PSI):**
  - PSI < 0.1: No shift
  - 0.1 < PSI < 0.2: Moderate shift → investigate
  - PSI > 0.2: Severe shift → retrain model

**Monitoring Strategy:**
1. **Feature Drift:** Track PSI for each input feature (weekly)
2. **Prediction Drift:** Monitor output distribution shifts
3. **Performance Degradation:** Track F1/precision on validation set
4. **Alerting:** Automated retraining trigger when PSI > 0.2



## 🧪 Reproducibility

All experiments use fixed random seeds:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```


