"""
api.py (PRODUÇÃO COM DRIFT DETECTION)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
import json
import numpy as np
from datetime import datetime

# =========================
# CONFIG
# =========================
logging.basicConfig(level=logging.INFO)

MODEL_VERSION = "2.0.0"
LOG_FILE = "predictions_log.json"

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")

app = FastAPI(title="ML API with Drift Detection")

# =========================
# BASELINE (DADOS DE TREINO)
# =========================
"""
Aqui simulamos estatísticas do treino.
Em produção real isso viria salvo do treino.
"""

TRAIN_MEAN = np.array([125, 3, 15, 10, 0.5])
TRAIN_STD = np.array([40, 1, 8, 5, 0.5])

DRIFT_THRESHOLD = 2  # em desvios padrão

# =========================
# INPUT SCHEMA
# =========================
class HouseFeatures(BaseModel):
    size: float
    rooms: int
    age: float
    distance: float
    garage: int

# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# DRIFT DETECTION
# =========================
def detect_drift(X):
    """
    Detecta se dados estão fora da distribuição do treino.

    Estratégia:
    - calcula z-score
    - se passar threshold → drift
    """

    z_scores = np.abs((X - TRAIN_MEAN) / TRAIN_STD)

    drift_flags = z_scores > DRIFT_THRESHOLD

    return drift_flags, z_scores

# =========================
# LOGGING (SIMULA BANCO)
# =========================
def log_prediction(data, prediction, drift_flags):
    """
    Salva histórico das requisições.

    Em produção:
    → isso iria para banco ou data lake
    """

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": data.dict(),
        "prediction": float(prediction),
        "drift_detected": drift_flags.tolist()
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(data: HouseFeatures):

    X = np.array([[
        data.size,
        data.rooms,
        data.age,
        data.distance,
        data.garage
    ]])

    prediction = model.predict(X)[0]

    # DRIFT
    drift_flags, z_scores = detect_drift(X[0])

    if np.any(drift_flags):
        logging.warning(f"⚠️ DRIFT DETECTADO! z-scores: {z_scores}")

    # LOG
    log_prediction(data, prediction, drift_flags)

    logging.info(f"Input: {data.dict()} | Prediction: {prediction}")

    return {
        "predicted_price": float(prediction),
        "model_version": MODEL_VERSION,
        "drift_detected": drift_flags.tolist()
    }
