from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, json

app = FastAPI()

with open("model_params.json") as f:
    params = json.load(f)

mean      = np.array(params["mean"])
std       = np.array(params["std"])
centroids = np.array(params["centroids_zscore"])

LABEL_MAP = {
    1: "Severe Insulin-Deficient Diabetes (SIDD)",
    2: "Mild Obesity-Related Diabetes (MOD)", 
    3: "Severe Insulin-Resistant Diabetes (SIRD)",
    4: "Mild Age-Related Diabetes (MARD)"
}

class InputData(BaseModel):
    Usia: float
    BMI: float
    HbA1c: float
    HOMA2B: float   # ← sesuaikan dengan yang Laravel kirim
    HOMA2IR: float  # ← sesuaikan dengan yang Laravel kirim

@app.post("/predict")
def predict(data: InputData):
    x = np.array([data.Usia, data.BMI, data.HbA1c, data.HOMA2B, data.HOMA2IR])
    x_scaled = (x - mean) / std
    klaster  = int(np.argmin(np.linalg.norm(centroids - x_scaled, axis=1))) + 1
    return {
        "klaster": klaster,
        "label": LABEL_MAP[klaster]  # ← tambahkan ini
    }
