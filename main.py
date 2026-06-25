from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, json

app = FastAPI()

with open("model_params.json") as f:
    params = json.load(f)

mean      = np.array(params["mean"])
std       = np.array(params["std"])
centroids = np.array(params["centroids_zscore"])

class InputData(BaseModel):
    Usia: float
    BMI: float
    HbA1c: float
    HOMA2_B: float
    HOMA2_IR: float

@app.post("/predict")
def predict(data: InputData):
    x = np.array([data.Usia, data.BMI, data.HbA1c, data.HOMA2_B, data.HOMA2_IR])
    x_scaled = (x - mean) / std
    klaster  = int(np.argmin(np.linalg.norm(centroids - x_scaled, axis=1))) + 1
    return {"klaster": klaster}
