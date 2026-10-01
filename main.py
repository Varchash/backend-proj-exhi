# import joblib
# import numpy as np
# import pandas as pd
# # import pickle
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field

# app = FastAPI(
#     title="Crop Recommendation API",
#     description="Predicts the best crop to grow given soil and climate conditions, using a trained KNN model.",
#     version="1.0.0",
# )

# # Allow the frontend (any origin during dev) to call this API.
# # In production, replace "*" with your actual frontend domain.
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # with open("knn_model.pkl", "rb") as f:
# #     model = pickle.load(f)
# # with open("scaler.pkl", "rb") as f:
# #     scaler = pickle.load(f)
# model = joblib.load("knn_model.pkl")
# scaler = joblib.load("scaler.pkl")

# class CropInput(BaseModel):
#     N: float
#     P: float
#     K: float
#     temperature: float
#     humidity: float
#     ph: float
#     rainfall: float


# @app.post("/predict")
# def predict_crop(data: CropInput):
#     # Base features + engineered features expected by the scaler
#     n_p_k = data.N + data.P + data.K
#     temp_hum = data.temperature * data.humidity
#     log_rain = np.log1p(data.rainfall)

#     features = np.array(
#         [[
#             data.N,
#             data.P,
#             data.K,
#             data.temperature,
#             data.humidity,
#             data.ph,
#             data.rainfall,
#             n_p_k,
#             temp_hum,
#             log_rain,
#         ]]
#     )

#     scaled_data = scaler.transform(features)
#     prediction = model.predict(features)[0]

#     return {"recommended_crop": prediction}


import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Crop Recommendation API",
    description="Predicts the best crop to grow given soil and climate conditions, using a trained KNN model.",
    version="1.0.0",
)

# Allow the frontend (any origin during dev) to call this API.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://melodic-jalebi-310e6e.netlify.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model artifacts at startup ----
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# The model was trained on these 10 columns, in this exact order:
# 7 raw fields + 3 engineered features (N_P_K, temp_humidity, log_rainfall).
# This must match crop.drop("label", axis=1).columns from the notebook.
FEATURE_ORDER = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "N_P_K", "temp_humidity", "log_rainfall",
]


class CropInput(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil", example=90)
    P: float = Field(..., description="Phosphorus content in soil", example=42)
    K: float = Field(..., description="Potassium content in soil", example=43)
    temperature: float = Field(..., description="Temperature in Celsius", example=20.88)
    humidity: float = Field(..., description="Relative humidity in %", example=82.0)
    ph: float = Field(..., description="Soil pH value", example=6.5)
    rainfall: float = Field(..., description="Rainfall in mm", example=202.9)


class CropPrediction(BaseModel):
    recommended_crop: str
    top_3: list[dict]


def build_features(data: CropInput) -> np.ndarray:
    """Recreate the same engineered features used during training."""
    row = data.dict()
    row["N_P_K"] = data.N + data.P + data.K
    row["temp_humidity"] = data.temperature * data.humidity
    row["log_rainfall"] = np.log1p(data.rainfall)

    ordered = [row[col] for col in FEATURE_ORDER]
    return np.array(ordered).reshape(1, -1)


@app.get("/")
def root():
    return {"status": "ok", "message": "Crop Recommendation API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=CropPrediction)
def predict(data: CropInput):
    try:
        features = build_features(data)
        scaled = scaler.transform(features)

        prediction = model.predict(scaled)[0]

        probs = model.predict_proba(scaled)[0]
        classes = model.classes_
        top_idx = np.argsort(probs)[::-1][:3]
        top_3 = [
            {"crop": classes[i], "confidence": round(float(probs[i]), 4)}
            for i in top_idx
        ]

        return CropPrediction(recommended_crop=prediction, top_3=top_3)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
