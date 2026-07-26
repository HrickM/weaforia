"""
================================================================================
 CLIMATE VISIBILITY — BACKEND API (FastAPI)
================================================================================
Serves the trained visibility model and additionally classifies a "scene"
(sunny / cloudy / foggy / rainy / stormy / snowy / clear-night etc.) from the
same input parameters so the frontend can render matching visual effects.

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

The frontend (frontend/index.html) expects this API at /api/predict.
Set MODEL_PATH env var if the .joblib file lives somewhere other than
../model/visibility_model.joblib
================================================================================
"""

import os
from datetime import datetime

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
app = FastAPI(
    title="Climate Visibility Prediction API",
    description="Predicts maximum visibility distance from weather parameters "
                 "and classifies the current 'scene' for UI theming.",
    version="1.0.0",
)

# Allow the frontend (any origin — tighten this for production) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "model", "visibility_model.joblib"),
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"Model file not found at {MODEL_PATH}. Run training/train_model.py first.",
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class WeatherInput(BaseModel):
    dry_bulb_temp_f: float = Field(..., description="Dry bulb (air) temperature in °F")
    wet_bulb_temp_f: float = Field(..., description="Wet bulb temperature in °F")
    dew_point_temp_f: float = Field(..., description="Dew point temperature in °F")
    relative_humidity: float = Field(..., ge=0, le=100, description="Relative humidity %")
    wind_speed: float = Field(..., ge=0, description="Wind speed (mph)")
    wind_direction: float = Field(..., ge=0, le=360, description="Wind direction (degrees)")
    station_pressure: float = Field(..., description="Station pressure (inHg)")
    sea_level_pressure: float = Field(..., description="Sea level pressure (inHg)")
    precip: float = Field(0.0, ge=0, description="Precipitation (inches)")
    date: str | None = Field(None, description="ISO datetime, defaults to now (used for seasonality features)")


class PredictionResponse(BaseModel):
    visibility_km: float
    visibility_miles: float
    scene: str
    scene_label: str
    severity: str
    confidence_note: str
    model_used: str


# --------------------------------------------------------------------------
# Feature engineering (must mirror training/train_model.py exactly)
# --------------------------------------------------------------------------
def build_features(payload: WeatherInput):
    dt = datetime.fromisoformat(payload.date) if payload.date else datetime.utcnow()
    month, hour = dt.month, dt.hour

    dew_point_spread = payload.dry_bulb_temp_f - payload.dew_point_temp_f
    fog_likelihood = (payload.relative_humidity / 100) * (1 / (abs(dew_point_spread) + 1))

    pressure_pa = payload.station_pressure * 3386.39
    temp_k = (payload.dry_bulb_temp_f - 32) * 5 / 9 + 273.15
    air_density = pressure_pa / (287.05 * temp_k)

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    return {
        "DRYBULBTEMPF": payload.dry_bulb_temp_f,
        "WETBULBTEMPF": payload.wet_bulb_temp_f,
        "DewPointTempF": payload.dew_point_temp_f,
        "RelativeHumidity": payload.relative_humidity,
        "WindSpeed": payload.wind_speed,
        "WindDirection": payload.wind_direction,
        "StationPressure": payload.station_pressure,
        "SeaLevelPressure": payload.sea_level_pressure,
        "Precip": payload.precip,
        "DewPointSpread": dew_point_spread,
        "FogLikelihood": fog_likelihood,
        "AirDensity": air_density,
        "Month_sin": month_sin,
        "Month_cos": month_cos,
        "Hour_sin": hour_sin,
        "Hour_cos": hour_cos,
    }, hour


# --------------------------------------------------------------------------
# Scene classification — drives the frontend's visual theme
# --------------------------------------------------------------------------
def classify_scene(payload: WeatherInput, visibility_km: float, hour: int):
    is_night = hour < 6 or hour >= 19
    heavy_precip = payload.precip >= 0.1
    light_precip = 0 < payload.precip < 0.1
    is_cold = payload.dry_bulb_temp_f <= 34
    dew_spread = payload.dry_bulb_temp_f - payload.dew_point_temp_f
    is_fog_prone = payload.relative_humidity >= 92 and abs(dew_spread) <= 3
    windy = payload.wind_speed >= 20

    if visibility_km <= 1.5 and (is_fog_prone or payload.relative_humidity >= 95):
        scene, label, severity = "fog", "Dense Fog", "severe"
    elif heavy_precip and is_cold:
        scene, label, severity = "snow", "Snowfall", "moderate"
    elif heavy_precip and windy:
        scene, label, severity = "storm", "Thunderstorm", "severe"
    elif heavy_precip:
        scene, label, severity = "rain", "Heavy Rain", "moderate"
    elif light_precip and is_cold:
        scene, label, severity = "snow", "Light Snow", "mild"
    elif light_precip:
        scene, label, severity = "rain", "Light Rain", "mild"
    elif payload.relative_humidity >= 80 and visibility_km < 6:
        scene, label, severity = "cloudy", "Overcast", "mild"
    elif payload.relative_humidity >= 65:
        scene, label, severity = "cloudy", "Partly Cloudy", "clear"
    else:
        scene, label, severity = "sunny", "Clear Skies", "clear"

    if is_night and scene in ("sunny", "cloudy"):
        scene = "clear-night" if scene == "sunny" else "cloudy-night"
        label = "Clear Night" if scene == "clear-night" else "Cloudy Night"

    return scene, label, severity


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: WeatherInput):
    bundle = get_bundle()
    model = bundle["model"]
    features_order = bundle["features"]

    features_dict, hour = build_features(payload)
    x = np.array([[features_dict[f] for f in features_order]])

    if bundle.get("uses_scaler"):
        x = bundle["scaler"].transform(x)

    pred_km = float(model.predict(x)[0])
    pred_km = max(0.0, min(pred_km, 15.0))  # clamp to sane sensor range

    scene, scene_label, severity = classify_scene(payload, pred_km, hour)

    return PredictionResponse(
        visibility_km=round(pred_km, 2),
        visibility_miles=round(pred_km * 0.621371, 2),
        scene=scene,
        scene_label=scene_label,
        severity=severity,
        confidence_note=f"Model: {bundle['model_name']} (test R²={bundle['metrics']['r2']:.2f})",
        model_used=bundle["model_name"],
    )


@app.get("/")
def root():
    return {
        "message": "Climate Visibility Prediction API",
        "docs": "/docs",
        "predict_endpoint": "/api/predict (POST)",
    }
