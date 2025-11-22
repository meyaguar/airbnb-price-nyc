# api/main.py
from pathlib import Path
import sys
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- bootstrap sys.path -> acceder a src/*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR  # rutas
from src.schemas import PropertyInput, PredictionResponse
from src.utils_geo import distance_to_times_sq, distance_to_wall_st


app = FastAPI(title="NYC Airbnb Price API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes restringir a ["http://localhost:8501"] si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache: Dict[str, Any] = {"pipe": None}


def get_model():
    if _model_cache["pipe"] is None:
        _model_cache["pipe"] = joblib.load(MODELS_DIR / "model.joblib")
    return _model_cache["pipe"]


@app.get("/health")
def health():
    return {"status": "ok"}


def _add_rowwise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features fila a fila, coherentes con lo que hicimos en src.features:
      - dist_times_sq_km, dist_wall_st_km
      - log1p_minimum_nights, etc.
    Sin usar información del conjunto completo -> sin leakage.
    """
    # Asegurar tipos numéricos
    numeric_cols = [
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Distancias geográficas
    if {"latitude", "longitude"}.issubset(df.columns):
        df["dist_times_sq_km"] = df.apply(
            lambda r: distance_to_times_sq(r["latitude"], r["longitude"]), axis=1
        )
        df["dist_wall_st_km"] = df.apply(
            lambda r: distance_to_wall_st(r["latitude"], r["longitude"]), axis=1
        )

    # Caps suaves (igual que en features.py)
    clip_rules = [
        ("minimum_nights", 1, 60),
        ("number_of_reviews", 0, 2000),
        ("reviews_per_month", 0, 30),
        ("calculated_host_listings_count", 0, 200),
        ("availability_365", 0, 365),
    ]
    for col, lo, hi in clip_rules:
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Features logarítmicas
    for col in [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]:
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


@app.post("/properties/predict_price", response_model=PredictionResponse)
def predict_price(payload: PropertyInput):
    pipe = get_model()

    # Convertir payload en DataFrame
    X = pd.DataFrame([payload.dict()])

    # Añadir features derivadas (distancias y logs)
    X = _add_rowwise_features(X)

    price = float(pipe.predict(X)[0])
    return PredictionResponse(predicted_price=price, currency="USD")
