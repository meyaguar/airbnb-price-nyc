# src/features.py
# --- bootstrap sys.path ---
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_INTERIM, DATA_PROCESSED, MODELS_DIR
from src.utils_geo import distance_to_times_sq, distance_to_wall_st

# Columnas que usaremos para entrenar (las mismas que puede dar el dashboard/API)
BASE_FEATURES: List[str] = [
    "neighbourhood_group",
    "room_type",
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

# Features numéricas derivadas (las construiremos aquí y también en la API)
LOG_FEATURES_SRC = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]


def _add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega distancias en km a Times Sq y Wall St a partir de lat/lon."""
    df["dist_times_sq_km"] = df.apply(
        lambda r: distance_to_times_sq(r["latitude"], r["longitude"]), axis=1
    )
    df["dist_wall_st_km"] = df.apply(
        lambda r: distance_to_wall_st(r["latitude"], r["longitude"]), axis=1
    )
    return df


def _clip_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica límites razonables para controlar outliers."""
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
    return df


def _add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables logarítmicas (log1p) de numéricas sesgadas."""
    for col in LOG_FEATURES_SRC:
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))
    return df


def build_features():
    """Lee el parquet limpio, construye features y guarda train/test + metadata."""
    src = DATA_INTERIM / "listings_clean.parquet"
    if not src.exists():
        raise FileNotFoundError("Ejecuta primero: `python -m src.eda --quick-clean`")

    print(f"📥 Cargando dataset limpio: {src}")
    df = pd.read_parquet(src)

    # Verificar que existe la columna objetivo
    if "price" not in df.columns:
        raise KeyError("La columna 'price' no existe en listings_clean.parquet.")

    # Asegurar que las columnas base existen
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas para features: {missing}")

    # Nos quedamos con columnas que necesitamos + target
    cols_keep = BASE_FEATURES + ["price"]
    df = df[cols_keep].copy()

    # Limpiar NaN esenciales
    df = df.dropna(subset=["latitude", "longitude", "room_type", "neighbourhood_group", "price"])
    print(f"📊 Filas después de dropna esenciales: {len(df):,}")

    # Caps numéricos
    df = _clip_numeric(df)

    # Features geográficas
    print("🗺️ Calculando distancias geográficas…")
    df = _add_geo_features(df)

    # Features logarítmicas
    df = _add_log_features(df)

    # Objetivo y matriz de features
    y = df["price"].astype(float)
    drop_cols = ["price"]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Columnas categóricas y numéricas
    cat_cols = [c for c in ["neighbourhood_group", "room_type"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Split train/test (intentamos estratificar por room_type)
    strat = X["room_type"] if "room_type" in X.columns else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=strat
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

    # Guardar datasets procesados
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(DATA_PROCESSED / "X_train.parquet", index=False)
    X_test.to_parquet(DATA_PROCESSED / "X_test.parquet", index=False)
    y_train.to_frame("price").to_parquet(DATA_PROCESSED / "y_train.parquet", index=False)
    y_test.to_frame("price").to_parquet(DATA_PROCESSED / "y_test.parquet", index=False)

    # Guardar metadata de columnas
    feature_meta = {
        "numeric": num_cols,
        "categorical": cat_cols,
        "target": "price",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "feature_metadata.json").write_text(
        json.dumps(feature_meta, indent=2)
    )

    print("✅ Features construidas y guardadas en data/processed/.")


if __name__ == "__main__":
    build_features()
