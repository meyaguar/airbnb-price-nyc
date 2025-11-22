# src/train.py
# --- bootstrap sys.path ---
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import DATA_PROCESSED, MODELS_DIR


def load_data():
    """
    Carga X_train e y_train desde data/processed.
    y_train viene en escala de precio real (USD).
    """
    X_train = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "y_train.parquet")["price"].values
    return X_train, y_train


def main():
    # 1) Cargar metadatos de features
    meta_path = MODELS_DIR / "feature_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # 2) Cargar datos de entrenamiento
    X_train, y_train_raw = load_data()

    # 3) Transformación del target: trabajamos en log(precio + 1)
    y_train = np.log1p(y_train_raw)

    # 4) Columnas categóricas y numéricas según metadata
    cat_cols = [c for c in meta["categorical"] if c in X_train.columns]
    num_like = [c for c in meta["numeric"] if c in X_train.columns]
    hi_card = [c for c in meta.get("hi_card_encoded", []) if c in X_train.columns]

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=False,
                ),
                cat_cols,
            ),
            ("num", "passthrough", num_like + hi_card),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # 5) Modelo: XGBoost sobre log1p(price)
    model = XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    print("🚀 Entrenando modelo XGBoost sobre log1p(price)…")
    pipe.fit(X_train, y_train)
    print("✅ Entrenamiento finalizado.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "model.joblib")
    print("💾 Modelo guardado en models/model.joblib (XGBoost + features extendidas).")


if __name__ == "__main__":
    main()
