# src/evaluate.py
# --- bootstrap sys.path ---
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.config import DATA_PROCESSED, MODELS_DIR


def load_data():
    X_train = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "y_train.parquet")["price"].values
    return X_train, y_train


def main():
    meta_path = MODELS_DIR / "feature_metadata.json"
    meta = json.loads(meta_path.read_text())

    X_train, y_train = load_data()

    cat_cols = [c for c in meta["categorical"] if c in X_train.columns]
    num_cols = [c for c in meta["numeric"] if c in X_train.columns]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Modelo más interpretable y menos propenso a sobreajuste que XGBoost sin control
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "model.joblib")
    print("✅ Modelo entrenado y guardado en models/model.joblib")

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics_values = json.load(f)

        print("📊 Métricas del modelo:")
        for name, value in metrics_values.items():
            print(f"  {name:12}: {value}")
    else:
        print("⚠️ No se encontró models/metrics.json")
    
    
if __name__ == "__main__":
    main()
