# src/config.py
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

# Permite sobreescribir dataset desde variable de entorno
DATASET_PATH = os.getenv("DATASET_PATH", "")
