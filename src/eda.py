# src/eda.py
# --- bootstrap sys.path ---
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
from src.config import DATA_RAW, DATA_INTERIM, DATASET_PATH as ENV_DATASET_PATH


# ===== Alias de columnas del CSV original -> nombres canónicos =====
ALIASES = {
    # originales del dataset
    "id": "id",
    "name": "name",
    "host_id": "host_id",
    "host_identity_verified": "host_identity_verified",
    "host_name": "host_name",
    "neighbourhood_group": "neighbourhood_group",
    "neighbourhood": "neighbourhood",
    "lat": "latitude",
    "long": "longitude",
    "country": "country",
    "country_code": "country_code",
    "instant_bookable": "instant_bookable",
    "cancellation_policy": "cancellation_policy",
    "room_type": "room_type",
    "construction_year": "construction_year",
    "price": "price",
    "service_fee": "service_fee",
    "minimum_nights": "minimum_nights",
    "number_of_reviews": "number_of_reviews",
    "last_review": "last_review",
    "reviews_per_month": "reviews_per_month",
    "review_rate_number": "review_rate_number",
    "calculated_host_listings_count": "calculated_host_listings_count",
    "availability_365": "availability_365",
    "house_rules": "house_rules",
    "license": "license",

    # variantes con espacios / mayúsculas (por seguridad)
    "host id": "host_id",
    "host name": "host_name",
    "neighbourhood group": "neighbourhood_group",
    "neighborhood group": "neighbourhood_group",
    "neighborhood": "neighbourhood",
    "room type": "room_type",
    "construction year": "construction_year",
    "service fee": "service_fee",
    "availability 365": "availability_365",
    "review rate number": "review_rate_number",
}

# columnas que queremos conservar para pasar a la etapa de features
KEEP = [
    "id",
    "name",
    "host_id",
    "host_identity_verified",
    "host_name",
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "country",
    "country_code",
    "instant_bookable",
    "cancellation_policy",
    "room_type",
    "construction_year",
    "price",
    "service_fee",
    "minimum_nights",
    "number_of_reviews",
    "last_review",
    "reviews_per_month",
    "review_rate_number",
    "calculated_host_listings_count",
    "availability_365",
    "house_rules",
    "license",
]

# columnas mínimas sin las cuales no tiene sentido seguir
REQUIRED = ["latitude", "longitude", "room_type", "price"]

# correcciones específicas para neighbourhood_group detectadas en el EDA
NEIGH_GROUP_FIXES = {
    "brookln": "Brooklyn",
    "manhatan": "Manhattan",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas:
    - trim
    - minúsculas
    - espacios, guiones y barras -> "_"
    - aplica ALIASES
    """
    ren = {}
    for c in df.columns:
        k = (
            c.strip()
            .lower()
            .replace("-", " ")
            .replace("/", " ")
            .replace("__", "_")
        )
        k = " ".join(k.split())
        k = k.replace(" ", "_")
        ren[c] = ALIASES.get(k, k)
    return df.rename(columns=ren)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte tipos de datos a formatos útiles para el modelo.
    Maneja:
    - price / service_fee con símbolo $
    - lat / long
    - numéricos varios
    - fechas
    """
    # PRICE: quitar símbolos, comas, espacios
    for col in ["price", "service_fee"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[^\d\.\,\-]", "", regex=True)  # deja dígitos, punto, coma, signo
                .str.replace(",", "", regex=False)
                .replace("", np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # LAT/LON
    for c in ["latitude", "longitude"]:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
                .replace("", np.nan)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Otros numéricos
    num_cols = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "review_rate_number",
        "calculated_host_listings_count",
        "availability_365",
        "construction_year",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fechas
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Categóricas
    for c in [
        "room_type",
        "neighbourhood_group",
        "neighbourhood",
        "country",
        "country_code",
        "instant_bookable",
        "cancellation_policy",
        "host_identity_verified",
        "license",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"": np.nan})

    # Textos libres
    for c in ["name", "house_rules"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    return df


def clean_neighbourhood_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige typos específicos en neighbourhood_group detectados en el dataset
    (por ejemplo, 'brookln' -> 'Brooklyn', 'manhatan' -> 'Manhattan').
    No modifica la lógica de tratamiento de NaN ni el resto del pipeline.
    """
    if "neighbourhood_group" not in df.columns:
        return df

    s = df["neighbourhood_group"].astype(str).str.strip()
    s = s.replace(NEIGH_GROUP_FIXES)
    df["neighbourhood_group"] = s
    return df


def _apply_filters(df: pd.DataFrame, bounds: dict, verbose: bool) -> pd.DataFrame:
    """Filtros razonables y clipping."""
    if verbose:
        print(f"🔧 Filtros: {bounds}")

    # dropna básicas
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude", "room_type", "price"])
    if verbose:
        print(f"➡️  dropna(required): {before} -> {len(df)}")

    # price
    lo, hi = bounds["price"]
    before = len(df)
    df = df[(df["price"] > lo) & (df["price"] < hi)]
    if verbose:
        print(f"➡️  price in ({lo},{hi}): {before} -> {len(df)}")

    # minimum_nights
    if "minimum_nights" in df.columns:
        lo, hi = bounds["minimum_nights"]
        before = len(df)
        df = df[(df["minimum_nights"] >= lo) & (df["minimum_nights"] <= hi)]
        if verbose:
            print(f"➡️  minimum_nights in [{lo},{hi}]: {before} -> {len(df)}")

    # reviews_per_month: NA -> 0
    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # clips suaves
    for c, (lo, hi) in bounds.get("clips", {}).items():
        if c in df.columns:
            before_na = df[c].isna().sum()
            df[c] = df[c].clip(lo, hi)
            if verbose:
                print(f"➡️  clip {c} to [{lo},{hi}] (n_na antes={before_na})")

    return df


def quick_clean(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f"🧾 Columnas originales: {list(df.columns)[:20]} ... (total {len(df.columns)})")

    df = normalize_columns(df)
    df = coerce_types(df)
    df = clean_neighbourhood_group(df)

    if verbose:
        print(f"🧭 Columnas normalizadas: {sorted(df.columns.tolist())}")

    # Comprobar columnas requeridas
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print("⚠️  Columnas requeridas no encontradas:", missing)
        print("🔎 Columnas disponibles tras normalización:")
        print(sorted(df.columns.tolist()))
        raise KeyError(f"Faltan columnas requeridas: {missing}")

    # Mantener columnas relevantes
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].copy()

    if verbose:
        print(f"📊 Filas iniciales tras selección KEEP: {len(df)}")

    # Filtros estándar centrados en Airbnb típico
    std_bounds = {
        "price": (10, 800),
        "minimum_nights": (1, 60),
        "clips": {
            "number_of_reviews": (0, 1000),
            "availability_365": (0, 365),
            "minimum_nights": (1, 30),
        },
    }
    df_std = _apply_filters(df.copy(), std_bounds, verbose)

    # Si quedan pocas filas, relajar filtros
    if len(df_std) < 1000:
        if verbose:
            print(
                f"⚠️  Muy pocas filas con filtros estándar ({len(df_std)}). "
                f"Reintentando con filtros relajados…"
            )
        soft_bounds = {
            "price": (5, 1500),
            "minimum_nights": (1, 365),
            "clips": {
                "number_of_reviews": (0, 5000),
                "availability_365": (0, 366),
                "minimum_nights": (1, 365),
            },
        }
        df_soft = _apply_filters(df.copy(), soft_bounds, verbose)
        if len(df_soft) > len(df_std):
            if verbose:
                print(f"✅ Usando filtros relajados: {len(df_soft)} filas (vs {len(df_std)} estándar)")
            return df_soft

    return df_std


def _read_csv_safely(src: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(src, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"No se pudo leer {src} con codificaciones estándar.")


def main(input_path: str, sample: float, quick: bool, verbose: bool):
    # Resolver archivo origen
    if input_path:
        src = Path(input_path)
    elif ENV_DATASET_PATH:
        src = Path(ENV_DATASET_PATH)
    else:
        candidates = list(DATA_RAW.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError("No CSV. Proporciona --input o DATASET_PATH.")
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
        src = candidates[0]

    if not src.exists():
        raise FileNotFoundError(f"No existe el archivo: {src}")

    print(f"📥 Leyendo CSV desde: {src}")
    df = _read_csv_safely(src)

    # Muestreo opcional para desarrollo
    if sample and 0 < sample < 1.0 and len(df) > 50_000:
        df = df.sample(frac=sample, random_state=42).reset_index(drop=True)
        print(f"🔎 Muestreo aplicado: {sample:.0%} ({len(df):,} filas)")

    if quick:
        df = quick_clean(df, verbose=verbose)

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    outp = DATA_INTERIM / "listings_clean.parquet"
    df.to_parquet(outp, index=False)
    print(f"✅ Guardado: {outp} ({len(df):,} filas)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--sample", type=float, default=0.0)
    ap.add_argument("--quick-clean", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="Imprime conteos por etapa")
    args = ap.parse_args()
    main(args.input, args.sample, args.quick_clean, args.verbose)
