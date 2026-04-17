from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

DOWNLOADS_DIR = Path(r"C:\Users\santhosh\Downloads")


def _resolve_data_path(filename: str) -> Path:
    project_path = DATA_DIR / filename
    download_path = DOWNLOADS_DIR / filename
    return project_path if project_path.exists() else download_path


REAL_RECOMMENDATION_DATASET_PATH = Path(
    os.getenv(
        "REAL_RECOMMENDATION_DATASET_PATH",
        str(_resolve_data_path("real_crop_recommendation_13.csv")),
    )
)
REAL_PRODUCTION_DATASET_PATH = Path(
    os.getenv(
        "REAL_PRODUCTION_DATASET_PATH",
        str(_resolve_data_path("real_crop_production_13.csv")),
    )
)
REAL_MARKET_LOOKUP_PATH = Path(
    os.getenv(
        "REAL_MARKET_LOOKUP_PATH",
        str(_resolve_data_path("real_market_lookup_13.csv")),
    )
)
REAL_PROFILE_DATASET_PATH = Path(
    os.getenv(
        "REAL_PROFILE_DATASET_PATH",
        str(_resolve_data_path("real_crop_reference_profiles_13.csv")),
    )
)
REAL_MASTER_DATASET_PATH = Path(
    os.getenv(
        "REAL_MASTER_DATASET_PATH",
        str(_resolve_data_path("real_project_master_13.csv")),
    )
)

CLASSIFICATION_FEATURES = [
    "nitrogen",
    "phosphorous",
    "potassium",
    "temperature_c",
    "humidity",
    "ph",
    "rainfall_mm",
]

REGRESSION_NUMERIC_FEATURES = [
    "crop_year",
    "area",
    "ideal_nitrogen",
    "ideal_phosphorous",
    "ideal_potassium",
    "ideal_temperature_c",
    "ideal_humidity",
    "ideal_ph",
    "ideal_rainfall_mm",
    "price_per_ton",
]

REGRESSION_CATEGORICAL_FEATURES = [
    "crop",
    "state_name",
    "district_name",
    "season",
]

# Backward-compatible aliases for helper modules that still import the older names.
OPTIONAL_CATEGORICALS: list[str] = []
REGRESSION_FEATURES = REGRESSION_NUMERIC_FEATURES + REGRESSION_CATEGORICAL_FEATURES

TARGET_CROP = "crop"
TARGET_YIELD = "yield"
RANDOM_STATE = 42
TOP_K_DEFAULT = 3
RANKING_POOL_SIZE = 5
