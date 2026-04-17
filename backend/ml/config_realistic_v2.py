from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "realistic_v2"
MODEL_DIR = PROJECT_ROOT / "models" / "realistic_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "realistic_v2"

CROP_RECOMMENDATION_DATASET_PATH = DATA_DIR / "crop_recommendation_realistic.csv"
CROP_PRODUCTION_DATASET_PATH = DATA_DIR / "crop_production_clean.csv"
CROP_PROFILES_DATASET_PATH = DATA_DIR / "crop_profiles_ranges.csv"
MARKET_PRICES_DATASET_PATH = DATA_DIR / "market_prices_realistic.csv"
PROJECT_MASTER_DATASET_PATH = DATA_DIR / "project_master_clean.csv"

CLASSIFICATION_NUMERIC_FEATURES = [
    "nitrogen",
    "phosphorous",
    "potassium",
    "temperature_c",
    "humidity",
    "ph",
    "rainfall_mm",
    "moisture",
]

CLASSIFICATION_CATEGORICAL_FEATURES = [
    "state_name",
    "season",
]

REGRESSION_NUMERIC_FEATURES = [
    "area_ha",
    "nitrogen",
    "phosphorous",
    "potassium",
    "temperature_c",
    "humidity",
    "ph",
    "rainfall_mm",
    "moisture",
]

REGRESSION_CATEGORICAL_FEATURES = [
    "crop",
    "state_name",
    "district_name",
    "season",
]

TARGET_CROP = "crop"
TARGET_YIELD = "target_yield_t_ha"
RANDOM_STATE = 42
