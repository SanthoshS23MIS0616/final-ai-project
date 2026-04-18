from __future__ import annotations

from functools import lru_cache

from backend.ml.config_realistic_v2 import MODEL_DIR
from backend.ml.inference_realistic_v2 import InferenceEngine


REQUIRED_MODEL_ARTIFACTS = (
    "runtime_metadata.json",
    "classification_preprocessor.joblib",
    "regression_preprocessor.joblib",
    "crop_label_encoder.joblib",
    "lgbm_model.joblib",
    "catboost_model.joblib",
    "yield_model.joblib",
    "stacking_model.joblib",
)


def model_artifact_status() -> dict[str, object]:
    missing = [name for name in REQUIRED_MODEL_ARTIFACTS if not (MODEL_DIR / name).exists()]
    return {"ready": not missing, "missing_artifacts": missing}


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    return InferenceEngine.from_artifacts(MODEL_DIR)


def reload_engine() -> InferenceEngine:
    get_engine.cache_clear()
    return get_engine()


def train_models(data_dir: str | None = None) -> dict:
    del data_dir
    from backend.ml.training_realistic_v2 import train_and_save

    report = train_and_save()
    reload_engine()
    return report
