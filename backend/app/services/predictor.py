from __future__ import annotations

from functools import lru_cache

from backend.ml.config_realistic_v2 import MODEL_DIR
from backend.ml.inference_realistic_v2 import InferenceEngine


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
