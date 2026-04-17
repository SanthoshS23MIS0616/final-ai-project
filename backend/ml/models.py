from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier

from backend.ml.config import RANDOM_STATE

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:  # pragma: no cover
    TabNetClassifier = None


def create_tabnet_classifier(num_classes: int) -> Any:
    if TabNetClassifier is not None:
        return TabNetClassifier(
            seed=RANDOM_STATE,
            verbose=0,
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
        )

    return ExtraTreesClassifier(
        n_estimators=450,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def fit_tabnet_classifier(model: Any, X_train, y_train, X_valid, y_valid) -> Any:
    if TabNetClassifier is not None and isinstance(model, TabNetClassifier):
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=["valid"],
            eval_metric=["accuracy"],
            max_epochs=120,
            patience=20,
            batch_size=2048,
            virtual_batch_size=256,
        )
        return model

    model.fit(X_train, y_train)
    return model


def save_tabnet_model(model: Any, path: Path) -> dict[str, str]:
    if TabNetClassifier is not None and isinstance(model, TabNetClassifier):
        model.save_model(str(path.with_suffix("")))
        return {"format": "tabnet_zip", "path": f"{path.stem}.zip"}

    joblib.dump(model, path.with_suffix(".joblib"))
    return {"format": "joblib", "path": f"{path.stem}.joblib"}


def load_tabnet_model(path: Path, model_format: str) -> Any:
    if model_format == "tabnet_zip":
        if TabNetClassifier is None:
            raise ImportError("pytorch-tabnet is required to load the saved TabNet model.")
        model = TabNetClassifier()
        model.load_model(str(path.with_suffix(".zip")))
        return model

    return joblib.load(path.with_suffix(".joblib"))


def create_lgbm_classifier(num_classes: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def create_catboost_classifier(num_classes: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=500,
        learning_rate=0.05,
        depth=8,
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def create_yield_regressor() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )