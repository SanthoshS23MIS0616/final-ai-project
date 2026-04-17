from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, top_k_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from backend.ml.config_realistic_v2 import (
    CLASSIFICATION_CATEGORICAL_FEATURES,
    CLASSIFICATION_NUMERIC_FEATURES,
    CROP_PROFILES_DATASET_PATH,
    CROP_RECOMMENDATION_DATASET_PATH,
    MARKET_PRICES_DATASET_PATH,
    MODEL_DIR,
    PROJECT_MASTER_DATASET_PATH,
    RANDOM_STATE,
    REGRESSION_CATEGORICAL_FEATURES,
    REGRESSION_NUMERIC_FEATURES,
    REPORT_DIR,
    TARGET_CROP,
    TARGET_YIELD,
)
from backend.ml.models import create_tabnet_classifier, fit_tabnet_classifier, save_tabnet_model


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


def _to_numpy(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def _classification_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                CLASSIFICATION_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CLASSIFICATION_CATEGORICAL_FEATURES,
            ),
        ]
    )


def _regression_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                REGRESSION_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                REGRESSION_CATEGORICAL_FEATURES,
            ),
        ]
    )


def _top3_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.arange(probabilities.shape[1])
    k = min(3, probabilities.shape[1])
    return float(top_k_accuracy_score(y_true, probabilities, k=k, labels=labels))


def _classification_summary(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "top3_accuracy": _top3_score(y_true, probabilities),
    }


def _regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse, "r2": float(r2_score(y_true, y_pred))}


def _classification_cv(name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracy_scores: list[float] = []
    f1_scores: list[float] = []
    top3_scores: list[float] = []

    for train_idx, valid_idx in splitter.split(X_train, y_train):
        model = clone(pipeline)
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train[train_idx]
        y_fold_valid = y_train[valid_idx]
        model.fit(X_fold_train, y_fold_train)
        probabilities = np.asarray(model.predict_proba(X_fold_valid))
        predictions = np.asarray(model.predict(X_fold_valid))
        accuracy_scores.append(accuracy_score(y_fold_valid, predictions))
        f1_scores.append(f1_score(y_fold_valid, predictions, average="weighted"))
        top3_scores.append(_top3_score(y_fold_valid, probabilities))

    return {
        "accuracy_mean": float(np.mean(accuracy_scores)),
        "accuracy_std": float(np.std(accuracy_scores, ddof=0)),
        "f1_weighted_mean": float(np.mean(f1_scores)),
        "f1_weighted_std": float(np.std(f1_scores, ddof=0)),
        "top3_accuracy_mean": float(np.mean(top3_scores)),
        "top3_accuracy_std": float(np.std(top3_scores, ddof=0)),
        "folds": 5,
        "model": name,
    }


def _regression_cv(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=splitter,
        scoring={"neg_rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1,
    )
    return {
        "rmse_mean": float(np.mean(-scores["test_neg_rmse"])),
        "rmse_std": float(np.std(-scores["test_neg_rmse"], ddof=0)),
        "r2_mean": float(np.mean(scores["test_r2"])),
        "r2_std": float(np.std(scores["test_r2"], ddof=0)),
        "folds": 5,
    }


def _warning_from_gap(label: str, train_value: float, test_value: float, threshold: float, warnings_list: list[str]) -> None:
    if abs(train_value - test_value) > threshold:
        warnings_list.append(
            f"{label} train/test gap is {abs(train_value - test_value):.4f}, which may indicate overfitting."
        )


def _sample_by_class(df: pd.DataFrame, label_column: str, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    fractions = (
        df[label_column]
        .value_counts(normalize=True)
        .mul(max_rows)
        .round()
        .clip(lower=1)
        .astype(int)
        .to_dict()
    )
    parts = []
    for label, frame in df.groupby(label_column, sort=False):
        take = min(len(frame), int(fractions.get(label, 1)))
        parts.append(frame.sample(n=take, random_state=RANDOM_STATE))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=RANDOM_STATE)
    return sampled.reset_index(drop=True)


def _compute_shap(lightgbm_pipeline: Pipeline, catboost_pipeline: Pipeline, X_train: pd.DataFrame, report_dir: Path) -> dict[str, Any]:
    import matplotlib
    import shap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _collapse(values: Any, sample_array: np.ndarray) -> np.ndarray:
        rows, cols = sample_array.shape
        if isinstance(values, list):
            stacked = np.stack([np.asarray(item, dtype=float) for item in values], axis=0)
            return np.mean(np.abs(stacked), axis=0)
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if arr.shape[0] == rows and arr.shape[1] == cols:
                return np.mean(np.abs(arr), axis=2)
            if arr.shape[1] == rows and arr.shape[2] == cols:
                return np.mean(np.abs(arr), axis=0)
        raise ValueError(f"Unsupported SHAP output shape: {np.asarray(values).shape}")

    def _plot(values: np.ndarray, sample_array: np.ndarray, names: list[str], path: Path) -> None:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(values, sample_array, feature_names=names, plot_type="bar", max_display=min(20, len(names)), show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()

    transformed = _to_numpy(lightgbm_pipeline.named_steps["preprocess"].transform(X_train))
    feature_names = list(lightgbm_pipeline.named_steps["preprocess"].get_feature_names_out())
    sample = transformed[: min(len(transformed), 250)]

    lgbm_values = _collapse(shap.TreeExplainer(lightgbm_pipeline.named_steps["model"]).shap_values(sample), sample)
    cat_transformed = _to_numpy(catboost_pipeline.named_steps["preprocess"].transform(X_train))
    cat_sample = cat_transformed[: min(len(cat_transformed), 250)]
    cat_values = _collapse(shap.TreeExplainer(catboost_pipeline.named_steps["model"]).shap_values(cat_sample), cat_sample)

    lgbm_plot = report_dir / "shap_summary_lightgbm.png"
    cat_plot = report_dir / "shap_summary_catboost.png"
    _plot(lgbm_values, sample, feature_names, lgbm_plot)
    _plot(cat_values, cat_sample, feature_names, cat_plot)

    lgbm_importance = np.mean(np.abs(lgbm_values), axis=0)
    cat_importance = np.mean(np.abs(cat_values), axis=0)
    ranking = sorted(
        (
            {
                "feature": feature_names[idx],
                "lightgbm_importance": float(lgbm_importance[idx]),
                "catboost_importance": float(cat_importance[idx]),
                "mean_importance": float((lgbm_importance[idx] + cat_importance[idx]) / 2.0),
            }
            for idx in range(len(feature_names))
        ),
        key=lambda item: item["mean_importance"],
        reverse=True,
    )
    summary = {
        "status": "ok",
        "lightgbm_summary_plot": str(lgbm_plot),
        "catboost_summary_plot": str(cat_plot),
        "top_features": ranking[:20],
    }
    (report_dir / "shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_and_save() -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    recommendation_df = pd.read_csv(CROP_RECOMMENDATION_DATASET_PATH).reset_index(drop=True)
    master_df = pd.read_csv(PROJECT_MASTER_DATASET_PATH).drop_duplicates().reset_index(drop=True)
    profiles_df = pd.read_csv(CROP_PROFILES_DATASET_PATH)
    market_df = pd.read_csv(MARKET_PRICES_DATASET_PATH)

    recommendation_df = _sample_by_class(recommendation_df, TARGET_CROP, 9000)
    master_df = _sample_by_class(master_df, TARGET_CROP, 12000)

    label_encoder = LabelEncoder()
    y_cls = label_encoder.fit_transform(recommendation_df[TARGET_CROP].astype(str))
    X_cls = recommendation_df[CLASSIFICATION_NUMERIC_FEATURES + CLASSIFICATION_CATEGORICAL_FEATURES]
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls,
        y_cls,
        test_size=0.2,
        stratify=y_cls,
        random_state=RANDOM_STATE,
    )

    class_pre = _classification_preprocessor()
    num_classes = len(label_encoder.classes_)
    classifier_pipelines = {
        "lightgbm": Pipeline(
            [
                ("preprocess", clone(class_pre)),
                (
                    "model",
                    LGBMClassifier(
                        objective="multiclass",
                        num_class=num_classes,
                        n_estimators=80,
                        learning_rate=0.05,
                        max_depth=6,
                        num_leaves=28,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_alpha=0.3,
                        reg_lambda=0.5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "catboost": Pipeline(
            [
                ("preprocess", clone(class_pre)),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=80,
                        learning_rate=0.05,
                        depth=5,
                        random_seed=RANDOM_STATE,
                        loss_function="MultiClass",
                        verbose=False,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", clone(class_pre)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=60,
                        max_depth=10,
                        min_samples_leaf=8,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", clone(class_pre)),
                ("model", LogisticRegression(max_iter=3000, C=0.85, random_state=RANDOM_STATE)),
            ]
        ),
    }

    stacking_pipeline = Pipeline(
        [
            ("preprocess", clone(class_pre)),
            (
                "model",
                StackingClassifier(
                    estimators=[
                        (
                            "lightgbm",
                            LGBMClassifier(
                                objective="multiclass",
                                num_class=num_classes,
                                n_estimators=60,
                                learning_rate=0.05,
                                max_depth=6,
                                num_leaves=28,
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                                verbosity=-1,
                            ),
                        ),
                        (
                            "catboost",
                            CatBoostClassifier(
                                iterations=60,
                                learning_rate=0.05,
                                depth=5,
                                random_seed=RANDOM_STATE,
                                loss_function="MultiClass",
                                verbose=False,
                            ),
                        ),
                        (
                            "random_forest",
                            RandomForestClassifier(
                                n_estimators=50,
                                max_depth=10,
                                min_samples_leaf=8,
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        ),
                    ],
                    final_estimator=LogisticRegression(max_iter=3000, C=0.8, random_state=RANDOM_STATE),
                    stack_method="predict_proba",
                    cv=5,
                    passthrough=False,
                ),
            ),
        ]
    )
    classifier_pipelines["stacking"] = stacking_pipeline

    classification_metrics: dict[str, dict[str, float]] = {}
    classification_cv_metrics: dict[str, dict[str, Any]] = {}
    warnings_list: list[str] = []
    fitted_lgbm: Pipeline | None = None
    fitted_catboost: Pipeline | None = None

    for name, pipeline in classifier_pipelines.items():
        classification_cv_metrics[name] = _classification_cv(name, pipeline, X_train_cls, y_train_cls)
        fitted = clone(pipeline)
        fitted.fit(X_train_cls, y_train_cls)

        train_prob = np.asarray(fitted.predict_proba(X_train_cls))
        test_prob = np.asarray(fitted.predict_proba(X_test_cls))
        train_pred = np.asarray(fitted.predict(X_train_cls))
        test_pred = np.asarray(fitted.predict(X_test_cls))
        train_summary = _classification_summary(y_train_cls, train_pred, train_prob)
        test_summary = _classification_summary(y_test_cls, test_pred, test_prob)
        classification_metrics[name] = {f"train_{k}": v for k, v in train_summary.items()} | test_summary
        _warning_from_gap(f"{name} accuracy", train_summary["accuracy"], test_summary["accuracy"], 0.06, warnings_list)
        if abs(classification_cv_metrics[name]["accuracy_mean"] - test_summary["accuracy"]) > 0.05:
            warnings_list.append(
                f"{name} CV accuracy ({classification_cv_metrics[name]['accuracy_mean']:.4f}) differs from test accuracy ({test_summary['accuracy']:.4f})."
            )

        if name == "lightgbm":
            fitted_lgbm = fitted
            joblib.dump(fitted.named_steps["model"], MODEL_DIR / "lgbm_model.joblib")
            joblib.dump(fitted.named_steps["preprocess"], MODEL_DIR / "classification_preprocessor.joblib")
        elif name == "catboost":
            fitted_catboost = fitted
            joblib.dump(fitted.named_steps["model"], MODEL_DIR / "catboost_model.joblib")
        elif name == "random_forest":
            joblib.dump(fitted, MODEL_DIR / "random_forest_model.joblib")
        elif name == "logistic_regression":
            joblib.dump(fitted, MODEL_DIR / "logistic_regression_model.joblib")
        elif name == "stacking":
            joblib.dump(fitted.named_steps["model"], MODEL_DIR / "stacking_model.joblib")

    classification_cv_metrics["tabnet"] = {"status": "skipped", "reason": "omitted_in_realistic_v2_for_runtime_efficiency"}

    X_reg = master_df[REGRESSION_NUMERIC_FEATURES + REGRESSION_CATEGORICAL_FEATURES]
    y_reg = master_df[TARGET_YIELD].astype(float)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    reg_pipeline = Pipeline(
        [
            ("preprocess", _regression_preprocessor()),
            (
                "model",
                LGBMRegressor(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.2,
                    reg_lambda=0.4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    regression_cv_metrics = _regression_cv(reg_pipeline, X_train_reg, y_train_reg)
    reg_pipeline.fit(X_train_reg, y_train_reg)
    train_pred = reg_pipeline.predict(X_train_reg)
    test_pred = reg_pipeline.predict(X_test_reg)
    regression_metrics = {f"train_{k}": v for k, v in _regression_summary(y_train_reg.to_numpy(), train_pred).items()} | _regression_summary(y_test_reg.to_numpy(), test_pred)
    _warning_from_gap("yield r2", regression_metrics["train_r2"], regression_metrics["r2"], 0.10, warnings_list)
    if abs(regression_cv_metrics["r2_mean"] - regression_metrics["r2"]) > 0.08:
        warnings_list.append(
            f"Regression CV R² ({regression_cv_metrics['r2_mean']:.4f}) differs from test R² ({regression_metrics['r2']:.4f})."
        )

    joblib.dump(reg_pipeline.named_steps["model"], MODEL_DIR / "yield_model.joblib")
    joblib.dump(reg_pipeline.named_steps["preprocess"], MODEL_DIR / "regression_preprocessor.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "crop_label_encoder.joblib")

    if fitted_lgbm is None or fitted_catboost is None:
        raise RuntimeError("LightGBM and CatBoost must be fitted before SHAP analysis.")
    shap_summary = _compute_shap(fitted_lgbm, fitted_catboost, X_train_cls, REPORT_DIR)

    best_classifier = max(
        {name: metrics["f1_weighted"] for name, metrics in classification_metrics.items() if "f1_weighted" in metrics},
        key=lambda key: classification_metrics[key]["f1_weighted"],
    )

    training_report = {
        "data_mode": "realistic_v2_leakage_safe_pipeline",
        "dataset_summary": {
            "classification_rows": int(len(recommendation_df)),
            "regression_rows": int(len(master_df)),
            "crop_count": int(master_df["crop"].nunique()),
            "state_count": int(master_df["state_name"].nunique()),
        },
        "classification_metrics": classification_metrics,
        "classification_cv_metrics": classification_cv_metrics,
        "regression_metrics": regression_metrics,
        "regression_cv_metrics": regression_cv_metrics,
        "model_comparison": {
            "best_classification_model": best_classifier,
            "best_regression_model": "lightgbm_regressor",
        },
        "shap_summary": shap_summary,
        "warnings": sorted(set(warnings_list)),
        "artifacts": {
            "lgbm_model": "lgbm_model.joblib",
            "catboost_model": "catboost_model.joblib",
            "random_forest_model": "random_forest_model.joblib",
            "logistic_regression_model": "logistic_regression_model.joblib",
            "yield_model": "yield_model.joblib",
            "stacking_model": "stacking_model.joblib",
        },
    }

    runtime_metadata = {
        "training_report": training_report,
        "profiles_rows": int(len(profiles_df)),
        "market_rows": int(len(market_df)),
        "feature_sets": {
            "classification_numeric": CLASSIFICATION_NUMERIC_FEATURES,
            "classification_categorical": CLASSIFICATION_CATEGORICAL_FEATURES,
            "regression_numeric": REGRESSION_NUMERIC_FEATURES,
            "regression_categorical": REGRESSION_CATEGORICAL_FEATURES,
        },
    }
    (MODEL_DIR / "runtime_metadata.json").write_text(json.dumps(runtime_metadata, indent=2), encoding="utf-8")
    (REPORT_DIR / "training_metrics.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    return training_report


def main() -> None:
    print(json.dumps(train_and_save(), indent=2))


if __name__ == "__main__":
    main()
