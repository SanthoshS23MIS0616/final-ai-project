from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.ml.config_realistic_v2 import (
    CLASSIFICATION_CATEGORICAL_FEATURES,
    CLASSIFICATION_NUMERIC_FEATURES,
    CROP_PROFILES_DATASET_PATH,
    MARKET_PRICES_DATASET_PATH,
    MODEL_DIR,
    PROJECT_MASTER_DATASET_PATH,
    REPORT_DIR,
    REGRESSION_CATEGORICAL_FEATURES,
    REGRESSION_NUMERIC_FEATURES,
)
from backend.ml.crop_rules_realistic_v2 import (
    CROP_DATABASE,
    current_analysis_context,
    derive_season,
    harvest_month,
    late_sowing_penalty,
    month_name,
    water_need_factor,
)

TOP_K_DEFAULT = 5
RANKING_POOL_SIZE = 13


@dataclass
class ModelBundle:
    classification_preprocessor: Any
    regression_preprocessor: Any
    label_encoder: Any
    lgbm_model: Any
    catboost_model: Any
    yield_model: Any
    stacking_model: Any
    metadata: dict[str, Any]


def _to_float32_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _candidate_scores(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _clip(value: float, minimum: float, maximum: float) -> float:
    return float(min(max(value, minimum), maximum))


def _report_url(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        relative = path.relative_to(REPORT_DIR.parent)
    except ValueError:
        relative = Path("realistic_v2") / path.name
    return "/reports/" + relative.as_posix()


def _extract_class_shap(values: Any, class_index: int) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[class_index], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0, :, class_index].reshape(1, -1)
        if arr.shape[1] == 1:
            return arr[class_index, 0, :].reshape(1, -1)
        if arr.shape[0] > class_index:
            return arr[class_index]
        return arr[:, :, class_index]
    raise ValueError(f"Unsupported SHAP class output shape: {arr.shape}")


def _build_metadata(runtime_metadata: dict[str, Any], master_df: pd.DataFrame, profiles_df: pd.DataFrame, market_df: pd.DataFrame) -> dict[str, Any]:
    defaults = {
        "nitrogen": float(master_df["nitrogen"].median()),
        "phosphorous": float(master_df["phosphorous"].median()),
        "potassium": float(master_df["potassium"].median()),
        "temperature_c": float(master_df["temperature_c"].median()),
        "humidity": float(master_df["humidity"].median()),
        "ph": float(master_df["ph"].median()),
        "rainfall_mm": float(master_df["rainfall_mm"].median()),
        "moisture": float(master_df["moisture"].median()),
        "area": float(master_df["area_ha"].median()),
        "season": str(master_df["season"].mode().iloc[0]),
        "state_name": str(master_df["state_name"].mode().iloc[0]),
        "district_name": str(master_df["district_name"].mode().iloc[0]),
        "crop_year": int(master_df["crop_year"].median()),
        "price_per_ton": float(master_df["price_per_ton"].median()),
    }

    climate_ranges = {
        "temperature_c": {"min": float(master_df["temperature_c"].min()), "max": float(master_df["temperature_c"].max())},
        "rainfall_mm": {"min": float(master_df["rainfall_mm"].min()), "max": float(master_df["rainfall_mm"].max())},
        "humidity": {"min": float(master_df["humidity"].min()), "max": float(master_df["humidity"].max())},
        "ph": {"min": float(master_df["ph"].min()), "max": float(master_df["ph"].max())},
    }

    crop_profiles: dict[str, dict[str, Any]] = {}
    for crop_name, frame in profiles_df.groupby("crop", sort=False):
        typical = frame.loc[frame["profile_variant"] == "typical"]
        row = typical.iloc[0] if not typical.empty else frame.iloc[0]
        crop_market = market_df.loc[market_df["crop"] == crop_name, "price_per_ton"]
        crop_yield = master_df.loc[master_df["crop"] == crop_name, "target_yield_t_ha"]
        crop_key = str(crop_name).strip().lower()
        crop_rules = CROP_DATABASE.get(crop_key, {})
        crop_profiles[str(crop_name)] = {
            "min_nitrogen": float(row["min_nitrogen"]),
            "max_nitrogen": float(row["max_nitrogen"]),
            "min_phosphorous": float(row["min_phosphorous"]),
            "max_phosphorous": float(row["max_phosphorous"]),
            "min_potassium": float(row["min_potassium"]),
            "max_potassium": float(row["max_potassium"]),
            "min_temp_c": float(row["min_temp_c"]),
            "max_temp_c": float(row["max_temp_c"]),
            "min_humidity": float(row["min_humidity"]),
            "max_humidity": float(row["max_humidity"]),
            "min_ph": float(row["min_ph"]),
            "max_ph": float(row["max_ph"]),
            "min_rainfall_mm": float(row["min_rainfall_mm"]),
            "max_rainfall_mm": float(row["max_rainfall_mm"]),
            "price_per_ton": float(crop_market.median()) if not crop_market.empty else defaults["price_per_ton"],
            "market_source": "market_prices_realistic.csv",
            "market_type": "state_season_realistic_market",
            "historical_median_yield": float(crop_yield.median()) if not crop_yield.empty else None,
            "duration_months": int(crop_rules.get("duration_months", 6)),
            "sowing_months": list(crop_rules.get("sowing_months", [6, 7, 8])),
            "peak_harvest_months": list(crop_rules.get("peak_harvest_months", [10, 11])),
        }

    shap_summary = runtime_metadata["training_report"].get("shap_summary", {})

    return {
        "dataset_summary": {
            "recommendation_rows": int(runtime_metadata["training_report"]["dataset_summary"]["classification_rows"]),
            "production_rows": int(runtime_metadata["training_report"]["dataset_summary"]["regression_rows"]),
            "crop_count": int(master_df["crop"].nunique()),
            "default_inputs": defaults,
        },
        "training_report": runtime_metadata["training_report"],
        "crop_profiles": crop_profiles,
        "climate_ranges": climate_ranges,
        "ranking_pool_size": RANKING_POOL_SIZE,
        "market_prices": market_df.to_dict(orient="records"),
        "xai_assets": {
            "lightgbm_summary_plot": _report_url(shap_summary.get("lightgbm_summary_plot")),
            "catboost_summary_plot": _report_url(shap_summary.get("catboost_summary_plot")),
            "top_features": shap_summary.get("top_features", []),
        },
    }


def load_model_bundle(model_dir: Path | None = None) -> ModelBundle:
    model_dir = Path(model_dir or MODEL_DIR)
    runtime_metadata = json.loads((model_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    master_df = pd.read_csv(PROJECT_MASTER_DATASET_PATH)
    profiles_df = pd.read_csv(CROP_PROFILES_DATASET_PATH)
    market_df = pd.read_csv(MARKET_PRICES_DATASET_PATH)
    metadata = _build_metadata(runtime_metadata, master_df, profiles_df, market_df)

    return ModelBundle(
        classification_preprocessor=joblib.load(model_dir / "classification_preprocessor.joblib"),
        regression_preprocessor=joblib.load(model_dir / "regression_preprocessor.joblib"),
        label_encoder=joblib.load(model_dir / "crop_label_encoder.joblib"),
        lgbm_model=joblib.load(model_dir / "lgbm_model.joblib"),
        catboost_model=joblib.load(model_dir / "catboost_model.joblib"),
        yield_model=joblib.load(model_dir / "yield_model.joblib"),
        stacking_model=joblib.load(model_dir / "stacking_model.joblib"),
        metadata=metadata,
    )


class InferenceEngine:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self._explainers: dict[str, Any] = {}

    @classmethod
    def from_artifacts(cls, model_dir: Path | None = None) -> "InferenceEngine":
        return cls(load_model_bundle(model_dir))

    def _fill_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self.bundle.metadata["dataset_summary"]["default_inputs"]
        prepared = defaults.copy()
        prepared.update({k: v for k, v in payload.items() if v is not None})
        prepared["season"] = derive_season(date.today().month)
        prepared["crop_year"] = date.today().year
        return prepared

    def _feature_name(self, raw_name: str) -> str:
        cleaned = raw_name.replace("num__", "").replace("cat__", "")
        cleaned = cleaned.replace("_", " ")
        return cleaned.title()

    def _get_explainer(self, key: str, model: Any) -> Any | None:
        if key in self._explainers:
            return self._explainers[key]
        try:
            import shap  # type: ignore
        except ImportError:
            self._explainers[key] = None
            return None
        explainer = shap.TreeExplainer(model)
        self._explainers[key] = explainer
        return explainer

    def _top_contributions(self, values: np.ndarray, feature_names: list[str], transformed_row: np.ndarray, top_n: int = 5) -> dict[str, list[dict[str, float | str]]]:
        flat_values = np.asarray(values, dtype=float).reshape(-1)
        flat_row = np.asarray(transformed_row, dtype=float).reshape(-1)
        pairs = [
            {
                "feature": self._feature_name(feature_names[idx]),
                "raw_feature": feature_names[idx],
                "contribution": float(flat_values[idx]),
                "magnitude": float(abs(flat_values[idx])),
                "feature_value": float(flat_row[idx]),
            }
            for idx in range(len(feature_names))
        ]
        pairs = [
            item
            for item in pairs
            if not (str(item["raw_feature"]).startswith("cat__") and abs(float(item["feature_value"])) < 1e-9)
        ]
        positive = sorted((item for item in pairs if item["contribution"] > 0), key=lambda item: item["magnitude"], reverse=True)[:top_n]
        negative = sorted((item for item in pairs if item["contribution"] < 0), key=lambda item: item["magnitude"], reverse=True)[:top_n]
        return {"positive": positive, "negative": negative}

    def _local_shap_explanations(
        self,
        X_class: np.ndarray,
        X_reg: np.ndarray,
        crop_name: str,
    ) -> dict[str, Any]:
        feature_names_cls = list(self.bundle.classification_preprocessor.get_feature_names_out())
        feature_names_reg = list(self.bundle.regression_preprocessor.get_feature_names_out())
        classification = {"available": False, "positive": [], "negative": []}
        regression = {"available": False, "positive": [], "negative": []}

        class_index = int(self.bundle.label_encoder.transform([crop_name])[0])
        cls_explainer = self._get_explainer("lgbm_classifier", self.bundle.lgbm_model)
        if cls_explainer is not None:
            try:
                shap_values = cls_explainer.shap_values(X_class)
                selected = _extract_class_shap(shap_values, class_index)
                contributions = self._top_contributions(selected, feature_names_cls, X_class[0])
                classification = {"available": True, **contributions}
            except Exception:
                classification = {"available": False, "positive": [], "negative": []}

        reg_explainer = self._get_explainer("yield_regressor", self.bundle.yield_model)
        if reg_explainer is not None:
            try:
                shap_values = np.asarray(reg_explainer.shap_values(X_reg), dtype=float)
                contributions = self._top_contributions(shap_values, feature_names_reg, X_reg[0])
                regression = {"available": True, **contributions}
            except Exception:
                regression = {"available": False, "positive": [], "negative": []}

        return {"classification": classification, "yield": regression}

    def _rule_based_explanation(self, prepared: dict[str, Any], profile: dict[str, Any], crop_name: str) -> dict[str, list[str]]:
        positives: list[str] = []
        concerns: list[str] = []

        checks = [
            ("nitrogen", "min_nitrogen", "max_nitrogen", "Nitrogen"),
            ("phosphorous", "min_phosphorous", "max_phosphorous", "Phosphorous"),
            ("potassium", "min_potassium", "max_potassium", "Potassium"),
            ("temperature_c", "min_temp_c", "max_temp_c", "Temperature"),
            ("humidity", "min_humidity", "max_humidity", "Humidity"),
            ("ph", "min_ph", "max_ph", "Soil pH"),
            ("rainfall_mm", "min_rainfall_mm", "max_rainfall_mm", "Rainfall"),
        ]
        for feature, min_key, max_key, label in checks:
            value = float(prepared[feature])
            min_val = float(profile[min_key])
            max_val = float(profile[max_key])
            if min_val <= value <= max_val:
                positives.append(f"{label} is within the suitable range for {crop_name}.")
            else:
                concerns.append(f"{label} is outside the preferred range for {crop_name}.")
        return {"positives": positives[:4], "concerns": concerns[:4]}

    def _market_price_rs_per_kg(self, crop_name: str, state_name: str, season: str) -> float:
        market_df = pd.DataFrame(self.bundle.metadata["market_prices"])
        crop_key = crop_name.strip().lower()
        crop_rules = CROP_DATABASE.get(crop_key, {})
        minimum, maximum = crop_rules.get("price_range_rs_per_kg", (12.0, 40.0))
        baseline = crop_rules.get("base_price_rs_per_kg", (minimum + maximum) / 2.0)
        exact = market_df[
            (market_df["crop"] == crop_name)
            & (market_df["state_name"] == state_name)
            & (market_df["season"].astype(str).str.strip() == season)
        ]
        if not exact.empty:
            observed = float(exact["price_per_ton"].median()) / 1000.0
        else:
            crop_only = market_df[market_df["crop"] == crop_name]
            if not crop_only.empty:
                observed = float(crop_only["price_per_ton"].median()) / 1000.0
            else:
                observed = float(baseline)
        return _clip((0.55 * float(baseline)) + (0.45 * observed), float(minimum), float(maximum))

    def _build_regression_frame(self, prepared: dict[str, Any], crop_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "area_ha": float(prepared["area"]),
                    "nitrogen": float(prepared["nitrogen"]),
                    "phosphorous": float(prepared["phosphorous"]),
                    "potassium": float(prepared["potassium"]),
                    "temperature_c": float(prepared["temperature_c"]),
                    "humidity": float(prepared["humidity"]),
                    "ph": float(prepared["ph"]),
                    "rainfall_mm": float(prepared["rainfall_mm"]),
                    "moisture": float(prepared["moisture"]),
                    "crop": crop_name,
                    "state_name": prepared["state_name"],
                    "district_name": prepared["district_name"],
                    "season": prepared["season"],
                }
            ]
        )

    def _baseline_yield(self, crop_name: str, predicted_yield: float, profile: dict[str, Any], crop_rules: dict[str, Any]) -> float:
        yield_candidates = [float(predicted_yield), float(crop_rules["yield_avg_t_ha"])]
        if profile.get("historical_median_yield") is not None:
            yield_candidates.append(float(profile["historical_median_yield"]))
        return max(float(np.median(yield_candidates)), 0.15)

    def _climate_adjustments(self, prepared: dict[str, Any], profile: dict[str, Any], crop_rules: dict[str, Any]) -> tuple[float, bool, list[str]]:
        penalties: list[float] = []
        reasons: list[str] = []

        rainfall = float(prepared["rainfall_mm"])
        rain_min = float(profile["min_rainfall_mm"])
        rain_max = float(profile["max_rainfall_mm"])
        if rainfall < rain_min:
            deficit = (rain_min - rainfall) / max(rain_min, 1.0)
            penalties.append(0.10 if deficit <= 0.30 else 0.22)
            reasons.append("rainfall deficit")
            if deficit > 0.55:
                return 1.0, True, reasons + ["severe rainfall mismatch"]
        elif rainfall > rain_max:
            excess = (rainfall - rain_max) / max(rain_max, 1.0)
            penalties.append(0.08 if excess <= 0.30 else 0.16)
            reasons.append("rainfall excess")
            if excess > 0.70 and crop_rules["water_need"] == "low":
                return 1.0, True, reasons + ["severe rainfall excess"]

        temperature = float(prepared["temperature_c"])
        temp_min = float(profile["min_temp_c"])
        temp_max = float(profile["max_temp_c"])
        temp_span = max(temp_max - temp_min, 1.0)
        if temperature < temp_min:
            mismatch = (temp_min - temperature) / temp_span
            penalties.append(0.08 if mismatch <= 0.30 else 0.18)
            reasons.append("temperature below range")
            if mismatch > 0.55:
                return 1.0, True, reasons + ["severe temperature mismatch"]
        elif temperature > temp_max:
            mismatch = (temperature - temp_max) / temp_span
            penalties.append(0.08 if mismatch <= 0.30 else 0.18)
            reasons.append("temperature above range")
            if mismatch > 0.55:
                return 1.0, True, reasons + ["severe temperature mismatch"]

        humidity = float(prepared["humidity"])
        if humidity < float(profile["min_humidity"]) or humidity > float(profile["max_humidity"]):
            penalties.append(0.03)
            reasons.append("humidity mismatch")

        ph_value = float(prepared["ph"])
        ph_min = float(profile["min_ph"])
        ph_max = float(profile["max_ph"])
        if ph_value < ph_min or ph_value > ph_max:
            ph_gap = min(abs(ph_value - ph_min), abs(ph_value - ph_max))
            penalties.append(0.05 if ph_gap <= 0.6 else 0.10)
            reasons.append("soil pH mismatch")
            if ph_gap > 1.2:
                return 1.0, True, reasons + ["severe soil pH mismatch"]

        nutrient_penalty = 0.0
        for key, min_key, max_key in [
            ("nitrogen", "min_nitrogen", "max_nitrogen"),
            ("phosphorous", "min_phosphorous", "max_phosphorous"),
            ("potassium", "min_potassium", "max_potassium"),
        ]:
            value = float(prepared[key])
            if value < float(profile[min_key]) or value > float(profile[max_key]):
                nutrient_penalty += 0.02
        if nutrient_penalty:
            penalties.append(min(nutrient_penalty, 0.06))
            reasons.append("nutrient mismatch")

        moisture = float(prepared["moisture"])
        if crop_rules["water_need"] == "high" and moisture < 28:
            penalties.append(0.06)
            reasons.append("low field moisture")
        elif crop_rules["water_need"] == "low" and moisture > 55:
            penalties.append(0.03)
            reasons.append("excess field moisture")

        return min(sum(penalties), 0.70), False, reasons

    def _price_adjustment(self, crop_name: str, sowing_month: int, crop_rules: dict[str, Any], prepared: dict[str, Any]) -> tuple[float, str]:
        base_price = self._market_price_rs_per_kg(crop_name, str(prepared["state_name"]), str(prepared["season"]))
        harvest_month_number = harvest_month(sowing_month, int(crop_rules["duration_months"]))
        peak_months = set(crop_rules["peak_harvest_months"])
        if harvest_month_number in peak_months:
            factor = 0.82 if float(crop_rules["price_volatility"]) >= 0.25 else 0.88
            reason = "peak harvest supply"
        elif float(crop_rules["price_volatility"]) >= 0.26:
            factor = 1.12
            reason = "off-season support"
        else:
            factor = 1.03
            reason = "stable market"
        adjusted = base_price * factor
        minimum, maximum = crop_rules["price_range_rs_per_kg"]
        return _clip(adjusted, float(minimum), float(maximum)), reason

    def _cost_model(self, crop_rules: dict[str, Any], area_ha: float) -> dict[str, float]:
        base_cost = float(crop_rules["base_cost_rs_per_ha"]) * 0.75
        fixed_cost = base_cost * 0.22
        fertilizers = base_cost * 0.18
        pesticides = base_cost * float(crop_rules["chemical_dependency"]) * 0.18
        irrigation = base_cost * water_need_factor(str(crop_rules["water_need"]))
        labour = base_cost * 0.20
        machinery = base_cost * 0.08
        post_harvest = base_cost * 0.09
        subtotal = fixed_cost + fertilizers + pesticides + irrigation + labour + machinery + post_harvest
        buffer = subtotal * 0.12
        total_cost_rs_per_ha = subtotal + buffer
        return {
            "fixed_cost_rs_per_ha": fixed_cost,
            "fertilizer_cost_rs_per_ha": fertilizers,
            "pesticide_cost_rs_per_ha": pesticides,
            "irrigation_cost_rs_per_ha": irrigation,
            "labour_cost_rs_per_ha": labour,
            "machinery_cost_rs_per_ha": machinery,
            "post_harvest_cost_rs_per_ha": post_harvest,
            "buffer_cost_rs_per_ha": buffer,
            "total_cost_rs_per_ha": total_cost_rs_per_ha,
            "total_cost": total_cost_rs_per_ha * area_ha,
        }

    def _stable_price_rs_per_kg(self, crop_name: str, prepared: dict[str, Any], crop_rules: dict[str, Any]) -> float:
        base_price = self._market_price_rs_per_kg(crop_name, str(prepared["state_name"]), str(prepared["season"]))
        minimum, maximum = crop_rules["price_range_rs_per_kg"]
        baseline = float(crop_rules["base_price_rs_per_kg"])
        stable_price = (0.70 * base_price) + (0.30 * baseline)
        return _clip(stable_price, float(minimum), float(maximum))

    def predict(self, payload: dict[str, Any], top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
        analysis_context = current_analysis_context()
        current_month = int(analysis_context["current_month_number"])
        current_season = str(analysis_context["season"])
        analysis_date = str(analysis_context["analysis_date"])
        prepared = self._fill_defaults(payload)
        class_frame = pd.DataFrame(
            [
                {
                    **{feature: prepared.get(feature) for feature in CLASSIFICATION_NUMERIC_FEATURES},
                    **{feature: prepared.get(feature) for feature in CLASSIFICATION_CATEGORICAL_FEATURES},
                }
            ]
        )
        X_class = _to_float32_array(self.bundle.classification_preprocessor.transform(class_frame))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            lgbm_probs = np.asarray(self.bundle.lgbm_model.predict_proba(X_class))[0]
            stacked_probs = np.asarray(self.bundle.stacking_model.predict_proba(X_class))[0]

        labels = self.bundle.label_encoder.inverse_transform(np.arange(len(stacked_probs)))
        ranked_indices = np.argsort(stacked_probs)[::-1][: self.bundle.metadata["ranking_pool_size"]]
        crop_profiles = self.bundle.metadata["crop_profiles"]
        area_ha = float(prepared["area"])

        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for idx in ranked_indices:
            crop_name = str(labels[idx])
            crop_key = crop_name.strip().lower()
            crop_rules = CROP_DATABASE.get(crop_key)
            profile = crop_profiles.get(crop_name, {})
            if not crop_rules or not profile:
                rejected.append({"crop": crop_name, "reason": "missing crop profile or crop rule"})
                continue

            sowing_months = list(crop_rules["sowing_months"])
            if current_month not in sowing_months:
                rejected.append({"crop": crop_name, "reason": f"outside sowing window for {month_name(current_month)}"})
                continue

            reg_frame = self._build_regression_frame(prepared, crop_name)
            X_reg = _to_float32_array(self.bundle.regression_preprocessor.transform(reg_frame))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                ml_yield = float(self.bundle.yield_model.predict(X_reg)[0])

            base_yield_t_ha = self._baseline_yield(crop_name, ml_yield, profile, crop_rules)
            climate_penalty, should_reject, climate_notes = self._climate_adjustments(prepared, profile, crop_rules)
            if should_reject:
                rejected.append({"crop": crop_name, "reason": ", ".join(climate_notes)})
                continue

            sowing_penalty = late_sowing_penalty(current_month, sowing_months)
            new_farmer_penalty = float(crop_rules["new_farmer_penalty"])
            total_yield_penalty = min(new_farmer_penalty + climate_penalty + sowing_penalty, 0.58)
            expected_yield_t_ha = max(base_yield_t_ha * (1.0 - total_yield_penalty), 0.15)

            adjusted_price_rs_per_kg, price_reason = self._price_adjustment(crop_name, current_month, crop_rules, prepared)
            harvest_month_number = harvest_month(current_month, int(crop_rules["duration_months"]))
            cost_model = self._cost_model(crop_rules, area_ha)
            revenue_rs_per_ha = expected_yield_t_ha * adjusted_price_rs_per_kg * 1000.0
            profit_rs_per_ha = revenue_rs_per_ha - cost_model["total_cost_rs_per_ha"]
            revenue = revenue_rs_per_ha * area_ha
            profit = profit_rs_per_ha * area_ha

            risk = _clip(
                (float(crop_rules["base_risk"]) * 0.30)
                + (water_need_factor(str(crop_rules["water_need"])) * 0.25)
                + (float(crop_rules["price_volatility"]) * 0.20)
                + (float(crop_rules["new_farmer_penalty"]) * 0.15)
                + (climate_penalty * 0.10),
                0.12,
                0.88,
            )
            sustainability = _clip(
                float(crop_rules["base_sustainability"])
                - (0.12 if crop_rules["water_need"] == "high" else 0.04 if crop_rules["water_need"] == "medium" else 0.0)
                - (float(crop_rules["chemical_dependency"]) * 0.18)
                - (float(crop_rules["soil_sensitivity"]) * 0.10),
                0.18,
                0.92,
            )

            candidates.append(
                {
                    "crop": crop_name,
                    "classification_probability": float(stacked_probs[idx]),
                    "tabnet_probability": None,
                    "lightgbm_probability": float(lgbm_probs[idx]),
                    "sowing_month": month_name(current_month),
                    "harvest_month": month_name(harvest_month_number),
                    "duration_months": int(crop_rules["duration_months"]),
                    "yield_start_month": month_name(harvest_month_number),
                    "predicted_yield": float(expected_yield_t_ha),
                    "expected_yield_t_ha": float(expected_yield_t_ha),
                    "base_yield_t_ha": float(base_yield_t_ha),
                    "ml_yield_signal_t_ha": float(ml_yield),
                    "adjusted_price_rs_per_kg": float(adjusted_price_rs_per_kg),
                    "market_price_per_ton": float(adjusted_price_rs_per_kg * 1000.0),
                    "total_cost_rs_per_ha": float(cost_model["total_cost_rs_per_ha"]),
                    "revenue_rs_per_ha": float(revenue_rs_per_ha),
                    "profit_rs_per_ha": float(profit_rs_per_ha),
                    "revenue": float(revenue),
                    "profit": float(profit),
                    "total_cost": float(cost_model["total_cost"]),
                    "initial_spend_rs_per_ha": float(cost_model["total_cost_rs_per_ha"]),
                    "risk": float(risk),
                    "risk_score": float(1.0 - risk),
                    "sustainability_score": sustainability,
                    "market_source": "conservative state-season market",
                    "market_type": price_reason,
                    "yield_penalty": float(total_yield_penalty),
                    "late_sowing_penalty": float(sowing_penalty),
                    "new_farmer_penalty": float(new_farmer_penalty),
                    "climate_penalty": float(climate_penalty),
                    "cost_breakdown": cost_model,
                    "advisory_notes": [
                        f"Current month is {month_name(current_month)} and season is {current_season}.",
                        f"Price adjusted for {price_reason}.",
                        "Yield is reduced for new farmer conditions, sowing timing, and climate fit.",
                        "Long-duration crops are allowed for long-term planning, but their higher initial spend is still penalized in ranking.",
                    ],
                    "rejection_checks_passed": climate_notes or ["seasonal and climate screening passed"],
                }
            )

        if not candidates:
            raise ValueError("No crop is currently suitable for sowing at this date and field condition.")

        profit_scores = _candidate_scores([item["profit_rs_per_ha"] for item in candidates])
        low_cost_scores = _candidate_scores([-item["initial_spend_rs_per_ha"] for item in candidates])
        for item, profit_score, low_cost_score in zip(candidates, profit_scores, low_cost_scores):
            item["profit_score"] = float(profit_score)
            item["low_cost_score"] = float(low_cost_score)
            item["final_score"] = float(
                (low_cost_score * 0.35)
                + (profit_score * 0.35)
                + ((1.0 - item["risk"]) * 0.20)
                + (item["sustainability_score"] * 0.10)
            )
            item["high_investment_profit_score"] = float(
                (profit_score * 0.60)
                + ((1.0 - low_cost_score) * 0.25)
                + ((1.0 - item["risk"]) * 0.10)
                + (item["sustainability_score"] * 0.05)
            )

        standard_candidates = sorted(candidates, key=lambda item: item["final_score"], reverse=True)
        preferred_slots = max(min(top_k, 5) - 1, 1)
        spend_values = [item["initial_spend_rs_per_ha"] for item in candidates]
        premium_threshold = float(np.quantile(spend_values, 0.65)) if len(spend_values) > 1 else spend_values[0]

        low_medium_pool = [item for item in standard_candidates if item["initial_spend_rs_per_ha"] <= premium_threshold]
        high_spend_pool = [item for item in candidates if item["initial_spend_rs_per_ha"] > premium_threshold]

        top_candidates = low_medium_pool[:preferred_slots]
        selected_crops = {item["crop"] for item in top_candidates}

        if len(top_candidates) < preferred_slots:
            low_medium_fallback = [item for item in standard_candidates if item["crop"] not in selected_crops]
            while len(top_candidates) < preferred_slots and low_medium_fallback:
                candidate = low_medium_fallback.pop(0)
                if candidate["crop"] not in selected_crops:
                    top_candidates.append(candidate)
                    selected_crops.add(candidate["crop"])

        premium_pool = [item for item in high_spend_pool if item["crop"] not in selected_crops]
        if premium_pool:
            premium_candidate = max(premium_pool, key=lambda item: item["high_investment_profit_score"])
            top_candidates.append(premium_candidate)
            selected_crops.add(premium_candidate["crop"])

        remaining_pool = [item for item in standard_candidates if item["crop"] not in selected_crops]
        while len(top_candidates) < top_k and remaining_pool:
            top_candidates.append(remaining_pool.pop(0))

        ideal_candidates: list[dict[str, Any]] = []
        for idx in ranked_indices:
            crop_name = str(labels[idx])
            crop_key = crop_name.strip().lower()
            crop_rules = CROP_DATABASE.get(crop_key)
            profile = crop_profiles.get(crop_name, {})
            if not crop_rules or not profile:
                continue

            reg_frame = self._build_regression_frame(prepared, crop_name)
            X_reg = _to_float32_array(self.bundle.regression_preprocessor.transform(reg_frame))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                ml_yield = float(self.bundle.yield_model.predict(X_reg)[0])

            base_yield_t_ha = self._baseline_yield(crop_name, ml_yield, profile, crop_rules)
            climate_penalty, should_reject, climate_notes = self._climate_adjustments(prepared, profile, crop_rules)
            if should_reject:
                continue

            conservative_new_farmer_penalty = min(float(crop_rules["new_farmer_penalty"]) + 0.05, 0.42)
            long_term_yield_penalty = min(conservative_new_farmer_penalty + climate_penalty, 0.55)
            expected_yield_t_ha = max(base_yield_t_ha * (1.0 - long_term_yield_penalty), 0.15)

            stable_price_rs_per_kg = self._stable_price_rs_per_kg(crop_name, prepared, crop_rules)
            cost_model = self._cost_model(crop_rules, area_ha)
            revenue_rs_per_ha = expected_yield_t_ha * stable_price_rs_per_kg * 1000.0
            profit_rs_per_ha = revenue_rs_per_ha - cost_model["total_cost_rs_per_ha"]

            risk = _clip(
                (float(crop_rules["base_risk"]) * 0.36)
                + (water_need_factor(str(crop_rules["water_need"])) * 0.22)
                + (float(crop_rules["price_volatility"]) * 0.10)
                + (float(crop_rules["new_farmer_penalty"]) * 0.16)
                + (climate_penalty * 0.16),
                0.10,
                0.88,
            )
            sustainability = _clip(
                float(crop_rules["base_sustainability"])
                - (0.12 if crop_rules["water_need"] == "high" else 0.04 if crop_rules["water_need"] == "medium" else 0.0)
                - (float(crop_rules["chemical_dependency"]) * 0.18)
                - (float(crop_rules["soil_sensitivity"]) * 0.10),
                0.18,
                0.92,
            )

            fit_components = []
            for feature, min_key, max_key in [
                ("nitrogen", "min_nitrogen", "max_nitrogen"),
                ("phosphorous", "min_phosphorous", "max_phosphorous"),
                ("potassium", "min_potassium", "max_potassium"),
                ("temperature_c", "min_temp_c", "max_temp_c"),
                ("humidity", "min_humidity", "max_humidity"),
                ("ph", "min_ph", "max_ph"),
                ("rainfall_mm", "min_rainfall_mm", "max_rainfall_mm"),
            ]:
                value = float(prepared[feature])
                lower = float(profile[min_key])
                upper = float(profile[max_key])
                if lower <= value <= upper:
                    fit_components.append(1.0)
                else:
                    gap = min(abs(value - lower), abs(value - upper))
                    span = max(upper - lower, 1.0)
                    fit_components.append(max(0.0, 1.0 - (gap / span)))

            land_suitability = float(np.mean(fit_components)) if fit_components else 0.0
            ideal_candidates.append(
                {
                    "crop": crop_name,
                    "expected_yield_t_ha": float(expected_yield_t_ha),
                    "stable_price_rs_per_kg": float(stable_price_rs_per_kg),
                    "total_cost_rs_per_ha": float(cost_model["total_cost_rs_per_ha"]),
                    "revenue_rs_per_ha": float(revenue_rs_per_ha),
                    "profit_rs_per_ha": float(profit_rs_per_ha),
                    "risk": float(risk),
                    "sustainability_score": float(sustainability),
                    "land_suitability": float(land_suitability),
                    "climate_penalty": float(climate_penalty),
                    "notes": climate_notes or ["strong general soil and climate fit"],
                    "duration_months": int(crop_rules["duration_months"]),
                }
            )

        ideal_profit_scores = _candidate_scores([item["profit_rs_per_ha"] for item in ideal_candidates])
        ideal_low_cost_scores = _candidate_scores([-item["total_cost_rs_per_ha"] for item in ideal_candidates])
        for item, profit_score, low_cost_score in zip(ideal_candidates, ideal_profit_scores, ideal_low_cost_scores):
            item["profit_score"] = float(profit_score)
            item["low_cost_score"] = float(low_cost_score)
            item["ideal_ground_score"] = float(
                (item["land_suitability"] * 0.35)
                + ((1.0 - item["risk"]) * 0.25)
                + (low_cost_score * 0.20)
                + (profit_score * 0.20)
            )

        ideal_candidates.sort(key=lambda item: item["ideal_ground_score"], reverse=True)
        ideal_ground = ideal_candidates[0] if ideal_candidates else None

        best_crop_name = top_candidates[0]["crop"]
        best_profile = crop_profiles[best_crop_name]
        best_reg_frame = self._build_regression_frame(prepared, best_crop_name)
        best_reg_matrix = _to_float32_array(self.bundle.regression_preprocessor.transform(best_reg_frame))
        local_explanations = self._local_shap_explanations(X_class, best_reg_matrix, best_crop_name)
        rule_explanations = self._rule_based_explanation(prepared, best_profile, best_crop_name)

        classification_metrics = self.bundle.metadata["training_report"].get("classification_metrics", {})
        model_overview = [
            {
                "model": name.replace("_", " ").title(),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
                "top3_accuracy": float(metrics.get("top3_accuracy", 0.0)),
            }
            for name, metrics in classification_metrics.items()
            if isinstance(metrics, dict) and "accuracy" in metrics
        ]
        model_overview.sort(key=lambda item: item["accuracy"], reverse=True)

        return {
            "best_crop": best_crop_name,
            "top_crops": top_candidates,
            "training_summary": self.bundle.metadata["training_report"],
            "analysis_context": {
                "analysis_date": analysis_date,
                "current_month": month_name(current_month),
                "season": current_season,
                "area_hectares": area_ha,
                "decision_rule": "Only crops within the current sowing window are recommended. The first four focus on low or medium spending cost with balanced profit, and the fifth slot highlights a higher-spend higher-profit option.",
            },
            "rejected_crops": rejected,
            "used_defaults": {
                "season": current_season,
                "state_name": prepared["state_name"],
                "district_name": prepared["district_name"],
                "crop_year": date.today().year,
                "price_per_ton": prepared["price_per_ton"],
            },
            "ideal_ground_recommendation": (
                {
                    "crop": ideal_ground["crop"],
                    "land_suitability": ideal_ground["land_suitability"],
                    "expected_yield_t_ha": ideal_ground["expected_yield_t_ha"],
                    "stable_price_rs_per_kg": ideal_ground["stable_price_rs_per_kg"],
                    "total_cost_rs_per_ha": ideal_ground["total_cost_rs_per_ha"],
                    "revenue_rs_per_ha": ideal_ground["revenue_rs_per_ha"],
                    "profit_rs_per_ha": ideal_ground["profit_rs_per_ha"],
                    "risk": ideal_ground["risk"],
                    "sustainability_score": ideal_ground["sustainability_score"],
                    "duration_months": ideal_ground["duration_months"],
                    "ideal_ground_score": ideal_ground["ideal_ground_score"],
                    "why": [
                        "This recommendation ignores the current sowing month and focuses on long-term land suitability.",
                        "It prioritizes strong soil and climate fit, lower risk, lower cost, and stable profit.",
                    ]
                    + ideal_ground["notes"][:3],
                }
                if ideal_ground
                else None
            ),
            "explainability": {
                "global_top_features": self.bundle.metadata.get("xai_assets", {}).get("top_features", [])[:8],
                "summary_plot_urls": {
                    "lightgbm": self.bundle.metadata.get("xai_assets", {}).get("lightgbm_summary_plot"),
                    "catboost": self.bundle.metadata.get("xai_assets", {}).get("catboost_summary_plot"),
                },
                "best_crop_local_explanation": {
                    "crop": best_crop_name,
                    "classification_shap": local_explanations["classification"],
                    "yield_shap": local_explanations["yield"],
                    "rule_based_positives": rule_explanations["positives"],
                    "rule_based_concerns": rule_explanations["concerns"],
                },
                "model_overview": model_overview,
            },
        }
