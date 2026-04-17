from __future__ import annotations

from datetime import date


MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


CROP_DATABASE: dict[str, dict[str, object]] = {
    "apple": {
        "duration_months": 18,
        "yield_avg_t_ha": 11.0,
        "water_need": "medium",
        "price_range_rs_per_kg": (28.0, 42.0),
        "base_price_rs_per_kg": 34.0,
        "base_cost_rs_per_ha": 210000.0,
        "sowing_months": [7, 8],
        "peak_harvest_months": [8, 9, 10],
        "new_farmer_penalty": 0.38,
        "base_risk": 0.58,
        "base_sustainability": 0.54,
        "price_volatility": 0.32,
        "chemical_dependency": 0.42,
        "soil_sensitivity": 0.34,
    },
    "banana": {
        "duration_months": 11,
        "yield_avg_t_ha": 28.0,
        "water_need": "high",
        "price_range_rs_per_kg": (11.0, 18.0),
        "base_price_rs_per_kg": 14.0,
        "base_cost_rs_per_ha": 185000.0,
        "sowing_months": list(range(1, 13)),
        "peak_harvest_months": [9, 10, 11],
        "new_farmer_penalty": 0.34,
        "base_risk": 0.46,
        "base_sustainability": 0.48,
        "price_volatility": 0.24,
        "chemical_dependency": 0.36,
        "soil_sensitivity": 0.26,
    },
    "blackgram": {
        "duration_months": 4,
        "yield_avg_t_ha": 0.9,
        "water_need": "low",
        "price_range_rs_per_kg": (58.0, 76.0),
        "base_price_rs_per_kg": 66.0,
        "base_cost_rs_per_ha": 38000.0,
        "sowing_months": [2, 3, 4, 7, 8],
        "peak_harvest_months": [8, 9, 10],
        "new_farmer_penalty": 0.22,
        "base_risk": 0.31,
        "base_sustainability": 0.78,
        "price_volatility": 0.18,
        "chemical_dependency": 0.14,
        "soil_sensitivity": 0.18,
    },
    "coconut": {
        "duration_months": 15,
        "yield_avg_t_ha": 8.0,
        "water_need": "high",
        "price_range_rs_per_kg": (45.0, 70.0),
        "base_price_rs_per_kg": 54.0,
        "base_cost_rs_per_ha": 170000.0,
        "sowing_months": list(range(1, 13)),
        "peak_harvest_months": [9, 10, 11],
        "new_farmer_penalty": 0.37,
        "base_risk": 0.55,
        "base_sustainability": 0.49,
        "price_volatility": 0.28,
        "chemical_dependency": 0.30,
        "soil_sensitivity": 0.32,
    },
    "coffee": {
        "duration_months": 18,
        "yield_avg_t_ha": 1.5,
        "water_need": "medium",
        "price_range_rs_per_kg": (180.0, 260.0),
        "base_price_rs_per_kg": 210.0,
        "base_cost_rs_per_ha": 155000.0,
        "sowing_months": [6, 7, 8],
        "peak_harvest_months": [11, 12, 1],
        "new_farmer_penalty": 0.39,
        "base_risk": 0.62,
        "base_sustainability": 0.58,
        "price_volatility": 0.30,
        "chemical_dependency": 0.34,
        "soil_sensitivity": 0.32,
    },
    "grapes": {
        "duration_months": 14,
        "yield_avg_t_ha": 16.0,
        "water_need": "medium",
        "price_range_rs_per_kg": (26.0, 42.0),
        "base_price_rs_per_kg": 31.0,
        "base_cost_rs_per_ha": 200000.0,
        "sowing_months": [10, 11, 12],
        "peak_harvest_months": [2, 3, 4],
        "new_farmer_penalty": 0.36,
        "base_risk": 0.57,
        "base_sustainability": 0.50,
        "price_volatility": 0.26,
        "chemical_dependency": 0.38,
        "soil_sensitivity": 0.28,
    },
    "jute": {
        "duration_months": 5,
        "yield_avg_t_ha": 2.3,
        "water_need": "medium",
        "price_range_rs_per_kg": (42.0, 58.0),
        "base_price_rs_per_kg": 50.0,
        "base_cost_rs_per_ha": 62000.0,
        "sowing_months": [3, 4, 5],
        "peak_harvest_months": [8, 9, 10],
        "new_farmer_penalty": 0.24,
        "base_risk": 0.38,
        "base_sustainability": 0.66,
        "price_volatility": 0.20,
        "chemical_dependency": 0.16,
        "soil_sensitivity": 0.20,
    },
    "lentil": {
        "duration_months": 5,
        "yield_avg_t_ha": 1.1,
        "water_need": "low",
        "price_range_rs_per_kg": (62.0, 84.0),
        "base_price_rs_per_kg": 72.0,
        "base_cost_rs_per_ha": 41000.0,
        "sowing_months": [10, 11],
        "peak_harvest_months": [2, 3],
        "new_farmer_penalty": 0.23,
        "base_risk": 0.33,
        "base_sustainability": 0.79,
        "price_volatility": 0.19,
        "chemical_dependency": 0.13,
        "soil_sensitivity": 0.18,
    },
    "maize": {
        "duration_months": 4,
        "yield_avg_t_ha": 3.9,
        "water_need": "medium",
        "price_range_rs_per_kg": (18.0, 26.0),
        "base_price_rs_per_kg": 22.0,
        "base_cost_rs_per_ha": 52000.0,
        "sowing_months": [2, 3, 4, 6, 7],
        "peak_harvest_months": [8, 9, 10],
        "new_farmer_penalty": 0.24,
        "base_risk": 0.36,
        "base_sustainability": 0.64,
        "price_volatility": 0.17,
        "chemical_dependency": 0.20,
        "soil_sensitivity": 0.20,
    },
    "mango": {
        "duration_months": 18,
        "yield_avg_t_ha": 9.0,
        "water_need": "medium",
        "price_range_rs_per_kg": (18.0, 32.0),
        "base_price_rs_per_kg": 24.0,
        "base_cost_rs_per_ha": 165000.0,
        "sowing_months": [6, 7, 8],
        "peak_harvest_months": [4, 5, 6],
        "new_farmer_penalty": 0.38,
        "base_risk": 0.55,
        "base_sustainability": 0.56,
        "price_volatility": 0.25,
        "chemical_dependency": 0.31,
        "soil_sensitivity": 0.28,
    },
    "orange": {
        "duration_months": 16,
        "yield_avg_t_ha": 11.0,
        "water_need": "medium",
        "price_range_rs_per_kg": (18.0, 30.0),
        "base_price_rs_per_kg": 24.0,
        "base_cost_rs_per_ha": 172000.0,
        "sowing_months": [6, 7, 8],
        "peak_harvest_months": [11, 12, 1],
        "new_farmer_penalty": 0.36,
        "base_risk": 0.52,
        "base_sustainability": 0.55,
        "price_volatility": 0.24,
        "chemical_dependency": 0.30,
        "soil_sensitivity": 0.27,
    },
    "papaya": {
        "duration_months": 10,
        "yield_avg_t_ha": 22.0,
        "water_need": "medium",
        "price_range_rs_per_kg": (11.0, 20.0),
        "base_price_rs_per_kg": 15.0,
        "base_cost_rs_per_ha": 145000.0,
        "sowing_months": list(range(1, 13)),
        "peak_harvest_months": [6, 7, 8],
        "new_farmer_penalty": 0.33,
        "base_risk": 0.43,
        "base_sustainability": 0.57,
        "price_volatility": 0.22,
        "chemical_dependency": 0.28,
        "soil_sensitivity": 0.24,
    },
    "rice": {
        "duration_months": 5,
        "yield_avg_t_ha": 3.6,
        "water_need": "high",
        "price_range_rs_per_kg": (20.0, 30.0),
        "base_price_rs_per_kg": 24.0,
        "base_cost_rs_per_ha": 65000.0,
        "sowing_months": [6, 7, 8, 11, 12],
        "peak_harvest_months": [10, 11, 12],
        "new_farmer_penalty": 0.26,
        "base_risk": 0.40,
        "base_sustainability": 0.44,
        "price_volatility": 0.16,
        "chemical_dependency": 0.24,
        "soil_sensitivity": 0.20,
    },
}


def derive_season(month: int) -> str:
    if month in (6, 7, 8, 9, 10):
        return "Kharif"
    if month in (11, 12, 1, 2):
        return "Rabi"
    return "Summer"


def month_name(month: int) -> str:
    return MONTH_NAMES[int(((month - 1) % 12) + 1)]


def harvest_month(sowing_month: int, duration_months: int) -> int:
    return int(((sowing_month - 1 + duration_months - 1) % 12) + 1)


def late_sowing_penalty(month: int, sowing_months: list[int]) -> float:
    if len(sowing_months) <= 1:
        return 0.0
    if month == sowing_months[-1]:
        return 0.25
    if month != sowing_months[0]:
        return 0.12
    return 0.0


def water_need_factor(level: str) -> float:
    return {"low": 0.12, "medium": 0.22, "high": 0.34}.get(level, 0.22)


def current_analysis_context() -> dict[str, object]:
    today = date.today()
    return {
        "analysis_date": today.isoformat(),
        "current_month_number": today.month,
        "current_month": month_name(today.month),
        "season": derive_season(today.month),
        "year": today.year,
    }
