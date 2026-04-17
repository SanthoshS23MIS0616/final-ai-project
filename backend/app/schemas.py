from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    nitrogen: Optional[float] = Field(default=None, ge=0)
    phosphorous: Optional[float] = Field(default=None, ge=0)
    potassium: Optional[float] = Field(default=None, ge=0)
    ph: Optional[float] = Field(default=None, ge=0)
    temperature_c: Optional[float] = None
    humidity: Optional[float] = Field(default=None, ge=0)
    rainfall_mm: Optional[float] = Field(default=None, ge=0)
    moisture: Optional[float] = Field(default=None, ge=0)
    area: Optional[float] = Field(default=None, gt=0)
    price_per_ton: Optional[float] = Field(default=None, ge=0)
    season: Optional[str] = None
    state_name: Optional[str] = None
    district_name: Optional[str] = None
    crop_year: Optional[int] = Field(default=None, ge=1990, le=2100)
    top_k: int = Field(default=3, ge=1, le=10)


class TrainRequest(BaseModel):
    data_dir: Optional[str] = None
