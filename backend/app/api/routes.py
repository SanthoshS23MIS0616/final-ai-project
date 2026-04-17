from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas import PredictionInput, TrainRequest
from backend.app.services.predictor import get_engine, train_models


router = APIRouter(prefix="/api", tags=["crop-intelligence"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metadata")
def metadata() -> dict:
    try:
        engine = get_engine()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"Models are not ready: {exc}") from exc
    return engine.bundle.metadata["dataset_summary"] | {
        "training_report": engine.bundle.metadata["training_report"],
        "xai_assets": engine.bundle.metadata.get("xai_assets", {}),
    }


@router.post("/predict")
def predict(request: PredictionInput) -> dict:
    try:
        engine = get_engine()
        payload = request.model_dump()
        top_k = payload.pop("top_k", 3)
        return engine.predict(payload, top_k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Trained models not found: {exc}") from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/train")
def train(request: TrainRequest) -> dict:
    try:
        return train_models(data_dir=request.data_dir)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
