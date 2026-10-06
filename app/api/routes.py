from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException

from .schemas import (
    BatchForecastRequest,
    ForecastRequest,
    ForecastResponse,
    StoreForecastRequest,
)
from app.inference.forecast_service import ForecastService


router = APIRouter()

_service: ForecastService | None = None


def get_service() -> ForecastService:
    """Return the shared forecasting service."""
    global _service

    if _service is None:
        _service = ForecastService()

    return _service


def build_response(result: dict) -> ForecastResponse:
    """Convert a forecast service result into an API response."""

    return ForecastResponse(
        store_id=result["store_id"],
        department_id=result["department_id"],
        model=result["model"],
        periods=result["periods"],
        predictions=result["predictions"],
        dates=[
            date[:10] if "T" in date else date
            for date in result["dates"]
        ],
    )


@router.post("/forecast", response_model=ForecastResponse)
async def forecast_single(
    request: ForecastRequest,
) -> ForecastResponse:
    """Generate a forecast for one Store/Department."""

    try:
        model = request.model or "prophet"

        if model != "prophet":
            raise ValueError(
                f"Model '{model}' is not currently available for production inference. "
                "Use 'prophet'."
            )

        service = get_service()

        result = service.forecast(
            store_id=request.store_id,
            dept_id=request.department_id,
            periods=request.periods,
            model=model,
        )

        return build_response(result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {exc}",
        ) from exc


@router.post(
    "/forecast/store",
    response_model=List[ForecastResponse],
)
async def forecast_store(
    request: StoreForecastRequest,
) -> List[ForecastResponse]:
    """Generate forecasts for all departments in a store."""

    try:
        service = get_service()

        processed = service.preprocessor.load_and_preprocess()

        store_data = processed[
            processed["Store"] == request.store_id
        ]

        if store_data.empty:
            raise ValueError(
                f"No data found for Store={request.store_id}"
            )

        departments = sorted(
            store_data["Dept"].dropna().unique().tolist()
        )

        results = []

        for dept_id in departments:
            result = service.forecast(
                store_id=request.store_id,
                dept_id=int(dept_id),
                periods=request.periods,
                model=request.model or "prophet",
            )

            results.append(build_response(result))

        return results

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Store forecast failed: {exc}",
        ) from exc


@router.post(
    "/forecast/batch",
    response_model=Dict[str, ForecastResponse],
)
async def forecast_batch(
    request: BatchForecastRequest,
) -> Dict[str, ForecastResponse]:
    """Generate forecasts for requested Store/Department pairs."""

    try:
        model = request.model or "prophet"

        if model != "prophet":
            raise ValueError(
                f"Model '{model}' is not currently available for production inference. "
                "Use 'prophet'."
            )

        service = get_service()
        results = {}

        for store_id, departments in request.stores_depts.items():
            for dept_id in departments:

                result = service.forecast(
                    store_id=int(store_id),
                    dept_id=int(dept_id),
                    periods=request.periods,
                    model=model,
                )

                key = f"{store_id}_{dept_id}"

                results[key] = build_response(result)

        return results

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch forecast failed: {exc}",
        ) from exc


@router.get("/models")
async def list_models() -> dict:
    """List models currently available for production inference."""

    return {
        "available_models": ["prophet"],
        "training_models": [
            "prophet",
            "xgboost",
            "lightgbm",
            "hybrid",
        ],
    }
