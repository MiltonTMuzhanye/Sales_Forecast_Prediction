from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
import numpy as np
from datetime import datetime
import pandas as pd

from .schemas import (
    ForecastRequest, ForecastResponse, 
    StoreForecastRequest, BatchForecastRequest
)

router = APIRouter()

# Global models
models = {}

def get_models():
    """Dependency to get loaded models"""
    return models

@router.post("/forecast", response_model=ForecastResponse)
async def forecast_single(request: ForecastRequest, models: dict = Depends(get_models)):
    """Make a single forecast"""
    try:
        model_name = request.model or 'prophet'
        
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # For demonstration - in production, use actual model
        predictions = np.random.normal(30000, 5000, request.periods)
        dates = pd.date_range(start=datetime.now(), periods=request.periods, freq='W')
        
        return {
            "store_id": request.store_id,
            "department_id": request.department_id,
            "model": model_name,
            "periods": request.periods,
            "predictions": predictions.tolist(),
            "dates": dates.strftime('%Y-%m-%d').tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast/store", response_model=List[ForecastResponse])
async def forecast_store(request: StoreForecastRequest, models: dict = Depends(get_models)):
    """Make forecasts for all departments in a store"""
    try:
        departments = [1, 2, 3, 4, 5]  # Example
        results = []
        
        for dept in departments:
            predictions = np.random.normal(30000, 5000, request.periods)
            dates = pd.date_range(start=datetime.now(), periods=request.periods, freq='W')
            
            results.append({
                "store_id": request.store_id,
                "department_id": dept,
                "model": request.model or 'prophet',
                "periods": request.periods,
                "predictions": predictions.tolist(),
                "dates": dates.strftime('%Y-%m-%d').tolist()
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast/batch", response_model=Dict[str, ForecastResponse])
async def forecast_batch(request: BatchForecastRequest, models: dict = Depends(get_models)):
    """Make forecasts for multiple stores and departments"""
    try:
        results = {}
        
        for store_id, depts in request.stores_depts.items():
            for dept_id in depts:
                key = f"{store_id}_{dept_id}"
                
                predictions = np.random.normal(30000, 5000, request.periods)
                dates = pd.date_range(start=datetime.now(), periods=request.periods, freq='W')
                
                results[key] = {
                    "store_id": int(store_id),
                    "department_id": dept_id,
                    "model": request.model or 'prophet',
                    "periods": request.periods,
                    "predictions": predictions.tolist(),
                    "dates": dates.strftime('%Y-%m-%d').tolist()
                }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models(models: dict = Depends(get_models)):
    """List available models"""
    return {
        "available_models": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }