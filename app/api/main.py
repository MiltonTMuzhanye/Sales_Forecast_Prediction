from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
from datetime import datetime
import joblib
from pathlib import Path

from .routes import router
from .schemas import HealthResponse

app = FastAPI(
    title="Sales Forecasting API",
    description="API for sales forecasting system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Global variables
models = {}

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global models
    
    try:
        from src.sales_forecasting.models.prophet_model import ProphetModel
        from src.sales_forecasting.models.xgboost_model import XGBoostModel
        from src.sales_forecasting.models.lightgbm_model import LightGBMModel
        
        model_path = Path("artifacts/trained_models/")
        
        # Load Prophet
        prophet_path = model_path / "prophet_model.pkl"
        if prophet_path.exists():
            prophet = ProphetModel()
            prophet.load_model(str(prophet_path))
            models['prophet'] = prophet
        
        # Load XGBoost
        xgboost_path = model_path / "xgboost_model.pkl"
        if xgboost_path.exists():
            xgb = XGBoostModel()
            xgb.load_model(str(xgboost_path))
            models['xgboost'] = xgb
        
        # Load LightGBM
        lightgbm_path = model_path / "lightgbm_model.pkl"
        if lightgbm_path.exists():
            lgb = LightGBMModel()
            lgb.load_model(str(lightgbm_path))
            models['lightgbm'] = lgb
        
        print(f"Loaded models: {list(models.keys())}")
        
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(models.keys())
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)