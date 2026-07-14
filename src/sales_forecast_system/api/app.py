from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging

from .schemas import (
    ForecastRequest, ForecastResponse, ModelInfo, 
    HistoricalDataRequest, ErrorAnalysisRequest
)
from .inference import ForecastInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sales Forecasting API",
    description="API for generating sales forecasts and analyzing forecast accuracy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine
inference_engine = ForecastInference()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Model information endpoint
@app.get("/models/info", response_model=Dict)
async def get_model_info():
    """Get information about available models"""
    try:
        models_dir = Path("../models")
        model_info = {}
        
        # Look for model files
        model_files = list(models_dir.glob("*.pkl"))
        
        for model_file in model_files:
            model_name = model_file.stem
            try:
                model = joblib.load(model_file)
                
                model_info[model_name] = {
                    "file_name": model_file.name,
                    "file_size": model_file.stat().st_size,
                    "created": datetime.fromtimestamp(model_file.stat().st_ctime).isoformat(),
                    "model_type": type(model).__name__,
                    "available": True
                }
            except:
                model_info[model_name] = {
                    "file_name": model_file.name,
                    "available": False
                }
        
        return {
            "total_models": len(model_info),
            "available_models": sum(1 for m in model_info.values() if m.get("available", False)),
            "models": model_info
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Forecast generation endpoint
@app.post("/forecast/generate", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate sales forecast for specified store and department"""
    try:
        logger.info(f"Generating forecast for Store {request.store_id}, Dept {request.dept_id}")
        
        # Generate forecast
        forecasts, metrics = inference_engine.generate_forecast(
            store_id=request.store_id,
            dept_id=request.dept_id,
            periods=request.periods,
            model_type=request.model_type,
            include_confidence=request.include_confidence
        )
        
        # Prepare response
        forecast_data = []
        for i, (date, row) in enumerate(forecasts.iterrows()):
            forecast_item = {
                "date": date.isoformat(),
                "forecast": float(row["forecast"]),
                "horizon": i + 1
            }
            
            if request.include_confidence and "lower_bound" in row and "upper_bound" in row:
                forecast_item["confidence_interval"] = {
                    "lower_bound": float(row["lower_bound"]),
                    "upper_bound": float(row["upper_bound"])
                }
            
            forecast_data.append(forecast_item)
        
        response = ForecastResponse(
            store_id=request.store_id,
            dept_id=request.dept_id,
            model_type=request.model_type,
            forecast_horizon=request.periods,
            generation_date=datetime.now().isoformat(),
            forecasts=forecast_data,
            metrics=metrics
        )
        
        logger.info(f"Forecast generated successfully: {len(forecast_data)} periods")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch forecast endpoint
@app.post("/forecast/batch")
async def generate_batch_forecast(
    store_ids: List[int] = Query(...),
    dept_ids: List[int] = Query(...),
    periods: int = Query(12, ge=1, le=52),
    model_type: str = Query("prophet")
):
    """Generate forecasts for multiple store-department combinations"""
    try:
        results = []
        
        for store_id in store_ids:
            for dept_id in dept_ids:
                try:
                    forecasts, metrics = inference_engine.generate_forecast(
                        store_id=store_id,
                        dept_id=dept_id,
                        periods=periods,
                        model_type=model_type
                    )
                    
                    result = {
                        "store_id": store_id,
                        "dept_id": dept_id,
                        "status": "success",
                        "forecast_count": len(forecasts),
                        "metrics": metrics
                    }
                    
                except Exception as e:
                    result = {
                        "store_id": store_id,
                        "dept_id": dept_id,
                        "status": "error",
                        "error": str(e)
                    }
                
                results.append(result)
        
        return {
            "total_requests": len(results),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Historical data endpoint
@app.post("/data/historical")
async def get_historical_data(request: HistoricalDataRequest):
    """Get historical sales data for analysis"""
    try:
        # Load features data
        features_path = Path("../data/features/time_series_features.parquet")
        if not features_path.exists():
            raise HTTPException(status_code=404, detail="Features data not found")
        
        df = pd.read_parquet(features_path)
        
        # Filter for specific store and department
        mask = (df["Store"] == request.store_id) & (df["Dept"] == request.dept_id)
        store_dept_data = df[mask].copy()
        
        # Sort by date
        store_dept_data = store_dept_data.sort_values("Date")
        
        # Apply date filter
        if request.start_date:
            start_date = pd.to_datetime(request.start_date)
            store_dept_data = store_dept_data[store_dept_data["Date"] >= start_date]
        
        if request.end_date:
            end_date = pd.to_datetime(request.end_date)
            store_dept_data = store_dept_data[store_dept_data["Date"] <= end_date]
        
        # Select columns
        if request.columns:
            available_cols = [col for col in request.columns if col in store_dept_data.columns]
            store_dept_data = store_dept_data[["Date", "Weekly_Sales"] + available_cols]
        
        # Convert to response format
        historical_data = []
        for _, row in store_dept_data.iterrows():
            item = {
                "date": row["Date"].isoformat(),
                "weekly_sales": float(row["Weekly_Sales"])
            }
            
            # Add additional columns if requested
            if request.columns:
                for col in request.columns:
                    if col in row and col not in ["Date", "Weekly_Sales"]:
                        if pd.notna(row[col]):
                            item[col] = float(row[col])
                        else:
                            item[col] = None
            
            historical_data.append(item)
        
        return {
            "store_id": request.store_id,
            "dept_id": request.dept_id,
            "record_count": len(historical_data),
            "date_range": {
                "start": store_dept_data["Date"].min().isoformat() if len(store_dept_data) > 0 else None,
                "end": store_dept_data["Date"].max().isoformat() if len(store_dept_data) > 0 else None
            },
            "data": historical_data
        }
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model performance endpoint
@app.get("/models/performance")
async def get_model_performance(
    store_id: int = Query(..., ge=1),
    dept_id: int = Query(..., ge=1),
    model_type: Optional[str] = None
):
    """Get performance metrics for trained models"""
    try:
        # Load predictions
        predictions_path = Path(f"../models/predictions_store_{store_id}_dept_{dept_id}.csv")
        if not predictions_path.exists():
            raise HTTPException(status_code=404, detail="Performance data not found for this combination")
        
        predictions_df = pd.read_csv(predictions_path)
        predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
        
        # Calculate metrics for each model
        performance = {}
        
        model_columns = [col for col in predictions_df.columns if "Prediction" in col]
        
        for pred_col in model_columns:
            model_name = pred_col.replace("_Prediction", "").lower()
            
            if model_type and model_type.lower() != model_name:
                continue
            
            actual = predictions_df["Actual"]
            predicted = predictions_df[pred_col]
            
            # Calculate metrics
            errors = predicted - actual
            
            metrics = {
                "mape": float(np.mean(np.abs(errors / actual)) * 100),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "bias": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
                "direction_accuracy": None
            }
            
            # Directional accuracy
            if len(actual) > 1:
                direction_actual = np.sign(np.diff(actual))
                direction_pred = np.sign(np.diff(predicted))
                direction_match = direction_actual == direction_pred
                metrics["direction_accuracy"] = float(np.mean(direction_match)) * 100 if len(direction_match) > 0 else 0
            
            performance[model_name] = metrics
        
        return {
            "store_id": store_id,
            "dept_id": dept_id,
            "evaluation_period": {
                "start": predictions_df["Date"].min().isoformat(),
                "end": predictions_df["Date"].max().isoformat(),
                "weeks": len(predictions_df)
            },
            "performance": performance
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error analysis endpoint
@app.post("/analysis/errors")
async def analyze_errors(request: ErrorAnalysisRequest):
    """Analyze forecast errors and identify patterns"""
    try:
        from src.interpretability.error_analysis import ErrorAnalyzer
        
        analyzer = ErrorAnalyzer()
        
        # Load forecast data
        analysis_df = analyzer.load_forecast_data(
            request.store_id, 
            request.dept_id
        )
        
        # Perform analysis
        root_causes = analyzer.generate_root_cause_analysis(
            analysis_df, 
            request.model_type or "prophet"
        )
        
        # Simplify response
        simplified_response = {
            "store_id": request.store_id,
            "dept_id": request.dept_id,
            "model_type": request.model_type or "prophet",
            "summary": root_causes.get("error_patterns", {}).get("summary", {}),
            "identified_issues": root_causes.get("identified_issues", []),
            "recommended_actions": root_causes.get("recommended_actions", [])
        }
        
        return simplified_response
        
    except Exception as e:
        logger.error(f"Error in error analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Forecast comparison endpoint
@app.get("/forecast/compare")
async def compare_forecasts(
    store_id: int = Query(..., ge=1),
    dept_id: int = Query(..., ge=1),
    periods: int = Query(12, ge=1, le=52)
):
    """Compare forecasts from different models"""
    try:
        model_types = ["prophet", "arima", "sarima"]
        comparisons = {}
        
        for model_type in model_types:
            try:
                forecasts, metrics = inference_engine.generate_forecast(
                    store_id=store_id,
                    dept_id=dept_id,
                    periods=periods,
                    model_type=model_type
                )
                
                comparisons[model_type] = {
                    "forecast_count": len(forecasts),
                    "total_forecast": float(forecasts["forecast"].sum()),
                    "mean_forecast": float(forecasts["forecast"].mean()),
                    "std_forecast": float(forecasts["forecast"].std()),
                    "uncertainty": float((forecasts["upper_bound"] - forecasts["lower_bound"]).mean()) 
                                  if "upper_bound" in forecasts.columns else None
                }
                
            except Exception as e:
                comparisons[model_type] = {
                    "error": str(e),
                    "available": False
                }
        
        # Determine best model based on uncertainty (if available)
        available_models = {k: v for k, v in comparisons.items() 
                          if v.get("available", True) and v.get("uncertainty") is not None}
        
        if available_models:
            best_model = min(available_models.items(), 
                           key=lambda x: x[1]["uncertainty"])[0]
        else:
            best_model = None
        
        return {
            "store_id": store_id,
            "dept_id": dept_id,
            "forecast_horizon": periods,
            "best_model": best_model,
            "comparisons": comparisons
        }
        
    except Exception as e:
        logger.error(f"Error comparing forecasts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model retraining endpoint
@app.post("/models/retrain")
async def retrain_model(
    store_id: int = Query(..., ge=1),
    dept_id: int = Query(..., ge=1),
    model_type: str = Query("prophet"),
    test_size: float = Query(0.2, ge=0.1, le=0.5)
):
    """Retrain a forecasting model with updated data"""
    try:
        from src.models.train import ModelTrainer
        
        trainer = ModelTrainer()
        
        # Retrain model
        results = trainer.train_all_models(
            features_path="../data/features/time_series_features.parquet",
            store_id=store_id,
            dept_id=dept_id
        )
        
        # Get the specific model performance
        model_performance = results["metrics"].get(model_type, {})
        
        return {
            "store_id": store_id,
            "dept_id": dept_id,
            "model_type": model_type,
            "status": "retrained",
            "performance": model_performance,
            "best_model": results["metadata"]["best_model"],
            "retraining_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Business impact endpoint
@app.get("/analysis/business-impact")
async def analyze_business_impact(
    store_id: int = Query(..., ge=1),
    dept_id: int = Query(..., ge=1),
    safety_stock_service_level: float = Query(0.95, ge=0.8, le=0.99),
    holding_cost_rate: float = Query(0.25, ge=0.1, le=0.5),
    stockout_cost_multiplier: float = Query(3.0, ge=1.0, le=10.0)
):
    """Analyze business impact of forecast accuracy"""
    try:
        # This would integrate with the business impact analysis from notebook 06
        # For now, return a placeholder response
        return {
            "store_id": store_id,
            "dept_id": dept_id,
            "analysis": {
                "safety_stock_service_level": safety_stock_service_level,
                "holding_cost_rate": holding_cost_rate,
                "stockout_cost_multiplier": stockout_cost_multiplier,
                "estimated_annual_savings": "Analysis would be implemented here",
                "recommended_actions": [
                    "Implement forecast-based inventory management",
                    "Monitor forecast accuracy monthly",
                    "Adjust safety stock based on forecast error distribution"
                ]
            },
            "note": "Business impact analysis would be fully implemented in production"
        }
        
    except Exception as e:
        logger.error(f"Error in business impact analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )