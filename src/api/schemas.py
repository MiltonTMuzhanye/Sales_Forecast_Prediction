from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum

class ModelType(str, Enum):
    """Available model types"""
    PROPHET = "prophet"
    ARIMA = "arima"
    SARIMA = "sarima"
    ENSEMBLE = "ensemble"

class ForecastRequest(BaseModel):
    """Request schema for forecast generation"""
    store_id: int = Field(..., ge=1, description="Store identifier")
    dept_id: int = Field(..., ge=1, description="Department identifier")
    periods: int = Field(12, ge=1, le=52, description="Number of periods to forecast")
    model_type: ModelType = Field(ModelType.PROPHET, description="Model type to use")
    include_confidence: bool = Field(True, description="Include confidence intervals")
    
    @validator('periods')
    def validate_periods(cls, v):
        if v > 52:
            raise ValueError("Forecast horizon cannot exceed 52 weeks")
        return v

class ForecastItem(BaseModel):
    """Individual forecast item"""
    date: str
    forecast: float
    horizon: int
    confidence_interval: Optional[Dict[str, float]] = None

class ForecastMetrics(BaseModel):
    """Forecast performance metrics"""
    mape: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    bias: Optional[float] = None
    direction_accuracy: Optional[float] = None

class ForecastResponse(BaseModel):
    """Response schema for forecast generation"""
    store_id: int
    dept_id: int
    model_type: str
    forecast_horizon: int
    generation_date: str
    forecasts: List[ForecastItem]
    metrics: Optional[Dict[str, Any]] = None

class HistoricalDataRequest(BaseModel):
    """Request schema for historical data"""
    store_id: int = Field(..., ge=1, description="Store identifier")
    dept_id: int = Field(..., ge=1, description="Department identifier")
    start_date: Optional[date] = Field(None, description="Start date for data filter")
    end_date: Optional[date] = Field(None, description="End date for data filter")
    columns: Optional[List[str]] = Field(None, description="Additional columns to include")

class ErrorAnalysisRequest(BaseModel):
    """Request schema for error analysis"""
    store_id: int = Field(..., ge=1, description="Store identifier")
    dept_id: int = Field(..., ge=1, description="Department identifier")
    model_type: Optional[str] = Field("prophet", description="Model type to analyze")

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    type: str
    created: str
    performance: Optional[Dict[str, float]] = None
    parameters: Optional[Dict[str, Any]] = None

class BatchForecastRequest(BaseModel):
    """Request schema for batch forecasts"""
    combinations: List[Dict[str, int]] = Field(
        ...,
        description="List of store-department combinations"
    )
    periods: int = Field(12, ge=1, le=52)
    model_type: ModelType = Field(ModelType.PROPHET)

class RetrainRequest(BaseModel):
    """Request schema for model retraining"""
    store_id: int = Field(..., ge=1)
    dept_id: int = Field(..., ge=1)
    model_type: ModelType = Field(ModelType.PROPHET)
    test_size: float = Field(0.2, ge=0.1, le=0.5)

class BusinessImpactRequest(BaseModel):
    """Request schema for business impact analysis"""
    store_id: int = Field(..., ge=1)
    dept_id: int = Field(..., ge=1)
    safety_stock_service_level: float = Field(0.95, ge=0.8, le=0.99)
    holding_cost_rate: float = Field(0.25, ge=0.1, le=0.5)
    stockout_cost_multiplier: float = Field(3.0, ge=1.0, le=10.0)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None

# Data validation schemas
class SalesData(BaseModel):
    """Sales data validation schema"""
    store: int
    dept: int
    date: date
    weekly_sales: float
    is_holiday: bool
    
    @validator('weekly_sales')
    def validate_sales(cls, v):
        if v < 0:
            raise ValueError('Weekly sales cannot be negative')
        return v

class StoreData(BaseModel):
    """Store data validation schema"""
    store: int
    type: str
    size: int
    
    @validator('type')
    def validate_type(cls, v):
        if v not in ['A', 'B', 'C']:
            raise ValueError('Store type must be A, B, or C')
        return v

class FeatureData(BaseModel):
    """Feature data validation schema"""
    store: int
    date: date
    temperature: Optional[float] = None
    fuel_price: Optional[float] = None
    cpi: Optional[float] = None
    unemployment: Optional[float] = None
    is_holiday: bool
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < -50 or v > 150):
            raise ValueError('Temperature out of reasonable range')
        return v
    
    @validator('fuel_price')
    def validate_fuel_price(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('Fuel price out of reasonable range')
        return v

# Configuration schemas
class ModelConfig(BaseModel):
    """Model configuration schema"""
    name: str
    type: ModelType
    parameters: Dict[str, Any]
    training_frequency: str = Field("monthly", description="How often to retrain")
    evaluation_metrics: List[str] = Field(["mape", "mae", "rmse"])

class ForecastConfig(BaseModel):
    """Forecast configuration schema"""
    default_horizon: int = Field(12, ge=1, le=52)
    default_model: ModelType = Field(ModelType.PROPHET)
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)
    include_historical: bool = Field(True)

class APIConfig(BaseModel):
    """API configuration schema"""
    host: str = Field("0.0.0.0")
    port: int = Field(8000, ge=1024, le=65535)
    debug: bool = Field(False)
    log_level: str = Field("info")
    rate_limit: Optional[int] = Field(None, ge=1)