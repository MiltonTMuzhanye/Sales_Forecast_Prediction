from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class ForecastRequest(BaseModel):
    store_id: int = Field(..., description="Store ID")
    department_id: int = Field(..., description="Department ID")
    periods: int = Field(12, description="Number of periods to forecast", ge=1, le=52)
    model: Optional[str] = Field("prophet", description="Model to use")
    include_uncertainty: bool = Field(False, description="Include uncertainty intervals")

class StoreForecastRequest(BaseModel):
    store_id: int = Field(..., description="Store ID")
    periods: int = Field(12, description="Number of periods to forecast", ge=1, le=52)
    model: Optional[str] = Field("prophet", description="Model to use")
    include_uncertainty: bool = Field(False, description="Include uncertainty intervals")

class BatchForecastRequest(BaseModel):
    stores_depts: Dict[int, List[int]] = Field(..., description="Mapping of store IDs to department IDs")
    periods: int = Field(12, description="Number of periods to forecast", ge=1, le=52)
    model: Optional[str] = Field("prophet", description="Model to use")
    include_uncertainty: bool = Field(False, description="Include uncertainty intervals")

class ForecastResponse(BaseModel):
    store_id: int
    department_id: int
    model: str
    periods: int
    predictions: List[float]
    dates: List[str]
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: List[str]