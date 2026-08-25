 class SalesForecastError(Exception):
    """Base exception for sales forecast system"""
    pass

class DataIngestionError(SalesForecastError):
    """Raised when data ingestion fails"""
    pass

class DataValidationError(SalesForecastError):
    """Raised when data validation fails"""
    pass

class ModelTrainingError(SalesForecastError):
    """Raised when model training fails"""
    pass

class ModelPredictionError(SalesForecastError):
    """Raised when model prediction fails"""
    pass

class ConfigurationError(SalesForecastError):
    """Raised when configuration is invalid"""
    pass

class FeatureEngineeringError(SalesForecastError):
    """Raised when feature engineering fails"""
    pass