import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import ModelTrainingError
import joblib
import mlflow

logger = setup_logger(__name__)

class SARIMAModel:
    """SARIMA model wrapper for sales forecast"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = None
        self.model_fit = None
        self.order = self.config.get('model.models.sarima.order', [1, 1, 1])
        self.seasonal_order = self.config.get('model.models.sarima.seasonal_order', [1, 1, 1, 12])
        
    def train(self, data: pd.Series) -> None:
        """Train SARIMA model"""
        logger.info(f"Training SARIMA model with order {self.order} and seasonal order {self.seasonal_order}...")
        
        try:
            self.model = SARIMAX(
                data, 
                order=tuple(self.order), 
                seasonal_order=tuple(self.seasonal_order)
            )
            self.model_fit = self.model.fit(disp=False)
            logger.info("SARIMA model trained successfully")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train SARIMA model: {e}")
    
    def predict(self, steps: int) -> np.ndarray:
        """Make predictions using SARIMA model"""
        if self.model_fit is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model_fit.forecast(steps=steps)
        return predictions
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate model performance"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        return metrics
    
    def save_model(self, path: str = 'artifacts/trained_models/sarima_model.pkl'):
        """Save model to disk"""
        if self.model_fit is None:
            raise ValueError("No model to save. Train first.")
        
        joblib.dump(self.model_fit, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = 'artifacts/trained_models/sarima_model.pkl'):
        """Load model from disk"""
        self.model_fit = joblib.load(path)
        logger.info(f"Model loaded from {path}")