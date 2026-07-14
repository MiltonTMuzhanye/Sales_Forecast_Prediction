import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import ModelTrainingError
import joblib
import mlflow

logger = setup_logger(__name__)

class ARIMAModel:
    """ARIMA model wrapper for sales forecast"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = None
        self.model_fit = None
        self.order = self.config.get('model.models.arima.order', [1, 1, 1])
        
    def train(self, data: pd.Series) -> None:
        """Train ARIMA model"""
        logger.info(f"Training ARIMA model with order {self.order}...")
        
        try:
            self.model = ARIMA(data, order=tuple(self.order))
            self.model_fit = self.model.fit()
            logger.info("ARIMA model trained successfully")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train ARIMA model: {e}")
    
    def predict(self, steps: int) -> np.ndarray:
        """Make predictions using ARIMA model"""
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
    
    def save_model(self, path: str = 'artifacts/trained_models/arima_model.pkl'):
        """Save model to disk"""
        if self.model_fit is None:
            raise ValueError("No model to save. Train first.")
        
        joblib.dump(self.model_fit, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = 'artifacts/trained_models/arima_model.pkl'):
        """Load model from disk"""
        self.model_fit = joblib.load(path)
        logger.info(f"Model loaded from {path}")