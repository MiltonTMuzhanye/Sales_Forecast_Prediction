import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import ModelTrainingError
from .prophet_model import ProphetModel
from .arima_model import ARIMAModel
from .sarima_model import SARIMAModel
import joblib

logger = setup_logger(__name__)

class EnsembleModel:
    """Ensemble model combining multiple forecasting models"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.models = {}
        self.weights = self.config.get('model.models.ensemble.weights', 
                                      {'prophet': 0.5, 'arima': 0.3, 'sarima': 0.2})
        self.trained = False
        
    def train(self, train_data: pd.Series, holidays_df: Optional[pd.DataFrame] = None) -> None:
        """Train all models in ensemble"""
        logger.info("Training ensemble models...")
        
        try:
            # Train Prophet
            prophet_model = ProphetModel(self.config)
            prophet_df = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
            prophet_model.train(prophet_df, holidays_df)
            self.models['prophet'] = prophet_model
            
            # Train ARIMA
            arima_model = ARIMAModel(self.config)
            arima_model.train(train_data)
            self.models['arima'] = arima_model
            
            # Train SARIMA
            sarima_model = SARIMAModel(self.config)
            sarima_model.train(train_data)
            self.models['sarima'] = sarima_model
            
            self.trained = True
            logger.info("All ensemble models trained successfully")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train ensemble models: {e}")
    
    def predict(self, steps: int) -> np.ndarray:
        """Make predictions using weighted ensemble"""
        if not self.trained:
            raise ValueError("Models not trained. Call train() first.")
        
        predictions = []
        
        for model_name, model in self.models.items():
            pred = model.predict(steps)
            predictions.append(pred * self.weights.get(model_name, 0.33))
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate ensemble performance"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        return metrics
    
    def save_model(self, path: str = 'artifacts/trained_models/ensemble_model.pkl'):
        """Save all models to disk"""
        if not self.trained:
            raise ValueError("No models to save. Train first.")
        
        for model_name, model in self.models.items():
            model.save_model(f'artifacts/trained_models/{model_name}_model.pkl')
        
        # Save weights
        joblib.dump(self.weights, f'artifacts/trained_models/ensemble_weights.pkl')
        logger.info("Ensemble models saved")