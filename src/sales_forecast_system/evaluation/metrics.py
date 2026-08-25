import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import logging
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class ModelEvaluator:
    """Evaluates model performance"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'smape': np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        }
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        
        return metrics
    
    def compare_models(self, predictions: Dict[str, np.ndarray], 
                       y_true: np.ndarray) -> pd.DataFrame:
        """Compare multiple models"""
        import pandas as pd
        
        results = []
        for model_name, y_pred in predictions.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['model'] = model_name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        return df.set_index('model')