import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class BaselineModels:
    """Baseline models for comparison"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.seasonal_period = 52
        
    def naive_forecast(self, train_data: pd.Series, test_size: int) -> np.ndarray:
        """Naive forecast: use the last value"""
        last_value = train_data.iloc[-1]
        return np.full(test_size, last_value)
    
    def seasonal_naive_forecast(self, train_data: pd.Series, test_size: int) -> np.ndarray:
        """Seasonal naive: use same period last year"""
        predictions = []
        for i in range(test_size):
            idx = len(train_data) - self.seasonal_period + i
            if idx >= 0 and idx < len(train_data):
                predictions.append(train_data.iloc[idx])
            else:
                predictions.append(train_data.iloc[-1])
        return np.array(predictions)
    
    def moving_average_forecast(self, train_data: pd.Series, test_size: int,
                               window: int = 4) -> np.ndarray:
        """Moving average forecast"""
        last_values = train_data.iloc[-window:].values
        avg = np.mean(last_values)
        return np.full(test_size, avg)
    
    def exponential_smoothing_forecast(self, train_data: pd.Series, test_size: int,
                                     alpha: float = 0.3) -> np.ndarray:
        """Exponential smoothing forecast"""
        smoothed = [train_data.iloc[0]]
        for i in range(1, len(train_data)):
            smoothed.append(alpha * train_data.iloc[i] + (1 - alpha) * smoothed[-1])
        last_smoothed = smoothed[-1]
        return np.full(test_size, last_smoothed)
    
    def evaluate_baselines(self, train_data: pd.Series, test_data: pd.Series) -> Dict:
        """Evaluate all baseline models"""
        test_size = len(test_data)
        results = {}
        
        # Naive
        naive_pred = self.naive_forecast(train_data, test_size)
        results['Naive'] = self._calculate_metrics(test_data.values, naive_pred)
        
        # Seasonal Naive
        seasonal_naive_pred = self.seasonal_naive_forecast(train_data, test_size)
        results['Seasonal_Naive'] = self._calculate_metrics(test_data.values, seasonal_naive_pred)
        
        # Moving Average
        ma_pred = self.moving_average_forecast(train_data, test_size)
        results['Moving_Average'] = self._calculate_metrics(test_data.values, ma_pred)
        
        # Exponential Smoothing
        es_pred = self.exponential_smoothing_forecast(train_data, test_size)
        results['Exponential_Smoothing'] = self._calculate_metrics(test_data.values, es_pred)
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }