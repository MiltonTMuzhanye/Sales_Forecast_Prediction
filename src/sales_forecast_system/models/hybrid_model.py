import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import ModelTrainingError
from .prophet_model import ProphetModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
import joblib
import os

logger = setup_logger(__name__)

class HybridModel:
    """Hybrid model combining Prophet and ML"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.prophet_model = ProphetModel(config)
        self.ml_model = None
        self.use_xgboost = self.config.get('model.models.hybrid.use_xgboost', True)
        self.trained = False
        
    def train(self, df: pd.DataFrame, target_col: str = 'Weekly_Sales',
              date_col: str = 'Date', holiday_col: str = 'IsHoliday') -> None:
        """Train hybrid model"""
        logger.info("Training hybrid model...")
        
        try:
            # Step 1: Train Prophet
            logger.info("Step 1: Training Prophet model...")
            prophet_data = df[[date_col, target_col]].copy()
            prophet_data.columns = ['ds', 'y']
            
            if holiday_col in df.columns:
                holidays_df = df[df[holiday_col] == 1][[date_col]].copy()
                holidays_df.columns = ['ds']
                holidays_df['holiday'] = 'store_holiday'
            else:
                holidays_df = None
            
            self.prophet_model.train(prophet_data, holidays_df)
            prophet_pred = self.prophet_model.predict(periods=0)
            prophet_fitted = prophet_pred['yhat'].values[:len(df)]
            
            # Step 2: Train ML on residuals
            logger.info("Step 2: Training ML model on residuals...")
            ml_data = df.copy()
            ml_data['prophet_prediction'] = prophet_fitted
            ml_data['residual'] = ml_data[target_col] - ml_data['prophet_prediction']
            
            if self.use_xgboost:
                self.ml_model = XGBoostModel(self.config)
            else:
                self.ml_model = LightGBMModel(self.config)
            
            self.ml_model.train(ml_data, target_col='residual', test_size=0.2)
            
            self.trained = True
            logger.info("Hybrid model trained successfully")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train hybrid model: {e}")
    
    def predict(self, df: pd.DataFrame, periods: int = 12) -> np.ndarray:
        """Make predictions"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        logger.info(f"Making predictions for {periods} periods...")
        
        try:
            # Prophet predictions
            prophet_pred_df = self.prophet_model.predict(periods)
            prophet_predictions = prophet_pred_df['yhat'].values
            
            # ML predictions on residuals
            last_row = df.iloc[-1:].copy()
            future_df = pd.concat([last_row] * periods, ignore_index=True)
            
            # Add future dates
            last_date = df[self.config.get('data.date_column', 'Date')].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), 
                                        periods=periods, freq='W')
            future_df[self.config.get('data.date_column', 'Date')] = future_dates
            
            residual_predictions = self.ml_model.predict(future_df)
            
            # Combine
            final_predictions = prophet_predictions + residual_predictions
            return final_predictions
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to make predictions: {e}")
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate model"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def save_model(self, path: str = 'artifacts/trained_models/hybrid_model/'):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        self.prophet_model.save_model(os.path.join(path, 'prophet_model.pkl'))
        self.ml_model.save_model(os.path.join(path, 'ml_model.pkl'))
        joblib.dump({'use_xgboost': self.use_xgboost, 'trained': self.trained}, 
                   os.path.join(path, 'config.pkl'))
        logger.info(f"Hybrid model saved to {path}")
    
    def load_model(self, path: str = 'artifacts/trained_models/hybrid_model/'):
        """Load model"""
        config_data = joblib.load(os.path.join(path, 'config.pkl'))
        self.use_xgboost = config_data['use_xgboost']
        self.trained = config_data['trained']
        
        self.prophet_model.load_model(os.path.join(path, 'prophet_model.pkl'))
        
        if self.use_xgboost:
            self.ml_model = XGBoostModel(self.config)
        else:
            self.ml_model = LightGBMModel(self.config)
        
        self.ml_model.load_model(os.path.join(path, 'ml_model.pkl'))
        logger.info(f"Hybrid model loaded from {path}")