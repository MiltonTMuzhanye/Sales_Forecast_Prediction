from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import ModelTrainingError
import joblib

logger = setup_logger(__name__)

class ProphetModel:
    """Prophet model wrapper"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = None
        self.holidays_df = None
        self.model_params = self.config.get('model.models.prophet', {})
        
    def prepare_data(self, df: pd.DataFrame, holidays_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare data for Prophet"""
        logger.info("Preparing data for Prophet...")
        
        if 'ds' not in df.columns and 'Date' in df.columns:
            df = df.rename(columns={'Date': 'ds'})
        if 'y' not in df.columns and 'Weekly_Sales' in df.columns:
            df = df.rename(columns={'Weekly_Sales': 'y'})
        
        if holidays_df is not None:
            if 'ds' not in holidays_df.columns and 'Date' in holidays_df.columns:
                holidays_df = holidays_df.rename(columns={'Date': 'ds'})
            if 'holiday' not in holidays_df.columns:
                holidays_df['holiday'] = 'store_holiday'
            self.holidays_df = holidays_df
        
        return df
    
    def build_model(self, **kwargs) -> Prophet:
        """Build Prophet model"""
        logger.info("Building Prophet model...")
        
        params = {
            'yearly_seasonality': self.model_params.get('yearly_seasonality', True),
            'weekly_seasonality': self.model_params.get('weekly_seasonality', True),
            'daily_seasonality': self.model_params.get('daily_seasonality', False),
            'seasonality_mode': self.model_params.get('seasonality_mode', 'additive'),
            'changepoint_prior_scale': self.model_params.get('changepoint_prior_scale', 0.05),
            'holidays_prior_scale': self.model_params.get('holidays_prior_scale', 10.0),
            'seasonality_prior_scale': self.model_params.get('seasonality_prior_scale', 10.0),
            'uncertainty_samples': self.model_params.get('uncertainty_samples', 1000)
        }
        params.update(kwargs)
        
        if self.holidays_df is not None:
            params['holidays'] = self.holidays_df
        
        return Prophet(**params)
    
    def train(self, df: pd.DataFrame, holidays_df: Optional[pd.DataFrame] = None, **kwargs) -> None:
        """Train Prophet model"""
        logger.info("Training Prophet model...")
        
        try:
            train_df = self.prepare_data(df, holidays_df)
            self.model = self.build_model(**kwargs)
            self.model.fit(train_df)
            logger.info("Prophet model trained successfully")
        except Exception as e:
            raise ModelTrainingError(f"Failed to train Prophet model: {e}")
    
    def predict(self, periods: int = 12) -> pd.DataFrame:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Making predictions for {periods} periods...")
        
        try:
            future = self.model.make_future_dataframe(periods=periods, freq='W')
            forecast = self.model.predict(future)
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            return predictions
        except Exception as e:
            raise ModelTrainingError(f"Failed to make predictions: {e}")
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate model"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def save_model(self, path: str = 'artifacts/trained_models/prophet_model.pkl'):
        """Save model"""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = 'artifacts/trained_models/prophet_model.pkl'):
        """Load model"""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")