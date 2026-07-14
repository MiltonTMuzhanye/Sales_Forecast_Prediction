import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from pathlib import Path
import joblib
from ..models.arima_model import ARIMAModel
from ..models.sarima_model import SARIMAModel
from ..models.prophet_model import ProphetModel
from ..models.ensemble_model import EnsembleModel
from ..data.preprocessing import DataPreprocessor
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class InferencePipeline:
    """Inference pipeline for making predictions"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.models = {}
        self.preprocessor = DataPreprocessor(config)
        self.loaded = False
        
    def load_models(self, model_path: str = 'artifacts/trained_models/'):
        """Load trained models"""
        logger.info("Loading models...")
        
        try:
            # Load ARIMA
            arima = ARIMAModel(self.config)
            arima.load_model(f"{model_path}/arima_model.pkl")
            self.models['ARIMA'] = arima
            
            # Load SARIMA
            sarima = SARIMAModel(self.config)
            sarima.load_model(f"{model_path}/sarima_model.pkl")
            self.models['SARIMA'] = sarima
            
            # Load Prophet
            prophet = ProphetModel(self.config)
            prophet.load_model(f"{model_path}/prophet_model.pkl")
            self.models['Prophet'] = prophet
            
            # Load Ensemble
            ensemble = EnsembleModel(self.config)
            ensemble.load_model(f"{model_path}/ensemble_model.pkl")
            self.models['Ensemble'] = ensemble
            
            self.loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def predict(self, data: pd.DataFrame, periods: int = 12, 
                model_name: str = 'Prophet') -> pd.DataFrame:
        """Make predictions using specified model"""
        if not self.loaded:
            self.load_models()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Prepare data for prediction
        if model_name == 'Prophet':
            # Prepare Prophet data
            prophet_data = data[['Date', 'Weekly_Sales']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Create holidays
            if 'IsHoliday' in data.columns:
                holidays_df = data[data['IsHoliday'] == 1][['Date']].copy()
                holidays_df.columns = ['ds']
                holidays_df['holiday'] = 'store_holiday'
            else:
                holidays_df = None
            
            # Train Prophet on full data and predict
            model.train(prophet_data, holidays_df)
            predictions = model.predict(periods)
            return predictions
            
        elif model_name in ['ARIMA', 'SARIMA']:
            # Prepare time series data
            data = data.sort_values('Date')
            ts_data = data.set_index('Date')['Weekly_Sales']
            
            # Train model on full data and predict
            model.train(ts_data)
            predictions = model.predict(periods)
            
            # Create dates for predictions
            last_date = ts_data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), 
                                        periods=periods, freq='W')
            
            result_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': predictions
            })
            return result_df
            
        else:
            raise ValueError(f"Prediction not implemented for {model_name}")