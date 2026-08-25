import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path
import joblib
from ..data.ingestion import DataIngestion
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..models.prophet_model import ProphetModel
from ..models.xgboost_model import XGBoostModel
from ..models.lightgbm_model import LightGBMModel
from ..models.hybrid_model import HybridModel
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class ForecastingPipeline:
    """End-to-end forecasting pipeline"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.preprocessor = DataPreprocessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.models = {}
        self.loaded = False
        
    def load_models(self, model_path: str = 'artifacts/trained_models/') -> None:
        """Load all trained models"""
        logger.info("Loading models...")
        
        try:
            # Load Prophet
            prophet = ProphetModel(self.config)
            prophet.load_model(f"{model_path}/prophet_model.pkl")
            self.models['prophet'] = prophet
            
            # Load XGBoost
            xgb = XGBoostModel(self.config)
            xgb.load_model(f"{model_path}/xgboost_model.pkl")
            self.models['xgboost'] = xgb
            
            # Load LightGBM
            lgb = LightGBMModel(self.config)
            lgb.load_model(f"{model_path}/lightgbm_model.pkl")
            self.models['lightgbm'] = lgb
            
            # Load Hybrid
            try:
                hybrid = HybridModel(self.config)
                hybrid.load_model(f"{model_path}/hybrid_model/")
                self.models['hybrid'] = hybrid
            except:
                logger.warning("Hybrid model not found")
            
            self.loaded = True
            logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def forecast(self, store_id: int, dept_id: int, periods: int = 12,
                model_name: str = 'prophet') -> Dict:
        """Generate forecast for a specific store and department"""
        logger.info(f"Generating forecast for Store {store_id}, Dept {dept_id}")
        
        if not self.loaded:
            self.load_models()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        try:
            # Load data
            ingestion = DataIngestion(self.config)
            data = ingestion.load_all_data()
            
            # Preprocess data
            processed_data = self.preprocessor.preprocess_all(
                data['train'], data['stores'], data['features']
            )
            
            # Filter data
            store_dept_data = processed_data[
                (processed_data['Store'] == store_id) & 
                (processed_data['Dept'] == dept_id)
            ]
            
            if len(store_dept_data) == 0:
                raise ValueError(f"No data found for Store {store_id}, Dept {dept_id}")
            
            # Make predictions
            if model_name == 'prophet':
                prophet_data = store_dept_data[['Date', 'Weekly_Sales']].copy()
                prophet_data.columns = ['ds', 'y']
                
                if 'IsHoliday' in store_dept_data.columns:
                    holidays_df = store_dept_data[store_dept_data['IsHoliday'] == 1][['Date']].copy()
                    holidays_df.columns = ['ds']
                    holidays_df['holiday'] = 'store_holiday'
                else:
                    holidays_df = None
                
                model.train(prophet_data, holidays_df)
                predictions_df = model.predict(periods)
                predictions = predictions_df['yhat'].values
                dates = predictions_df['ds'].values
                
            elif model_name in ['xgboost', 'lightgbm']:
                store_dept_data = self.feature_engineer.engineer_all_features(store_dept_data)
                predictions = model.predict(store_dept_data)
                dates = store_dept_data['Date'].values[-len(predictions):]
                
            elif model_name == 'hybrid':
                predictions = model.predict(store_dept_data, periods)
                last_date = store_dept_data['Date'].max()
                dates = pd.date_range(start=last_date + pd.Timedelta(days=7), 
                                     periods=periods, freq='W')
            
            result = {
                'store_id': store_id,
                'department_id': dept_id,
                'model': model_name,
                'periods': periods,
                'dates': dates.tolist(),
                'predictions': predictions.tolist(),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Forecast failed: {e}")
            raise