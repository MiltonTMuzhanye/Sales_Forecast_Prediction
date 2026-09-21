import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path
import joblib
import mlflow
from ..data.ingestion import DataIngestion
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..models.prophet_model import ProphetModel
from ..models.xgboost_model import XGBoostModel
from ..models.lightgbm_model import LightGBMModel
from ..models.hybrid_model import HybridModel
from ..models.baseline import BaselineModels
from ..evaluation.metrics import ModelEvaluator
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class ModelTrainer:
    """Handles model training pipeline"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.preprocessor = DataPreprocessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.evaluator = ModelEvaluator(config)
        self.models = {}
        self.results = {}
        
    def train_prophet(self, data: pd.DataFrame, store_id: int, dept_id: int) -> Dict:
        """Train Prophet model"""
        logger.info(f"Training Prophet for Store {store_id}, Dept {dept_id}")
        
        # Filter data
        store_dept_data = data[(data['Store'] == store_id) & (data['Dept'] == dept_id)].copy()
        store_dept_data = store_dept_data.sort_values('Date')
        
        # Prepare data
        prophet_data = store_dept_data[['Date', 'Weekly_Sales']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Create holidays
        if 'IsHoliday' in store_dept_data.columns:
            holidays_df = store_dept_data[store_dept_data['IsHoliday'] == 1][['Date']].copy()
            holidays_df.columns = ['ds']
            holidays_df['holiday'] = 'store_holiday'
        else:
            holidays_df = None
        
        # Split data
        train_size = int(len(prophet_data) * 0.8)
        train_data = prophet_data.iloc[:train_size]
        test_data = prophet_data.iloc[train_size:]
        
        # Train model
        model = ProphetModel(self.config)
        model.train(train_data, holidays_df)
        
        # Predict
        predictions = model.predict(len(test_data))
        y_pred = predictions['yhat'].values
        y_true = test_data['y'].values
        
        # Evaluate
        metrics = model.evaluate(y_true, y_pred)
        
        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'train_data': train_data,
            'test_data': test_data
        }
    
    def train_xgboost(self, data: pd.DataFrame, store_id: int, dept_id: int) -> Dict:
        """Train XGBoost model"""
        logger.info(f"Training XGBoost for Store {store_id}, Dept {dept_id}")
        
        # Filter data
        store_dept_data = data[(data['Store'] == store_id) & (data['Dept'] == dept_id)].copy()
        store_dept_data = store_dept_data.sort_values('Date')
        
        # Feature engineering
        store_dept_data = self.feature_engineer.engineer_all_features(store_dept_data)
        
        # Split data
        train_size = int(len(store_dept_data) * 0.8)
        train_data = store_dept_data.iloc[:train_size]
        test_data = store_dept_data.iloc[train_size:]
        
        # Train model
        model = XGBoostModel(self.config)
        model.train(train_data)
        
        # Predict
        y_pred = model.predict(test_data)
        y_true = test_data['Weekly_Sales'].values
        
        # Evaluate
        metrics = model.evaluate(y_true, y_pred)
        
        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'train_data': train_data,
            'test_data': test_data
        }
    
    def train_lightgbm(self, data: pd.DataFrame, store_id: int, dept_id: int) -> Dict:
        """Train LightGBM model"""
        logger.info(f"Training LightGBM for Store {store_id}, Dept {dept_id}")
        
        # Filter data
        store_dept_data = data[(data['Store'] == store_id) & (data['Dept'] == dept_id)].copy()
        store_dept_data = store_dept_data.sort_values('Date')
        
        # Feature engineering
        store_dept_data = self.feature_engineer.engineer_all_features(store_dept_data)
        
        # Split data
        train_size = int(len(store_dept_data) * 0.8)
        train_data = store_dept_data.iloc[:train_size]
        test_data = store_dept_data.iloc[train_size:]
        
        # Train model
        model = LightGBMModel(self.config)
        model.train(train_data)
        
        # Predict
        y_pred = model.predict(test_data)
        y_true = test_data['Weekly_Sales'].values
        
        # Evaluate
        metrics = model.evaluate(y_true, y_pred)
        
        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'train_data': train_data,
            'test_data': test_data
        }
    
    def train_all_models(self, data: pd.DataFrame, store_id: int, dept_id: int) -> Dict:
        """Train all models"""
        logger.info(f"Training all models for Store {store_id}, Dept {dept_id}")
        
        results = {}
        
        # Train Prophet
        results['prophet'] = self.train_prophet(data, store_id, dept_id)
        
        # Train XGBoost
        results['xgboost'] = self.train_xgboost(data, store_id, dept_id)
        
        # Train LightGBM
        results['lightgbm'] = self.train_lightgbm(data, store_id, dept_id)
        
        # Train Hybrid
        try:
            store_dept_data = data[(data['Store'] == store_id) & (data['Dept'] == dept_id)].copy()
            store_dept_data = store_dept_data.sort_values('Date')
            
            hybrid = HybridModel(self.config)
            hybrid.train(store_dept_data)
            
            # Predict
            y_pred = hybrid.predict(store_dept_data, 12)
            y_true = store_dept_data.tail(12)['Weekly_Sales'].values
            
            results['hybrid'] = {
                'model': hybrid,
                'predictions': y_pred,
                'metrics': hybrid.evaluate(y_true, y_pred)
            }
        except Exception as e:
            logger.warning(f"Hybrid model failed: {e}")
        
        # Compare models
        predictions = {}
        for name, result in results.items():
            if 'predictions' in result:
                predictions[name] = result['predictions']
        
        if 'test_data' in results.get('prophet', {}):
            y_true = results['prophet']['test_data']['y'].values
            comparison = self.evaluator.compare_models(predictions, y_true)
            results['comparison'] = comparison
        
        self.results = results
        return results
    
    def save_models(self, base_path: str = 'artifacts/trained_models/') -> None:
        """Save all trained models"""
        logger.info("Saving models...")
        
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        for model_name, result in self.results.items():
            if 'model' in result and hasattr(result['model'], 'save_model'):
                result['model'].save_model(f"{base_path}/{model_name}_model.pkl")
        
        # Save results
        joblib.dump(self.results, f"{base_path}/training_results.pkl")
        logger.info("All models saved")