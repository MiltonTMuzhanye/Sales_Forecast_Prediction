import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import ModelTrainingError
import joblib

logger = setup_logger(__name__)

class LightGBMModel:
    """LightGBM model wrapper"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = None
        self.feature_cols = None
        self.target_col = self.config.get('data.target_column', 'Weekly_Sales')
        
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for LightGBM"""
        logger.info("Preparing data for LightGBM...")
        
        feature_cols = [col for col in df.columns if col != self.target_col]
        feature_cols = [col for col in feature_cols if df[col].dtype != 'object']
        
        self.feature_cols = feature_cols
        X = df[feature_cols].copy().fillna(0)
        y = df[self.target_col].values
        
        return X, y
    
    def build_model(self, **kwargs) -> lgb.LGBMRegressor:
        """Build LightGBM model"""
        logger.info("Building LightGBM model...")
        
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': -1
        }
        params.update(kwargs)
        
        return lgb.LGBMRegressor(**params)
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, **kwargs) -> None:
        """Train LightGBM model"""
        logger.info("Training LightGBM model...")
        
        try:
            X, y = self.prepare_data(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            self.model = self.build_model(**kwargs)
            self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            
            logger.info("LightGBM model trained successfully")
        except Exception as e:
            raise ModelTrainingError(f"Failed to train LightGBM model: {e}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = df[self.feature_cols].copy().fillna(0)
        return self.model.predict(X)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate model"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def save_model(self, path: str = 'artifacts/trained_models/lightgbm_model.pkl'):
        """Save model"""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, path)
        joblib.dump(self.feature_cols, 'artifacts/feature_lists/lightgbm_features.pkl')
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = 'artifacts/trained_models/lightgbm_model.pkl'):
        """Load model"""
        self.model = joblib.load(path)
        self.feature_cols = joblib.load('artifacts/feature_lists/lightgbm_features.pkl')
        logger.info(f"Model loaded from {path}")