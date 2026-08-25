import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class HyperparameterTuner:
    """Hyperparameter tuning for models"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
    def tune_xgboost(self, X: pd.DataFrame, y: pd.Series,
                    param_grid: Optional[Dict] = None) -> Dict:
        """Tune XGBoost hyperparameters"""
        logger.info("Tuning XGBoost hyperparameters...")
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=42)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def tune_lightgbm(self, X: pd.DataFrame, y: pd.Series,
                     param_grid: Optional[Dict] = None) -> Dict:
        """Tune LightGBM hyperparameters"""
        logger.info("Tuning LightGBM hyperparameters...")
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=42, verbosity=-1)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }