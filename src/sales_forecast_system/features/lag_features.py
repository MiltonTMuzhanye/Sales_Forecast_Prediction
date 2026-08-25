import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class LagFeatureCreator:
    """Creates lag features for time series forecasting"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.target_col = self.config.get('data.target_column', 'Weekly_Sales')
        self.date_col = self.config.get('data.date_column', 'Date')
        
    def create_lag_features(self, df: pd.DataFrame, 
                           lags: List[int] = [1, 2, 3, 4, 8, 12, 26, 52],
                           group_cols: List[str] = ['Store', 'Dept']) -> pd.DataFrame:
        """Create lag features"""
        logger.info(f"Creating lag features: {lags}")
        df_copy = df.copy()
        df_copy = df_copy.sort_values([*group_cols, self.date_col])
        
        for lag in lags:
            col_name = f'lag_{lag}'
            df_copy[col_name] = df_copy.groupby(group_cols)[self.target_col].shift(lag)
        
        return df_copy
    
    def create_multi_step_lags(self, df: pd.DataFrame, 
                              max_lag: int = 52,
                              step: int = 1,
                              group_cols: List[str] = ['Store', 'Dept']) -> pd.DataFrame:
        """Create multiple lag features"""
        lags = list(range(step, max_lag + 1, step))
        return self.create_lag_features(df, lags, group_cols)