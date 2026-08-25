import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class CalendarFeatureCreator:
    """Creates calendar and time-based features"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.date_col = self.config.get('data.date_column', 'Date')
        
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encoding for time-based features"""
        logger.info("Creating cyclical features")
        df_copy = df.copy()
        
        # Month cyclical encoding
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['Month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['Month'] / 12)
        
        # Day of week cyclical encoding
        df_copy['dayofweek_sin'] = np.sin(2 * np.pi * df_copy['DayOfWeek'] / 7)
        df_copy['dayofweek_cos'] = np.cos(2 * np.pi * df_copy['DayOfWeek'] / 7)
        
        # Quarter cyclical encoding
        df_copy['quarter_sin'] = np.sin(2 * np.pi * df_copy['Quarter'] / 4)
        df_copy['quarter_cos'] = np.cos(2 * np.pi * df_copy['Quarter'] / 4)
        
        return df_copy
    
    def create_seasonal_features(self, df: pd.DataFrame, 
                                harmonics: int = 3) -> pd.DataFrame:
        """Create seasonal features using Fourier terms"""
        logger.info("Creating seasonal features")
        df_copy = df.copy()
        
        t = np.arange(len(df_copy))
        
        # Annual seasonality (52 weeks)
        for k in range(1, harmonics + 1):
            df_copy[f'sin_annual_{k}'] = np.sin(2 * np.pi * k * t / 52)
            df_copy[f'cos_annual_{k}'] = np.cos(2 * np.pi * k * t / 52)
        
        # Quarterly seasonality (13 weeks)
        for k in range(1, harmonics):
            df_copy[f'sin_quarterly_{k}'] = np.sin(2 * np.pi * k * t / 13)
            df_copy[f'cos_quarterly_{k}'] = np.cos(2 * np.pi * k * t / 13)
        
        return df_copy
    
    def create_all_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all calendar features"""
        logger.info("Creating all calendar features")
        df = self.create_cyclical_features(df)
        df = self.create_seasonal_features(df)
        return df