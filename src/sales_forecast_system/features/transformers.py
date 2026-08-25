import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class FeatureTransformer:
    """Handles feature transformations"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.scalers = {}
        
    def scale_features(self, df: pd.DataFrame, 
                      columns: List[str],
                      method: str = 'standard') -> pd.DataFrame:
        """Scale selected features"""
        logger.info(f"Scaling features using {method}")
        df_copy = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        self.scalers[method] = scaler
        
        return df_copy
    
    def log_transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply log transformation"""
        logger.info(f"Applying log transformation to {column}")
        df_copy = df.copy()
        df_copy[f'{column}_log'] = np.log1p(df_copy[column])
        return df_copy
    
    def sqrt_transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply square root transformation"""
        logger.info(f"Applying sqrt transformation to {column}")
        df_copy = df.copy()
        df_copy[f'{column}_sqrt'] = np.sqrt(df_copy[column])
        return df_copy
    
    def power_transform(self, df: pd.DataFrame, column: str, 
                        power: float = 2) -> pd.DataFrame:
        """Apply power transformation"""
        logger.info(f"Applying power transformation to {column} with power {power}")
        df_copy = df.copy()
        df_copy[f'{column}_power_{power}'] = df_copy[column] ** power
        return df_copy