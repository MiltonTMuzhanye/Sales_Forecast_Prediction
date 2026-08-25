import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class RollingFeatureCreator:
    """Creates rolling window features"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.target_col = self.config.get('data.target_column', 'Weekly_Sales')
        self.date_col = self.config.get('data.date_column', 'Date')
        
    def create_rolling_features(self, df: pd.DataFrame,
                               windows: List[int] = [4, 8, 12, 26, 52],
                               stats: List[str] = ['mean', 'std', 'min', 'max', 'median'],
                               group_cols: List[str] = ['Store', 'Dept']) -> pd.DataFrame:
        """Create rolling window features"""
        logger.info(f"Creating rolling features with windows: {windows}")
        df_copy = df.copy()
        df_copy = df_copy.sort_values([*group_cols, self.date_col])
        
        for window in windows:
            for stat in stats:
                col_name = f'rolling_{stat}_{window}'
                
                if stat == 'mean':
                    df_copy[col_name] = df_copy.groupby(group_cols)[self.target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                elif stat == 'std':
                    df_copy[col_name] = df_copy.groupby(group_cols)[self.target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
                elif stat == 'min':
                    df_copy[col_name] = df_copy.groupby(group_cols)[self.target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).min()
                    )
                elif stat == 'max':
                    df_copy[col_name] = df_copy.groupby(group_cols)[self.target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
                elif stat == 'median':
                    df_copy[col_name] = df_copy.groupby(group_cols)[self.target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).median()
                    )
        
        return df_copy
    
    def create_expanding_features(self, df: pd.DataFrame,
                                 stats: List[str] = ['mean', 'std'],
                                 group_cols: List[str] = ['Store', 'Dept']) -> pd.DataFrame:
        """Create expanding window features"""
        logger.info("Creating expanding features")
        df_copy = df.copy()
        df_copy = df_copy.sort_values([*group_cols, self.date_col])
        
        for stat in stats:
            col_name = f'expanding_{stat}'
            
            if stat == 'mean':
                df_copy[col_name] = df_copy.groupby(group_cols)[self.target_col].transform(
                    lambda x: x.expanding().mean()
                )
            elif stat == 'std':
                df_copy[col_name] = df_copy.groupby(group_cols)[self.target_col].transform(
                    lambda x: x.expanding().std()
                )
        
        return df_copy