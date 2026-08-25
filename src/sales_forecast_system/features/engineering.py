import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class FeatureEngineer:
    """Handles feature engineering for sales forecast"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.target_col = self.config.get('data.target_column', 'Weekly_Sales')
        self.date_col = self.config.get('data.date_column', 'Date')
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating time features...")
        df_copy = df.copy()
        
        # Cyclical encoding
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['Month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['Month'] / 12)
        df_copy['dayofweek_sin'] = np.sin(2 * np.pi * df_copy['DayOfWeek'] / 7)
        df_copy['dayofweek_cos'] = np.cos(2 * np.pi * df_copy['DayOfWeek'] / 7)
        df_copy['quarter_sin'] = np.sin(2 * np.pi * df_copy['Quarter'] / 4)
        df_copy['quarter_cos'] = np.cos(2 * np.pi * df_copy['Quarter'] / 4)
        
        # Weekend indicator
        df_copy['is_weekend'] = (df_copy['DayOfWeek'] >= 5).astype(int)
        
        return df_copy
    
    def create_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store-based features"""
        logger.info("Creating store features...")
        df_copy = df.copy()
        
        if 'Type' in df_copy.columns:
            type_mapping = {'A': 0, 'B': 1, 'C': 2}
            df_copy['Type_encoded'] = df_copy['Type'].map(type_mapping)
        
        if 'Size' in df_copy.columns:
            df_copy['Size_log'] = np.log1p(df_copy['Size'])
            df_copy['Size_scaled'] = (df_copy['Size'] - df_copy['Size'].mean()) / df_copy['Size'].std()
        
        return df_copy
    
    def create_markdown_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create markdown-based features"""
        logger.info("Creating markdown features...")
        df_copy = df.copy()
        
        markdown_cols = self.config.get('data.markdown_features', 
                                       ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
        existing_markdowns = [col for col in markdown_cols if col in df_copy.columns]
        
        if existing_markdowns:
            df_copy['MarkDown_Total'] = df_copy[existing_markdowns].sum(axis=1)
            df_copy['MarkDown_Avg'] = df_copy[existing_markdowns].mean(axis=1)
            df_copy['MarkDown_Count'] = (df_copy[existing_markdowns] > 0).sum(axis=1)
            df_copy['MarkDown_Max'] = df_copy[existing_markdowns].max(axis=1)
            df_copy['MarkDown_Min'] = df_copy[existing_markdowns].min(axis=1)
        
        return df_copy
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday-based features"""
        logger.info("Creating holiday features...")
        df_copy = df.copy()
        
        if 'IsHoliday' in df_copy.columns:
            # Holiday leads and lags
            for i in [1, 2, 3]:
                df_copy[f'holiday_lead_{i}'] = df_copy['IsHoliday'].shift(-i).fillna(0)
                df_copy[f'holiday_lag_{i}'] = df_copy['IsHoliday'].shift(i).fillna(0)
            
            # Holiday window
            df_copy['holiday_window_3'] = df_copy['IsHoliday'].rolling(3, min_periods=1).sum().fillna(0)
            df_copy['holiday_window_5'] = df_copy['IsHoliday'].rolling(5, min_periods=1).sum().fillna(0)
        
        return df_copy
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        logger.info("Creating interaction features...")
        df_copy = df.copy()
        
        if 'Type_encoded' in df_copy.columns and 'IsHoliday' in df_copy.columns:
            df_copy['Type_Holiday'] = df_copy['Type_encoded'] * df_copy['IsHoliday']
        
        if 'Temperature' in df_copy.columns and 'IsHoliday' in df_copy.columns:
            df_copy['Temp_Holiday'] = df_copy['Temperature'] * df_copy['IsHoliday']
        
        if 'Fuel_Price' in df_copy.columns and 'IsHoliday' in df_copy.columns:
            df_copy['Fuel_Holiday'] = df_copy['Fuel_Price'] * df_copy['IsHoliday']
        
        return df_copy
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        logger.info("Running complete feature engineering pipeline...")
        
        df = self.create_time_features(df)
        df = self.create_store_features(df)
        df = self.create_markdown_features(df)
        df = self.create_holiday_features(df)
        df = self.create_interaction_features(df)
        
        logger.info(f"Engineered dataset shape: {df.shape}")
        return df