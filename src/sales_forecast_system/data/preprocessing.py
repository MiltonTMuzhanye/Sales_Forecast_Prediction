import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import LabelEncoder
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import DataValidationError

logger = setup_logger(__name__)

class DataPreprocessor:
    """Handles data preprocessing for sales forecast system"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.markdown_cols = self.config.get('data.markdown_features', 
                                            ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
        self.target_col = self.config.get('data.target_column', 'Weekly_Sales')
        self.date_col = self.config.get('data.date_column', 'Date')
        self.categorical_features = self.config.get('data.categorical_features', ['Type'])
        
    def merge_data(self, train_df: pd.DataFrame, stores_df: pd.DataFrame, 
                   features_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all dataframes into a single dataset"""
        logger.info("Merging data...")
        
        # Convert date columns
        train_df[self.date_col] = pd.to_datetime(train_df[self.date_col])
        features_df[self.date_col] = pd.to_datetime(features_df[self.date_col])
        
        # Merge train with features
        merged = pd.merge(
            train_df, features_df, 
            on=['Store', self.date_col], 
            how='left',
            suffixes=('', '_features')
        )
        
        # Merge with stores
        merged = pd.merge(
            merged, stores_df,
            on=['Store'],
            how='left'
        )
        
        logger.info(f"Merged dataset shape: {merged.shape}")
        return merged
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        df_copy = df.copy()
        
        # Handle markdown missing values - fill with 0
        for col in self.markdown_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(0)
        
        # Handle CPI and Unemployment - forward fill then backward fill
        for col in ['CPI', 'Unemployment']:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(method='ffill')
                df_copy[col] = df_copy[col].fillna(method='bfill')
        
        logger.info(f"Missing values after handling: {df_copy.isnull().sum().sum()}")
        return df_copy
    
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date-based features"""
        logger.info("Creating date features...")
        df_copy = df.copy()
        
        # Ensure date column is datetime
        df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col])
        
        # Create date features
        df_copy['Year'] = df_copy[self.date_col].dt.year
        df_copy['Month'] = df_copy[self.date_col].dt.month
        df_copy['Week'] = df_copy[self.date_col].dt.isocalendar().week
        df_copy['Day'] = df_copy[self.date_col].dt.day
        df_copy['DayOfWeek'] = df_copy[self.date_col].dt.dayofweek
        df_copy['Quarter'] = df_copy[self.date_col].dt.quarter
        df_copy['DayOfYear'] = df_copy[self.date_col].dt.dayofyear
        
        # Create holiday indicator
        if 'IsHoliday_y' in df_copy.columns:
            df_copy['IsHoliday'] = df_copy['IsHoliday_y'].astype(int)
        elif 'IsHoliday_x' in df_copy.columns:
            df_copy['IsHoliday'] = df_copy['IsHoliday_x'].astype(int)
        elif 'IsHoliday' in df_copy.columns:
            df_copy['IsHoliday'] = df_copy['IsHoliday'].astype(int)
        else:
            df_copy['IsHoliday'] = 0
        
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        df_copy = df.copy()
        
        for col in self.categorical_features:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[col + '_encoded'] = le.fit_transform(df_copy[col].astype(str))
                logger.info(f"Encoded {col}")
        
        return df_copy
    
    def preprocess_all(self, train_df: pd.DataFrame, stores_df: pd.DataFrame, 
                       features_df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        logger.info("Running complete preprocessing pipeline...")
        
        # Merge data
        merged = self.merge_data(train_df, stores_df, features_df)
        
        # Handle missing values
        merged = self.handle_missing_values(merged)
        
        # Create date features
        merged = self.create_date_features(merged)
        
        # Encode categorical variables
        merged = self.encode_categorical(merged)
        
        logger.info(f"Preprocessed dataset shape: {merged.shape}")
        return merged