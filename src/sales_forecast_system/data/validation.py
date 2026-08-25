import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from ..utils.logger import setup_logger
from ..utils.exceptions import DataValidationError
from ..utils.config import Config

logger = setup_logger(__name__)

class DataValidator:
    """Validates data for sales forecast system"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.target_col = self.config.get('data.target_column', 'Weekly_Sales')
        self.date_col = self.config.get('data.date_column', 'Date')
        
    def validate_columns(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """Validate required columns exist"""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")
        return True
    
    def validate_data_types(self, df: pd.DataFrame, column_types: Dict[str, str]) -> bool:
        """Validate column data types"""
        for col, expected_type in column_types.items():
            if col in df.columns:
                if not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                    logger.warning(f"Column {col} expected {expected_type}, got {df[col].dtype}")
        return True
    
    def validate_date_range(self, df: pd.DataFrame, date_col: str) -> bool:
        """Validate date range is reasonable"""
        if date_col not in df.columns:
            return True
        
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        logger.info(f"Date range: {min_date} to {max_date}")
        
        if max_date > pd.Timestamp.now():
            logger.warning(f"Data contains future dates: {max_date}")
        
        return True
    
    def validate_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> bool:
        """Validate missing values are within threshold"""
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > threshold]
        
        if not high_missing.empty:
            logger.warning(f"Columns with >{threshold*100}% missing values: {high_missing.index.tolist()}")
        
        return True
    
    def validate_sales_values(self, df: pd.DataFrame, sales_col: str) -> bool:
        """Validate sales values are reasonable"""
        if sales_col not in df.columns:
            return True
        
        # Check for negative sales
        neg_sales = df[df[sales_col] < 0]
        if not neg_sales.empty:
            logger.warning(f"Found {len(neg_sales)} rows with negative sales")
        
        # Check for zero sales
        zero_sales = df[df[sales_col] == 0]
        if not zero_sales.empty:
            logger.warning(f"Found {len(zero_sales)} rows with zero sales")
        
        return True
    
    def validate_all(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Perform all validations"""
        try:
            for name, df in data.items():
                logger.info(f"Validating {name} data")
                
                if df.empty:
                    raise DataValidationError(f"{name} data is empty")
                
                if name == 'train':
                    self.validate_columns(df, ['Store', 'Dept', self.date_col, self.target_col])
                    self.validate_sales_values(df, self.target_col)
                    self.validate_date_range(df, self.date_col)
                
                elif name == 'stores':
                    self.validate_columns(df, ['Store', 'Type', 'Size'])
                
                elif name == 'features':
                    self.validate_columns(df, ['Store', self.date_col, 'Temperature', 'Fuel_Price'])
                    self.validate_date_range(df, self.date_col)
                
                self.validate_missing_values(df)
            
            logger.info("All validations passed")
            return True
            
        except Exception as e:
            raise DataValidationError(f"Validation failed: {e}")