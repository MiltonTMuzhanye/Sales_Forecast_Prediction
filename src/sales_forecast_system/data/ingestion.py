import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import DataIngestionError

logger = setup_logger(__name__)

class DataIngestion:
    """Handles data ingestion for sales forecast system"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.data_path = Path(self.config.get('data.raw_path', 'data/raw/'))
        
    def load_train_data(self) -> pd.DataFrame:
        """Load training data"""
        try:
            file_path = self.data_path / self.config.get('data.train_file', 'train.csv')
            logger.info(f"Loading training data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows")
            return df
        except Exception as e:
            raise DataIngestionError(f"Failed to load train data: {e}")
    
    def load_stores_data(self) -> pd.DataFrame:
        """Load stores data"""
        try:
            file_path = self.data_path / self.config.get('data.stores_file', 'stores.csv')
            logger.info(f"Loading stores data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows")
            return df
        except Exception as e:
            raise DataIngestionError(f"Failed to load stores data: {e}")
    
    def load_features_data(self) -> pd.DataFrame:
        """Load features data"""
        try:
            file_path = self.data_path / self.config.get('data.features_file', 'features.csv')
            logger.info(f"Loading features data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows")
            return df
        except Exception as e:
            raise DataIngestionError(f"Failed to load features data: {e}")
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data files"""
        return {
            'train': self.load_train_data(),
            'stores': self.load_stores_data(),
            'features': self.load_features_data()
        }