import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate raw data files"""
    
    def __init__(self, data_dir='../data/raw'):
        self.data_dir = Path(data_dir)
        
    def load_sales_data(self, file_name='train.csv'):
        """Load sales transaction data"""
        try:
            file_path = self.data_dir / file_name
            df = pd.read_csv(file_path)
            logger.info(f"Loaded sales data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading sales data: {e}")
            raise
    
    def load_store_data(self, file_name='stores.csv'):
        """Load store information"""
        try:
            file_path = self.data_dir / file_name
            df = pd.read_csv(file_path)
            logger.info(f"Loaded store data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading store data: {e}")
            raise
    
    def load_features_data(self, file_name='features.csv'):
        """Load external features"""
        try:
            file_path = self.data_dir / file_name
            df = pd.read_csv(file_path)
            logger.info(f"Loaded features data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading features data: {e}")
            raise
    
    def load_all_data(self):
        """Load all datasets"""
        sales_data = self.load_sales_data()
        store_data = self.load_store_data()
        features_data = self.load_features_data()
        
        return {
            'sales': sales_data,
            'stores': store_data,
            'features': features_data
        }

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    data = loader.load_all_data()
    print("Data loaded successfully:")
    for name, df in data.items():
        print(f"{name}: {df.shape}")