import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and preprocess retail data"""
    
    def __init__(self):
        self.markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    
    def merge_datasets(self, sales_df, features_df, stores_df):
        """Merge all datasets on common keys"""
        try:
            # Convert date columns
            sales_df['Date'] = pd.to_datetime(sales_df['Date'])
            features_df['Date'] = pd.to_datetime(features_df['Date'])
            
            # Merge features with sales
            merged = pd.merge(sales_df, features_df, 
                            on=['Store', 'Date'], 
                            how='left', 
                            suffixes=('', '_feature'))
            
            # Merge with stores
            merged = pd.merge(merged, stores_df, 
                            on=['Store'], 
                            how='left')
            
            logger.info(f"Merged dataset shape: {merged.shape}")
            return merged
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        try:
            # Fill markdown columns with 0
            for col in self.markdown_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            # Forward fill and backward fill for time-series columns
            time_series_cols = ['CPI', 'Unemployment', 'Temperature', 'Fuel_Price']
            for col in time_series_cols:
                if col in df.columns:
                    df[col] = df.groupby('Store')[col].transform(
                        lambda x: x.ffill().bfill()
                    )
            
            # Check for remaining missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                logger.warning(f"Remaining missing values:\n{missing[missing > 0]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
    
    def create_features(self, df):
        """Create additional time-based features"""
        try:
            df = df.copy()
            
            # Extract date components
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Week'] = df['Date'].dt.isocalendar().week
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Quarter'] = df['Date'].dt.quarter
            
            # Create holiday flag (consolidate from both sources)
            if 'IsHoliday' in df.columns and 'IsHoliday_feature' in df.columns:
                df['IsHoliday'] = df['IsHoliday'].fillna(df['IsHoliday_feature'])
            elif 'IsHoliday_feature' in df.columns:
                df['IsHoliday'] = df['IsHoliday_feature']
            
            # Convert holiday to integer
            df['IsHoliday'] = df['IsHoliday'].astype(int)
            
            # Create month-year identifier
            df['MonthYear'] = df['Date'].dt.strftime('%Y-%m')
            
            logger.info(f"Created {len(df.columns) - len(['Year', 'Month', 'Week', 'Day', 'DayOfWeek', 'Quarter', 'IsHoliday', 'MonthYear'])} new features")
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise
    
    def remove_outliers(self, df, column='Weekly_Sales', method='iqr', threshold=3):
        """Remove outliers from specified column"""
        try:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                original_size = len(df)
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                removed = original_size - len(df)
                
                logger.info(f"Removed {removed} outliers ({removed/original_size*100:.2f}%) using IQR method")
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[column]))
                original_size = len(df)
                df = df[z_scores < threshold]
                removed = original_size - len(df)
                
                logger.info(f"Removed {removed} outliers ({removed/original_size*100:.2f}%) using Z-score method")
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            raise
    
    def save_cleaned_data(self, df, output_path):
        """Save cleaned data to parquet format"""
        try:
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved cleaned data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
            raise

def main():
    """Main cleaning pipeline"""
    import sys
    sys.path.append('..')
    
    from data.load_data import DataLoader
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Clean data
    cleaner = DataCleaner()
    
    # Merge datasets
    merged_data = cleaner.merge_datasets(
        data['sales'],
        data['features'],
        data['stores']
    )
    
    # Handle missing values
    cleaned_data = cleaner.handle_missing_values(merged_data)
    
    # Create features
    cleaned_data = cleaner.create_features(cleaned_data)
    
    # Remove outliers (optional)
    # cleaned_data = cleaner.remove_outliers(cleaned_data, 'Weekly_Sales')
    
    # Save cleaned data
    output_path = '../data/processed/cleaned_sales.parquet'
    cleaner.save_cleaned_data(cleaned_data, output_path)
    
    print("Data cleaning completed successfully!")
    print(f"Final dataset shape: {cleaned_data.shape}")
    print(f"Columns: {len(cleaned_data.columns)}")

if __name__ == "__main__":
    main()