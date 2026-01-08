import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureBuilder:
    """Build time series features for forecasting"""
    
    def __init__(self, data_dir='../data/processed'):
        self.data_dir = Path(data_dir)
        
    def load_data(self):
        """Load cleaned data"""
        try:
            df = pd.read_parquet(self.data_dir / 'cleaned_sales.parquet')
            logger.info(f"Loaded data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_calendar_features(self, df):
        """Create comprehensive calendar features"""
        df = df.copy()
        
        # Basic date features
        df['day_of_month'] = df['Date'].dt.day
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['week_of_year'] = df['Date'].dt.isocalendar().week
        df['quarter'] = df['Date'].dt.quarter
        
        # Holiday proximity features
        df['days_to_christmas'] = (
            pd.to_datetime(df['Date'].dt.year.astype(str) + '-12-25') - df['Date']
        ).dt.days.apply(lambda x: x if x >= 0 else x + 365)
        
        # Month start/end flags
        df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df['Date'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['Date'].dt.is_year_end.astype(int)
        
        # Season features
        df['season'] = df['Month'].apply(
            lambda x: 1 if x in [12, 1, 2] else  # Winter
                      2 if x in [3, 4, 5] else   # Spring
                      3 if x in [6, 7, 8] else   # Summer
                      4                          # Fall
        )
        
        logger.info("Created calendar features")
        return df
    
    def create_lag_features(self, df, store_id, dept_id, lags=[1, 2, 3, 4, 8, 12, 26, 52]):
        """Create lag features for specific store-department combination"""
        
        # Filter for specific store and department
        mask = (df['Store'] == store_id) & (df['Dept'] == dept_id)
        store_dept_df = df[mask].copy().sort_values('Date')
        
        # Create lag features
        for lag in lags:
            store_dept_df[f'lag_{lag}'] = store_dept_df['Weekly_Sales'].shift(lag)
        
        # Year-over-year comparison
        if 52 in lags and 53 in lags:
            store_dept_df['yoy_growth'] = (
                store_dept_df['lag_52'] / store_dept_df['lag_53'] - 1
            )
        
        logger.info(f"Created lag features for Store {store_id}, Dept {dept_id}")
        return store_dept_df
    
    def create_rolling_features(self, df, windows=[4, 8, 12, 26]):
        """Create rolling window statistics"""
        df = df.copy().sort_values('Date')
        
        for window in windows:
            df[f'rolling_mean_{window}'] = (
                df.groupby(['Store', 'Dept'])['Weekly_Sales']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            
            df[f'rolling_std_{window}'] = (
                df.groupby(['Store', 'Dept'])['Weekly_Sales']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )
            
            df[f'rolling_min_{window}'] = (
                df.groupby(['Store', 'Dept'])['Weekly_Sales']
                .rolling(window=window, min_periods=1)
                .min()
                .reset_index(level=[0, 1], drop=True)
            )
            
            df[f'rolling_max_{window}'] = (
                df.groupby(['Store', 'Dept'])['Weekly_Sales']
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )
        
        # Expanding window features
        df['expanding_mean'] = (
            df.groupby(['Store', 'Dept'])['Weekly_Sales']
            .expanding()
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        
        logger.info("Created rolling features")
        return df
    
    def create_difference_features(self, df, diffs=[1, 7, 30, 365]):
        """Create difference features (day-over-day, week-over-week, etc.)"""
        df = df.copy().sort_values('Date')
        
        for diff in diffs:
            df[f'diff_{diff}'] = (
                df.groupby(['Store', 'Dept'])['Weekly_Sales']
                .diff(diff)
            )
            
            df[f'pct_diff_{diff}'] = (
                df.groupby(['Store', 'Dept'])['Weekly_Sales']
                .pct_change(diff)
            )
        
        logger.info("Created difference features")
        return df
    
    def create_external_feature_interactions(self, df):
        """Create interactions between external features and sales"""
        df = df.copy()
        
        # Economic indicator interactions
        if 'CPI' in df.columns:
            df['cpi_to_sales_ratio'] = df['CPI'] / df['Weekly_Sales']
            df['cpi_change'] = df.groupby('Store')['CPI'].pct_change()
        
        if 'Unemployment' in df.columns:
            df['unemployment_to_sales_ratio'] = df['Unemployment'] / df['Weekly_Sales']
            df['unemployment_change'] = df.groupby('Store')['Unemployment'].pct_change()
        
        if 'Fuel_Price' in df.columns:
            df['fuel_price_to_sales_ratio'] = df['Fuel_Price'] / df['Weekly_Sales']
            df['fuel_price_change'] = df.groupby('Store')['Fuel_Price'].pct_change()
        
        # Temperature interactions
        if 'Temperature' in df.columns:
            df['temp_deviation'] = (
                df.groupby(['Store', 'Month'])['Temperature']
                .transform(lambda x: x - x.mean())
            )
            
            df['is_extreme_temp'] = (
                (df['Temperature'] < df['Temperature'].quantile(0.1)) |
                (df['Temperature'] > df['Temperature'].quantile(0.9))
            ).astype(int)
        
        logger.info("Created external feature interactions")
        return df
    
    def create_aggregated_features(self, df):
        """Create store-level and department-level aggregated features"""
        
        # Store-level aggregations
        store_stats = df.groupby(['Store', 'Date']).agg({
            'Weekly_Sales': ['mean', 'std', 'sum', 'median']
        }).reset_index()
        
        store_stats.columns = ['Store', 'Date', 
                              'store_mean_sales', 'store_std_sales',
                              'store_total_sales', 'store_median_sales']
        
        # Department-level aggregations
        dept_stats = df.groupby(['Dept', 'Date']).agg({
            'Weekly_Sales': ['mean', 'std', 'sum', 'median']
        }).reset_index()
        
        dept_stats.columns = ['Dept', 'Date',
                             'dept_mean_sales', 'dept_std_sales',
                             'dept_total_sales', 'dept_median_sales']
        
        # Merge back
        df = pd.merge(df, store_stats, on=['Store', 'Date'], how='left')
        df = pd.merge(df, dept_stats, on=['Dept', 'Date'], how='left')
        
        # Relative performance metrics
        df['store_sales_ratio'] = df['Weekly_Sales'] / df['store_mean_sales']
        df['dept_sales_ratio'] = df['Weekly_Sales'] / df['dept_mean_sales']
        df['store_percentile'] = df.groupby(['Store', 'Date'])['Weekly_Sales'].rank(pct=True)
        df['dept_percentile'] = df.groupby(['Dept', 'Date'])['Weekly_Sales'].rank(pct=True)
        
        logger.info("Created aggregated features")
        return df
    
    def create_trend_features(self, df):
        """Create trend features"""
        df = df.copy().sort_values(['Store', 'Dept', 'Date'])
        
        # Linear trend
        df['linear_trend'] = df.groupby(['Store', 'Dept']).cumcount()
        
        # Exponential moving average
        df['ema_4'] = (
            df.groupby(['Store', 'Dept'])['Weekly_Sales']
            .transform(lambda x: x.ewm(span=4).mean())
        )
        
        df['ema_12'] = (
            df.groupby(['Store', 'Dept'])['Weekly_Sales']
            .transform(lambda x: x.ewm(span=12).mean())
        )
        
        # Detrended series (residuals from moving average)
        df['detrended_8'] = df['Weekly_Sales'] - df['rolling_mean_8']
        
        logger.info("Created trend features")
        return df
    
    def build_all_features(self, store_id=None, dept_id=None):
        """Build all features for the dataset"""
        
        # Load data
        df = self.load_data()
        
        # Create basic calendar features
        df = self.create_calendar_features(df)
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Create difference features
        df = self.create_difference_features(df)
        
        # Create trend features
        df = self.create_trend_features(df)
        
        # Create external feature interactions
        df = self.create_external_feature_interactions(df)
        
        # Create aggregated features
        df = self.create_aggregated_features(df)
        
        # If specific store and department requested, add lag features
        if store_id is not None and dept_id is not None:
            df = self.create_lag_features(df, store_id, dept_id)
        
        # Handle any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Final feature set shape: {df.shape}")
        logger.info(f"Number of features: {len(df.columns)}")
        
        return df
    
    def save_features(self, df, output_path='../data/features/time_series_features.parquet'):
        """Save engineered features"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved features to {output_path}")
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise

def main():
    """Main feature engineering pipeline"""
    import sys
    sys.path.append('..')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Build features
    builder = FeatureBuilder()
    
    print("Building features for all stores and departments...")
    features_df = builder.build_all_features()
    
    # Save features
    builder.save_features(features_df)
    
    # Also build specific features for Store 1, Dept 1 (for demonstration)
    print("\nBuilding detailed features for Store 1, Department 1...")
    specific_features = builder.build_all_features(store_id=1, dept_id=1)
    
    # Save specific features
    specific_output = '../data/features/time_series_features_store_1_dept_1.parquet'
    builder.save_features(specific_features, specific_output)
    
    print("\nFeature engineering completed!")
    print(f"Total features created: {len(features_df.columns)}")
    print(f"Total records: {len(features_df)}")
    
    # Show feature categories
    feature_categories = {
        'Calendar': [col for col in features_df.columns if any(x in col for x in ['day', 'week', 'month', 'quarter', 'season', 'is_'])],
        'Lag': [col for col in features_df.columns if 'lag' in col],
        'Rolling': [col for col in features_df.columns if 'rolling' in col or 'ema' in col],
        'Difference': [col for col in features_df.columns if 'diff' in col or 'pct' in col],
        'Aggregated': [col for col in features_df.columns if 'store_' in col or 'dept_' in col or 'ratio' in col or 'percentile' in col],
        'Trend': [col for col in features_df.columns if 'trend' in col or 'detrended' in col],
        'External': [col for col in features_df.columns if col in ['CPI', 'Unemployment', 'Fuel_Price', 'Temperature', 'MarkDown'] or 'cpi' in col or 'unemployment' in col or 'fuel' in col or 'temp' in col]
    }
    
    print("\nFeature Categories:")
    for category, cols in feature_categories.items():
        print(f"  {category}: {len(cols)} features")

if __name__ == "__main__":
    main()