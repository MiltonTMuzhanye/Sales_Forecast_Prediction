import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class LagFeatureGenerator:
    """Generate lag features for time series forecasting"""
    
    def __init__(self, default_lags=[1, 2, 3, 4, 8, 12, 26, 52]):
        self.default_lags = default_lags
    
    def create_simple_lags(self, series, lags=None):
        """Create simple lag features from a time series"""
        if lags is None:
            lags = self.default_lags
        
        lags_df = pd.DataFrame(index=series.index)
        lags_df['original'] = series
        
        for lag in lags:
            lags_df[f'lag_{lag}'] = series.shift(lag)
        
        return lags_df
    
    def create_rolling_lags(self, series, window_sizes=[4, 8, 12]):
        """Create rolling window lag features"""
        rolling_df = pd.DataFrame(index=series.index)
        rolling_df['original'] = series
        
        for window in window_sizes:
            rolling_df[f'rolling_mean_{window}'] = series.rolling(window=window).mean().shift(1)
            rolling_df[f'rolling_std_{window}'] = series.rolling(window=window).std().shift(1)
            rolling_df[f'rolling_min_{window}'] = series.rolling(window=window).min().shift(1)
            rolling_df[f'rolling_max_{window}'] = series.rolling(window=window).max().shift(1)
        
        return rolling_df
    
    def create_expanding_lags(self, series):
        """Create expanding window lag features"""
        expanding_df = pd.DataFrame(index=series.index)
        expanding_df['original'] = series
        
        expanding_df['expanding_mean'] = series.expanding().mean().shift(1)
        expanding_df['expanding_std'] = series.expanding().std().shift(1)
        expanding_df['expanding_min'] = series.expanding().min().shift(1)
        expanding_df['expanding_max'] = series.expanding().max().shift(1)
        
        return expanding_df
    
    def create_seasonal_lags(self, series, seasonality=52):
        """Create seasonal lag features"""
        seasonal_df = pd.DataFrame(index=series.index)
        seasonal_df['original'] = series
        
        # Seasonal lags
        for i in range(1, 5):
            lag = seasonality * i
            if lag < len(series):
                seasonal_df[f'seasonal_lag_{lag}'] = series.shift(lag)
        
        # Seasonal averages
        for window in [4, 8, 12]:
            seasonal_avg = series.rolling(window=window * seasonality).mean().shift(seasonality)
            if len(seasonal_avg) > 0:
                seasonal_df[f'seasonal_avg_{window}y'] = seasonal_avg
        
        return seasonal_df
    
    def create_difference_lags(self, series, diffs=[1, 7, 30, 365]):
        """Create differenced lag features"""
        diff_df = pd.DataFrame(index=series.index)
        diff_df['original'] = series
        
        for diff in diffs:
            diff_df[f'diff_{diff}'] = series.diff(diff).shift(1)
            diff_df[f'pct_diff_{diff}'] = series.pct_change(diff).shift(1)
        
        return diff_df
    
    def create_all_lags(self, series, store_id=None, dept_id=None):
        """Create all types of lag features"""
        logger.info(f"Creating lag features for series of length {len(series)}")
        
        # Combine all lag features
        all_lags = pd.DataFrame(index=series.index)
        all_lags['target'] = series
        
        # Simple lags
        simple_lags = self.create_simple_lags(series)
        all_lags = pd.concat([all_lags, simple_lags.drop('original', axis=1)], axis=1)
        
        # Rolling lags
        rolling_lags = self.create_rolling_lags(series)
        all_lags = pd.concat([all_lags, rolling_lags.drop('original', axis=1)], axis=1)
        
        # Expanding lags
        expanding_lags = self.create_expanding_lags(series)
        all_lags = pd.concat([all_lags, expanding_lags.drop('original', axis=1)], axis=1)
        
        # Seasonal lags (if enough data)
        if len(series) > 104:  # At least 2 years of weekly data
            seasonal_lags = self.create_seasonal_lags(series)
            all_lags = pd.concat([all_lags, seasonal_lags.drop('original', axis=1)], axis=1)
        
        # Difference lags
        diff_lags = self.create_difference_lags(series)
        all_lags = pd.concat([all_lags, diff_lags.drop('original', axis=1)], axis=1)
        
        # Add metadata if provided
        if store_id is not None:
            all_lags['store_id'] = store_id
        if dept_id is not None:
            all_lags['dept_id'] = dept_id
        
        logger.info(f"Created {len(all_lags.columns)} lag features")
        return all_lags
    
    def validate_lags(self, lag_df, target_col='target'):
        """Validate that lag features don't have data leakage"""
        
        issues = []
        
        # Check for NaN values at the beginning (expected for lags)
        for col in lag_df.columns:
            if col != target_col:
                first_valid = lag_df[col].first_valid_index()
                if first_valid is not None:
                    # Check if any lag feature has data before it should
                    target_start = lag_df[target_col].first_valid_index()
                    if first_valid < target_start:
                        issues.append(f"Potential leakage in {col}: data starts before target")
        
        # Check correlation with future values (should be low)
        for col in lag_df.columns:
            if col != target_col and 'lag' in col:
                # Calculate correlation with target (should be high for proper lags)
                correlation = lag_df[col].corr(lag_df[target_col])
                if abs(correlation) < 0.1:
                    issues.append(f"Low correlation in {col}: {correlation:.3f}")
        
        return issues

def main():
    """Test lag feature generation"""
    import sys
    sys.path.append('..')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    from data.load_data import DataLoader
    
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Prepare time series for Store 1, Dept 1
    sales_data = data['sales']
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    
    store_dept_sales = sales_data[
        (sales_data['Store'] == 1) & 
        (sales_data['Dept'] == 1)
    ].sort_values('Date')
    
    series = store_dept_sales.set_index('Date')['Weekly_Sales']
    
    # Generate lag features
    lag_generator = LagFeatureGenerator()
    lag_features = lag_generator.create_all_lags(series, store_id=1, dept_id=1)
    
    print(f"Generated lag features shape: {lag_features.shape}")
    print(f"Number of features: {len(lag_features.columns)}")
    
    # Validate features
    issues = lag_generator.validate_lags(lag_features)
    if issues:
        print("\nValidation Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Lag features validated successfully")
    
    # Show sample of features
    print("\nSample of lag features:")
    print(lag_features.head().round(2))
    
    # Save features
    output_path = '../data/features/lag_features_store_1_dept_1.parquet'
    lag_features.to_parquet(output_path)
    print(f"\n✅ Lag features saved to {output_path}")

if __name__ == "__main__":
    main()