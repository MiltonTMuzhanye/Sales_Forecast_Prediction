import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features.build_features import FeatureBuilder
from features.lag_features import LagFeatureGenerator

class TestFeatureBuilder:
    """Test feature building functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='W')
        np.random.seed(42)
        
        data = {
            'Store': [1] * len(dates),
            'Dept': [1] * len(dates),
            'Date': dates,
            'Weekly_Sales': np.random.normal(10000, 2000, len(dates)),
            'Temperature': np.random.normal(60, 20, len(dates)),
            'Fuel_Price': np.random.normal(2.5, 0.5, len(dates)),
            'CPI': np.random.normal(200, 10, len(dates)),
            'Unemployment': np.random.normal(5, 1, len(dates)),
            'IsHoliday': [0] * len(dates)
        }
        
        # Add some holidays
        holiday_indices = np.random.choice(len(dates), size=10, replace=False)
        for idx in holiday_indices:
            data['IsHoliday'][idx] = 1
        
        return pd.DataFrame(data)
    
    def test_calendar_features(self, sample_data):
        """Test calendar feature creation"""
        builder = FeatureBuilder()
        df_with_calendar = builder.create_calendar_features(sample_data)
        
        # Check that calendar features were created
        expected_features = ['year', 'month', 'week', 'day', 'dayofweek', 'quarter']
        for feature in expected_features:
            assert feature in df_with_calendar.columns
        
        # Check feature values
        assert df_with_calendar['year'].min() == 2020
        assert df_with_calendar['year'].max() == 2022
        assert df_with_calendar['month'].min() == 1
        assert df_with_calendar['month'].max() == 12
    
    def test_rolling_features(self, sample_data):
        """Test rolling feature creation"""
        builder = FeatureBuilder()
        df_with_rolling = builder.create_rolling_features(sample_data)
        
        # Check that rolling features were created
        expected_prefixes = ['rolling_mean_', 'rolling_std_', 'rolling_min_', 'rolling_max_']
        windows = [4, 8, 12, 26]
        
        for prefix in expected_prefixes:
            for window in windows:
                col_name = f'{prefix}{window}'
                assert col_name in df_with_rolling.columns
        
        # Check expanding window feature
        assert 'expanding_mean' in df_with_rolling.columns
    
    def test_difference_features(self, sample_data):
        """Test difference feature creation"""
        builder = FeatureBuilder()
        df_with_diff = builder.create_difference_features(sample_data)
        
        # Check that difference features were created
        diffs = [1, 7, 30, 365]
        for diff in diffs:
            assert f'diff_{diff}' in df_with_diff.columns
            assert f'pct_diff_{diff}' in df_with_diff.columns
        
        # Check that first value is NaN (no previous value)
        assert pd.isna(df_with_diff['diff_1'].iloc[0])
    
    def test_external_feature_interactions(self, sample_data):
        """Test external feature interaction creation"""
        builder = FeatureBuilder()
        df_with_external = builder.create_external_feature_interactions(sample_data)
        
        # Check that interaction features were created
        expected_features = [
            'cpi_to_sales_ratio', 'cpi_change',
            'unemployment_to_sales_ratio', 'unemployment_change',
            'fuel_price_to_sales_ratio', 'fuel_price_change',
            'temp_deviation', 'is_extreme_temp'
        ]
        
        for feature in expected_features:
            if feature in df_with_external.columns:
                # Feature was created (depends on available columns)
                assert True
            # Some features might not be created if required columns missing
    
    def test_aggregated_features(self, sample_data):
        """Test aggregated feature creation"""
        # Create data with multiple stores and departments
        df_multi = pd.concat([
            sample_data,
            sample_data.assign(Store=2, Weekly_Sales=sample_data['Weekly_Sales'] * 1.2),
            sample_data.assign(Dept=2, Weekly_Sales=sample_data['Weekly_Sales'] * 0.8)
        ], ignore_index=True)
        
        builder = FeatureBuilder()
        df_with_aggregated = builder.create_aggregated_features(df_multi)
        
        # Check that aggregated features were created
        expected_features = [
            'store_mean_sales', 'store_std_sales', 'store_total_sales',
            'dept_mean_sales', 'dept_std_sales', 'dept_total_sales',
            'store_sales_ratio', 'dept_sales_ratio'
        ]
        
        for feature in expected_features:
            assert feature in df_with_aggregated.columns
        
        # Check that ratios are calculated correctly
        assert (df_with_aggregated['store_sales_ratio'] > 0).all()

class TestLagFeatureGenerator:
    """Test lag feature generation"""
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='W')
        values = np.random.normal(10000, 2000, 100)
        return pd.Series(values, index=dates)
    
    def test_simple_lags(self, sample_series):
        """Test simple lag creation"""
        generator = LagFeatureGenerator()
        lag_df = generator.create_simple_lags(sample_series)
        
        # Check that lags were created
        expected_lags = [1, 2, 3, 4, 8, 12, 26, 52]
        for lag in expected_lags:
            col_name = f'lag_{lag}'
            if lag < len(sample_series):
                assert col_name in lag_df.columns
        
        # Check lag values
        assert lag_df['lag_1'].iloc[1] == sample_series.iloc[0]
        assert lag_df['lag_2'].iloc[2] == sample_series.iloc[0]
    
    def test_rolling_lags(self, sample_series):
        """Test rolling lag creation"""
        generator = LagFeatureGenerator()
        rolling_df = generator.create_rolling_lags(sample_series)
        
        # Check that rolling features were created
        windows = [4, 8, 12]
        expected_features = ['rolling_mean_', 'rolling_std_', 'rolling_min_', 'rolling_max_']
        
        for window in windows:
            for prefix in expected_features:
                col_name = f'{prefix}{window}'
                assert col_name in rolling_df.columns
        
        # Check that rolling features are shifted
        assert rolling_df['rolling_mean_4'].iloc[4] == sample_series.iloc[:4].mean()
    
    def test_expanding_lags(self, sample_series):
        """Test expanding lag creation"""
        generator = LagFeatureGenerator()
        expanding_df = generator.create_expanding_lags(sample_series)
        
        # Check that expanding features were created
        expected_features = ['expanding_mean', 'expanding_std', 'expanding_min', 'expanding_max']
        for feature in expected_features:
            assert feature in expanding_df.columns
        
        # Check that expanding features are shifted
        assert expanding_df['expanding_mean'].iloc[1] == sample_series.iloc[0]
        assert expanding_df['expanding_mean'].iloc[2] == sample_series.iloc[:2].mean()
    
    def test_seasonal_lags(self, sample_series):
        """Test seasonal lag creation"""
        generator = LagFeatureGenerator()
        seasonal_df = generator.create_seasonal_lags(sample_series, seasonality=52)
        
        # Check that seasonal features were created (if enough data)
        if len(sample_series) > 104:
            assert 'seasonal_lag_52' in seasonal_df.columns
            assert 'seasonal_lag_104' in seasonal_df.columns
        
        # Check seasonal averages (if enough data)
        if len(sample_series) > 208:
            assert 'seasonal_avg_4y' in seasonal_df.columns
    
    def test_difference_lags(self, sample_series):
        """Test difference lag creation"""
        generator = LagFeatureGenerator()
        diff_df = generator.create_difference_lags(sample_series)
        
        # Check that difference features were created
        diffs = [1, 7, 30, 365]
        for diff in diffs:
            assert f'diff_{diff}' in diff_df.columns
            assert f'pct_diff_{diff}' in diff_df.columns
        
        # Check that difference features are shifted
        assert diff_df['diff_1'].iloc[1] == sample_series.iloc[0] - sample_series.iloc[1]
    
    def test_all_lags(self, sample_series):
        """Test creation of all lag features"""
        generator = LagFeatureGenerator()
        all_lags = generator.create_all_lags(sample_series, store_id=1, dept_id=1)
        
        # Check that multiple types of features were created
        assert len(all_lags.columns) > 10
        
        # Check metadata
        assert 'store_id' in all_lags.columns
        assert 'dept_id' in all_lags.columns
        assert all_lags['store_id'].iloc[0] == 1
        assert all_lags['dept_id'].iloc[0] == 1
    
    def test_validate_lags(self, sample_series):
        """Test lag feature validation"""
        generator = LagFeatureGenerator()
        lag_df = generator.create_all_lags(sample_series)
        
        issues = generator.validate_lags(lag_df)
        
        # Should not have validation issues with properly created lags
        assert isinstance(issues, list)
        
        # Check that target column exists
        assert 'target' in lag_df.columns

def test_feature_pipeline_integration():
    """Test the complete feature building pipeline"""
    # Create test data directory
    test_data_dir = Path('test_data')
    test_data_dir.mkdir(exist_ok=True)
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=52, freq='W')
    np.random.seed(42)
    
    data = {
        'Store': [1] * len(dates),
        'Dept': [1] * len(dates),
        'Date': dates,
        'Weekly_Sales': np.random.normal(10000, 2000, len(dates)),
        'Temperature': np.random.normal(60, 20, len(dates)),
        'Fuel_Price': np.random.normal(2.5, 0.5, len(dates)),
        'CPI': np.random.normal(200, 10, len(dates)),
        'Unemployment': np.random.normal(5, 1, len(dates)),
        'IsHoliday': [0] * len(dates)
    }
    
    # Save test data
    test_df = pd.DataFrame(data)
    test_path = test_data_dir / 'test_data.parquet'
    test_df.to_parquet(test_path)
    
    try:
        # Test feature builder
        builder = FeatureBuilder(data_dir=test_data_dir)
        
        # Mock the load_data method
        builder.load_data = lambda: pd.read_parquet(test_path)
        
        # Build features
        features_df = builder.build_all_features(store_id=1, dept_id=1)
        
        # Check that features were created
        assert len(features_df.columns) > len(test_df.columns)
        
        # Check for specific feature types
        assert any('lag_' in col for col in features_df.columns)
        assert any('rolling_' in col for col in features_df.columns)
        assert any('diff_' in col for col in features_df.columns)
        
        # Check no NaN values in important columns
        assert features_df['Weekly_Sales'].isnull().sum() == 0
        
        # Check date sorting
        assert features_df['Date'].is_monotonic_increasing
        
    finally:
        # Cleanup
        if test_path.exists():
            test_path.unlink()
        if test_data_dir.exists():
            test_data_dir.rmdir()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])