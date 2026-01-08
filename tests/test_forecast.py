import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.train import ModelTrainer
from models.forecast import ForecastGenerator
from models.evaluate import ModelEvaluator

class TestModelTrainer:
    """Test model training functionality"""
    
    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='W')
        
        # Create trend + seasonality + noise
        trend = np.linspace(10000, 15000, len(dates))
        seasonality = 2000 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
        noise = np.random.normal(0, 500, len(dates))
        
        values = trend + seasonality + noise
        return pd.Series(values, index=dates, name='Weekly_Sales')
    
    @pytest.fixture
    def sample_features_data(self):
        """Create sample features data"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='W')
        
        data = {
            'Store': [1] * len(dates),
            'Dept': [1] * len(dates),
            'Date': dates,
            'Weekly_Sales': np.random.normal(10000, 2000, len(dates)),
            'IsHoliday': [0] * len(dates)
        }
        
        # Add some holidays
        holiday_indices = np.random.choice(len(dates), size=10, replace=False)
        for idx in holiday_indices:
            data['IsHoliday'][idx] = 1
        
        return pd.DataFrame(data)
    
    def test_train_test_split(self, sample_time_series):
        """Test time series train-test split"""
        trainer = ModelTrainer()
        train_data, test_data = trainer.train_test_split(sample_time_series, test_size=0.2)
        
        # Check sizes
        assert len(train_data) == 80  # 80% of 100
        assert len(test_data) == 20   # 20% of 100
        
        # Check no overlap
        assert train_data.index.max() < test_data.index.min()
        
        # Check type preservation
        assert isinstance(train_data, pd.Series)
        assert isinstance(test_data, pd.Series)
    
    def test_arima_training(self, sample_time_series):
        """Test ARIMA model training"""
        trainer = ModelTrainer()
        
        # Split data
        train_data, test_data = trainer.train_test_split(sample_time_series)
        
        # Train model
        model = trainer.train_arima(train_data, order=(1, 1, 1))
        
        # Check model was trained
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'forecast')
        
        # Test forecasting
        predictions, metrics = trainer.evaluate_model(
            model, 'arima', train_data, test_data
        )
        
        # Check predictions
        assert len(predictions) == len(test_data)
        assert isinstance(predictions, pd.Series)
        
        # Check metrics
        assert 'mape' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert isinstance(metrics['mape'], float)
    
    def test_prophet_training(self, sample_features_data):
        """Test Prophet model training"""
        trainer = ModelTrainer()
        
        # Prepare data
        store_dept_data = sample_features_data
        ts_data = store_dept_data.set_index('Date')['Weekly_Sales']
        
        # Split data
        train_data, test_data = trainer.train_test_split(ts_data)
        
        # Train model
        model = trainer.train_prophet(store_dept_data, train_data)
        
        # Check model was trained
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # Test forecasting
        predictions, metrics = trainer.evaluate_model(
            model, 'prophet', train_data, test_data, store_dept_data
        )
        
        # Check predictions
        assert len(predictions) == len(test_data)
        
        # Check metrics
        assert 'mape' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
    
    def test_time_series_cv(self, sample_time_series):
        """Test time series cross-validation"""
        trainer = ModelTrainer()
        
        # Run cross-validation
        cv_results = trainer.time_series_cv(
            sample_time_series, 
            model_type='prophet',
            n_splits=3,
            test_size=4
        )
        
        # Check results structure
        assert 'mean_mape' in cv_results
        assert 'std_mape' in cv_results
        assert 'scores' in cv_results
        
        # Check values
        assert isinstance(cv_results['mean_mape'], float)
        assert isinstance(cv_results['std_mape'], float)
        assert len(cv_results['scores']) == 3
        
        # Check scores are positive
        assert all(score >= 0 for score in cv_results['scores'])
    
    def test_model_saving(self, sample_time_series, tmp_path):
        """Test model saving functionality"""
        trainer = ModelTrainer()
        trainer.models_dir = tmp_path
        
        # Train a simple model
        train_data, _ = trainer.train_test_split(sample_time_series)
        model = trainer.train_arima(train_data)
        
        # Save model
        saved_path = trainer.save_model(model, 'arima', 1, 1)
        
        # Check file was created
        assert saved_path.exists()
        assert saved_path.suffix == '.pkl'
        
        # Check latest reference was created
        latest_path = tmp_path / 'arima_store_1_dept_1_latest.pkl'
        assert latest_path.exists()
    
    def test_metadata_saving(self, tmp_path):
        """Test metadata saving functionality"""
        trainer = ModelTrainer()
        trainer.models_dir = tmp_path
        
        # Create sample metadata
        metadata = {
            'store_id': 1,
            'dept_id': 1,
            'training_date': datetime.now().isoformat(),
            'performance': {'mape': 15.5, 'mae': 1000}
        }
        
        # Save metadata
        trainer.save_metadata(metadata, 1, 1)
        
        # Check file was created
        metadata_path = tmp_path / 'metadata_store_1_dept_1.json'
        assert metadata_path.exists()
        
        # Check content
        import json
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata['store_id'] == 1
        assert loaded_metadata['dept_id'] == 1

class TestForecastGenerator:
    """Test forecast generation functionality"""
    
    @pytest.fixture
    def sample_prophet_model(self):
        """Create a sample Prophet model for testing"""
        from prophet import Prophet
        
        # Create synthetic data
        dates = pd.date_range(start='2020-01-01', periods=52, freq='W')
        df = pd.DataFrame({
            'ds': dates,
            'y': np.random.normal(10000, 2000, len(dates))
        })
        
        # Train simple model
        model = Prophet()
        model.fit(df)
        
        return model
    
    @pytest.fixture
    def sample_arima_model(self, sample_time_series):
        """Create a sample ARIMA model for testing"""
        from statsmodels.tsa.arima.model import ARIMA
        
        model = ARIMA(sample_time_series, order=(1, 1, 1))
        model_fit = model.fit()
        
        return model_fit
    
    def test_prophet_forecast_generation(self, sample_prophet_model):
        """Test Prophet forecast generation"""
        generator = ForecastGenerator()
        
        # Generate forecast
        forecast_df = generator.generate_prophet_forecast(
            sample_prophet_model, 
            periods=12
        )
        
        # Check results
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == 12
        assert 'forecast' in forecast_df.columns
        assert 'lower_bound' in forecast_df.columns
        assert 'upper_bound' in forecast_df.columns
        
        # Check index is datetime
        assert isinstance(forecast_df.index, pd.DatetimeIndex)
        
        # Check values are reasonable
        assert (forecast_df['forecast'] > 0).all()
        assert (forecast_df['lower_bound'] < forecast_df['upper_bound']).all()
    
    def test_arima_forecast_generation(self, sample_arima_model):
        """Test ARIMA forecast generation"""
        generator = ForecastGenerator()
        
        # Generate forecast
        forecast_df = generator.generate_arima_forecast(
            sample_arima_model,
            periods=8
        )
        
        # Check results
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == 8
        assert 'forecast' in forecast_df.columns
        
        # Check confidence intervals
        if 'lower_bound' in forecast_df.columns:
            assert (forecast_df['lower_bound'] < forecast_df['upper_bound']).all()
    
    def test_ensemble_forecast_generation(self):
        """Test ensemble forecast generation"""
        generator = ForecastGenerator()
        
        # Create sample forecasts
        dates = pd.date_range(start='2023-01-01', periods=12, freq='W')
        
        forecast1 = pd.DataFrame({
            'date': dates,
            'forecast': np.random.normal(10000, 1000, 12),
            'lower_bound': np.random.normal(8000, 1000, 12),
            'upper_bound': np.random.normal(12000, 1000, 12)
        }).set_index('date')
        
        forecast2 = pd.DataFrame({
            'date': dates,
            'forecast': np.random.normal(11000, 1000, 12),
            'lower_bound': np.random.normal(9000, 1000, 12),
            'upper_bound': np.random.normal(13000, 1000, 12)
        }).set_index('date')
        
        # Generate ensemble
        forecasts = {'model1': forecast1, 'model2': forecast2}
        ensemble_df = generator.generate_ensemble_forecast(forecasts)
        
        # Check results
        assert isinstance(ensemble_df, pd.DataFrame)
        assert len(ensemble_df) == 12
        assert 'forecast' in ensemble_df.columns
        
        # Check ensemble calculation
        expected_forecast = (forecast1['forecast'] + forecast2['forecast']) / 2
        assert np.allclose(ensemble_df['forecast'], expected_forecast, rtol=1e-5)
    
    def test_forecast_metrics_calculation(self):
        """Test forecast metrics calculation"""
        generator = ForecastGenerator()
        
        # Create sample forecast
        dates = pd.date_range(start='2023-01-01', periods=12, freq='W')
        forecast_df = pd.DataFrame({
            'date': dates,
            'forecast': np.linspace(10000, 15000, 12),
            'lower_bound': np.linspace(8000, 13000, 12),
            'upper_bound': np.linspace(12000, 17000, 12)
        }).set_index('date')
        
        # Calculate metrics
        metrics = generator.calculate_forecast_metrics(forecast_df)
        
        # Check metrics
        expected_metrics = [
            'forecast_periods', 'forecast_start', 'forecast_end',
            'total_forecast', 'mean_forecast', 'std_forecast',
            'min_forecast', 'max_forecast'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check values
        assert metrics['forecast_periods'] == 12
        assert metrics['total_forecast'] == forecast_df['forecast'].sum()
        assert metrics['mean_forecast'] == forecast_df['forecast'].mean()
        
        # Check uncertainty metrics
        if 'lower_bound' in forecast_df.columns:
            assert 'mean_uncertainty_range' in metrics
            assert 'max_uncertainty_range' in metrics
    
    def test_forecast_validation(self):
        """Test forecast validation"""
        generator = ForecastGenerator()
        
        # Create valid forecast
        dates = pd.date_range(start='2023-01-01', periods=12, freq='W')
        valid_forecast = pd.DataFrame({
            'date': dates,
            'forecast': np.random.normal(10000, 1000, 12),
            'lower_bound': np.random.normal(8000, 1000, 12),
            'upper_bound': np.random.normal(12000, 1000, 12)
        }).set_index('date')
        
        # Validate
        issues = generator.validate_forecast(valid_forecast)
        
        # Should have no issues with valid forecast
        assert isinstance(issues, list)
        
        # Create invalid forecast with negative values
        invalid_forecast = valid_forecast.copy()
        invalid_forecast.loc[invalid_forecast.index[0], 'forecast'] = -1000
        
        # Validate
        issues = generator.validate_forecast(invalid_forecast)
        
        # Should detect negative forecasts
        assert any('negative' in issue.lower() for issue in issues)
        
        # Create invalid forecast with confidence interval issues
        invalid_bounds = valid_forecast.copy()
        invalid_bounds.loc[invalid_bounds.index[0], 'lower_bound'] = 12000
        invalid_bounds.loc[invalid_bounds.index[0], 'upper_bound'] = 8000
        
        # Validate
        issues = generator.validate_forecast(invalid_bounds)
        
        # Should detect invalid bounds
        assert any('invalid' in issue.lower() for issue in issues)

class TestModelEvaluator:
    """Test model evaluation functionality"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions data"""
        dates = pd.date_range(start='2022-01-01', periods=20, freq='W')
        
        np.random.seed(42)
        actual = np.random.normal(10000, 2000, 20)
        
        # Create predictions with some error
        prophet_pred = actual + np.random.normal(500, 300, 20)
        arima_pred = actual + np.random.normal(800, 400, 20)
        
        df = pd.DataFrame({
            'Date': dates,
            'Actual': actual,
            'Prophet_Prediction': prophet_pred,
            'ARIMA_Prediction': arima_pred,
            'IsHoliday': [0] * 20
        })
        
        # Add some holidays
        df.loc[df.index[5], 'IsHoliday'] = 1
        df.loc[df.index[15], 'IsHoliday'] = 1
        
        return df
    
    def test_error_metrics_calculation(self, sample_predictions):
        """Test error metrics calculation"""
        evaluator = ModelEvaluator()
        
        # Calculate metrics for Prophet
        metrics = evaluator.calculate_error_metrics(
            sample_predictions['Actual'],
            sample_predictions['Prophet_Prediction'],
            'prophet'
        )
        
        # Check metrics structure
        expected_metrics = [
            'model', 'mape', 'mae', 'rmse', 'mse', 'bias',
            'std_error', 'mad', 'max_error'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check values
        assert metrics['model'] == 'prophet'
        assert isinstance(metrics['mape'], float)
        assert metrics['mape'] >= 0
        assert isinstance(metrics['direction_accuracy'], (float, type(None)))
    
    def test_error_distribution_analysis(self, sample_predictions):
        """Test error distribution analysis"""
        evaluator = ModelEvaluator()
        
        errors = sample_predictions['Prophet_Prediction'] - sample_predictions['Actual']
        analysis = evaluator.analyze_error_distribution(errors)
        
        # Check analysis structure
        expected_stats = [
            'mean', 'median', 'std', 'skewness', 'kurtosis',
            'q1', 'q3', 'iqr', 'range'
        ]
        
        for stat in expected_stats:
            assert stat in analysis
        
        # Check values
        assert isinstance(analysis['mean'], float)
        assert isinstance(analysis['std'], float)
        assert analysis['iqr'] >= 0
    
    def test_error_by_horizon_analysis(self, sample_predictions):
        """Test error by horizon analysis"""
        evaluator = ModelEvaluator()
        
        analysis = evaluator.analyze_error_by_horizon(sample_predictions)
        
        # Check analysis structure
        assert isinstance(analysis, pd.DataFrame)
        assert 'horizon' in analysis.columns
        assert 'metrics' in analysis.columns
        
        # Check horizons
        assert analysis['horizon'].min() == 1
        assert analysis['horizon'].max() <= len(sample_predictions)
        
        # Check metrics for each horizon
        for _, row in analysis.iterrows():
            metrics = row['metrics']
            assert isinstance(metrics, dict)
            
            if 'prophet' in metrics:
                assert 'mape' in metrics['prophet']
                assert 'bias' in metrics['prophet']
                assert 'std_error' in metrics['prophet']
    
    def test_error_by_condition_analysis(self, sample_predictions):
        """Test error by condition analysis"""
        evaluator = ModelEvaluator()
        
        analysis = evaluator.analyze_error_by_condition(sample_predictions)
        
        # Check analysis structure
        assert isinstance(analysis, dict)
        
        # Check holiday analysis if column exists
        if 'IsHoliday' in sample_predictions.columns:
            assert 'holiday' in analysis
        
        # Check sales volume analysis
        assert 'sales_volume' in analysis
    
    def test_forecast_bias_detection(self, sample_predictions):
        """Test forecast bias detection"""
        evaluator = ModelEvaluator()
        
        bias_analysis = evaluator.detect_forecast_bias(sample_predictions, 'Prophet')
        
        # Check analysis structure
        assert isinstance(bias_analysis, dict)
        
        expected_stats = [
            'mean_bias', 'median_bias', 'percent_positive',
            'percent_negative', 't_test_pvalue', 'is_significant_bias'
        ]
        
        for stat in expected_stats:
            assert stat in bias_analysis
        
        # Check values
        assert isinstance(bias_analysis['mean_bias'], float)
        assert isinstance(bias_analysis['t_test_pvalue'], float)
        assert bias_analysis['percent_positive'] >= 0
        assert bias_analysis['percent_positive'] <= 100
    
    def test_performance_report_generation(self, sample_predictions):
        """Test performance report generation"""
        evaluator = ModelEvaluator()
        
        report = evaluator.generate_performance_report(
            sample_predictions, 1, 1
        )
        
        # Check report structure
        expected_sections = [
            'store_id', 'dept_id', 'evaluation_date',
            'summary', 'model_performance'
        ]
        
        for section in expected_sections:
            assert section in report
        
        # Check model performance
        assert 'prophet' in report['model_performance']
        assert 'arima' in report['model_performance']
        
        # Check metrics for each model
        for model_metrics in report['model_performance'].values():
            assert 'mape' in model_metrics
            assert 'mae' in model_metrics
            assert 'rmse' in model_metrics
        
        # Check summary
        assert 'best_model' in report['summary']
        assert 'best_mape' in report['summary']

def test_full_pipeline_integration(tmp_path):
    """Test integration of training, forecasting, and evaluation"""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=104, freq='W')
    
    # Create synthetic sales data with trend and seasonality
    trend = np.linspace(8000, 12000, len(dates))
    seasonality = 2000 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
    noise = np.random.normal(0, 500, len(dates))
    sales = trend + seasonality + noise
    
    # Create features dataframe
    features_df = pd.DataFrame({
        'Store': [1] * len(dates),
        'Dept': [1] * len(dates),
        'Date': dates,
        'Weekly_Sales': sales,
        'IsHoliday': [0] * len(dates)
    })
    
    # Add holidays
    holiday_months = [2, 9, 11, 12]  # Super Bowl, Labor Day, Thanksgiving, Christmas
    for month in holiday_months:
        month_dates = dates[dates.month == month]
        if len(month_dates) > 0:
            holiday_idx = np.random.choice(len(month_dates))
            idx = features_df[features_df['Date'] == month_dates[holiday_idx]].index[0]
            features_df.loc[idx, 'IsHoliday'] = 1
    
    # Save test data
    test_data_dir = tmp_path / 'test_data'
    test_data_dir.mkdir()
    
    features_path = test_data_dir / 'time_series_features.parquet'
    features_df.to_parquet(features_path)
    
    # Test ModelTrainer
    trainer = ModelTrainer(models_dir=tmp_path)
    
    # Mock the prepare_data method
    def mock_prepare_data(features_path, store_id, dept_id):
        df = pd.read_parquet(features_path)
        mask = (df['Store'] == store_id) & (df['Dept'] == dept_id)
        store_dept_data = df[mask].copy().sort_values('Date')
        ts_data = store_dept_data.set_index('Date')['Weekly_Sales']
        return store_dept_data, ts_data
    
    trainer.prepare_data = mock_prepare_data
    
    # Train models
    results = trainer.train_all_models(
        str(features_path),
        store_id=1,
        dept_id=1
    )
    
    # Check training results
    assert 'models' in results
    assert 'predictions' in results
    assert 'metrics' in results
    assert 'metadata' in results
    
    # Test ForecastGenerator
    generator = ForecastGenerator(models_dir=tmp_path)
    
    # Generate forecasts
    forecasts, metadata = generator.generate_all_forecasts(
        store_id=1,
        dept_id=1,
        periods=12
    )
    
    # Check forecasts
    assert isinstance(forecasts, dict)
    assert len(forecasts) > 0
    
    for model_name, forecast_df in forecasts.items():
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == 12
        assert 'forecast' in forecast_df.columns
    
    # Test ModelEvaluator
    evaluator = ModelEvaluator(models_dir=tmp_path)
    
    # Create predictions file for evaluator
    predictions_df = pd.DataFrame({
        'Date': dates[-20:],
        'Actual': sales[-20:],
        'Prophet_Prediction': sales[-20:] + np.random.normal(0, 500, 20),
        'ARIMA_Prediction': sales[-20:] + np.random.normal(0, 600, 20)
    })
    
    predictions_path = tmp_path / 'predictions_store_1_dept_1.csv'
    predictions_df.to_csv(predictions_path, index=False)
    
    # Load predictions
    loaded_predictions = evaluator.load_predictions(1, 1)
    assert len(loaded_predictions) == 20
    
    # Generate performance report
    report = evaluator.generate_performance_report(loaded_predictions, 1, 1)
    
    # Check report
    assert report['store_id'] == 1
    assert report['dept_id'] == 1
    assert 'model_performance' in report
    assert 'summary' in report

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])