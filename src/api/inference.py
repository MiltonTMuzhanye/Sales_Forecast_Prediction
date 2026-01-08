import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

class ForecastInference:
    """Inference engine for generating forecasts"""
    
    def __init__(self, models_dir: str = "../models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, store_id: int, dept_id: int, model_type: str = "prophet"):
        """Load trained model for specific store and department"""
        try:
            model_filename = f"{model_type}_store_{store_id}_dept_{dept_id}_latest.pkl"
            model_path = self.models_dir / model_filename
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            model = joblib.load(model_path)
            logger.info(f"Loaded {model_type} model for Store {store_id}, Dept {dept_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_metadata(self, store_id: int, dept_id: int):
        """Load model metadata"""
        try:
            metadata_filename = f"metadata_store_{store_id}_dept_{dept_id}.json"
            metadata_path = self.models_dir / metadata_filename
            
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error loading metadata: {e}")
            return None
    
    def generate_prophet_forecast(self, model: Prophet, periods: int = 12) -> pd.DataFrame:
        """Generate forecast using Prophet model"""
        try:
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq='W')
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract future predictions only
            last_training_date = model.history['ds'].max()
            future_forecast = forecast[forecast['ds'] > last_training_date].copy()
            
            # Prepare results
            results = pd.DataFrame({
                'date': future_forecast['ds'],
                'forecast': future_forecast['yhat'],
                'lower_bound': future_forecast['yhat_lower'],
                'upper_bound': future_forecast['yhat_upper']
            })
            
            results.set_index('date', inplace=True)
            return results
            
        except Exception as e:
            logger.error(f"Error generating Prophet forecast: {e}")
            raise
    
    def generate_arima_forecast(self, model: ARIMA, periods: int = 12) -> pd.DataFrame:
        """Generate forecast using ARIMA model"""
        try:
            # Generate forecast
            forecast = model.forecast(steps=periods)
            
            # Create dates (assuming weekly frequency)
            last_date = model.data.endog.index[-1] if hasattr(model.data.endog, 'index') else None
            
            if last_date and hasattr(last_date, 'strftime'):
                dates = pd.date_range(
                    start=last_date + timedelta(days=7),
                    periods=periods,
                    freq='W-FRI'
                )
            else:
                dates = pd.date_range(
                    start=datetime.now(),
                    periods=periods,
                    freq='W-FRI'
                )
            
            # Calculate confidence intervals
            if hasattr(model, 'get_forecast'):
                forecast_result = model.get_forecast(steps=periods)
                conf_int = forecast_result.conf_int()
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            else:
                # Fallback: use residual standard deviation
                resid_std = np.std(model.resid) if hasattr(model, 'resid') else forecast.std()
                lower_bound = forecast - 1.96 * resid_std
                upper_bound = forecast + 1.96 * resid_std
            
            # Prepare results
            results = pd.DataFrame({
                'date': dates,
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
            results.set_index('date', inplace=True)
            return results
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {e}")
            raise
    
    def generate_sarima_forecast(self, model: SARIMAX, periods: int = 12) -> pd.DataFrame:
        """Generate forecast using SARIMA model"""
        try:
            # Generate forecast
            forecast = model.forecast(steps=periods)
            
            # Create dates (assuming weekly frequency)
            last_date = model.data.endog.index[-1] if hasattr(model.data.endog, 'index') else None
            
            if last_date and hasattr(last_date, 'strftime'):
                dates = pd.date_range(
                    start=last_date + timedelta(days=7),
                    periods=periods,
                    freq='W-FRI'
                )
            else:
                dates = pd.date_range(
                    start=datetime.now(),
                    periods=periods,
                    freq='W-FRI'
                )
            
            # Get confidence intervals
            if hasattr(model, 'get_forecast'):
                forecast_result = model.get_forecast(steps=periods)
                conf_int = forecast_result.conf_int()
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            else:
                # Fallback
                resid_std = np.std(model.resid) if hasattr(model, 'resid') else forecast.std()
                lower_bound = forecast - 1.96 * resid_std
                upper_bound = forecast + 1.96 * resid_std
            
            # Prepare results
            results = pd.DataFrame({
                'date': dates,
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
            results.set_index('date', inplace=True)
            return results
            
        except Exception as e:
            logger.error(f"Error generating SARIMA forecast: {e}")
            raise
    
    def generate_ensemble_forecast(self, forecasts: Dict[str, pd.DataFrame], 
                                 weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Generate ensemble forecast from multiple models"""
        try:
            if not forecasts:
                raise ValueError("No forecasts provided for ensemble")
            
            # Use equal weights if not specified
            if weights is None:
                weights = {model: 1/len(forecasts) for model in forecasts.keys()}
            
            # Check all forecasts have same dates
            first_dates = list(forecasts.values())[0].index
            for model_name, forecast_df in forecasts.items():
                if not forecast_df.index.equals(first_dates):
                    raise ValueError(f"Forecast dates don't match for {model_name}")
            
            # Calculate weighted ensemble
            forecast_values = np.zeros(len(first_dates))
            lower_values = np.zeros(len(first_dates))
            upper_values = np.zeros(len(first_dates))
            
            for model_name, forecast_df in forecasts.items():
                weight = weights.get(model_name, 0)
                forecast_values += forecast_df['forecast'].values * weight
                
                if 'lower_bound' in forecast_df.columns:
                    lower_values += forecast_df['lower_bound'].values * weight
                if 'upper_bound' in forecast_df.columns:
                    upper_values += forecast_df['upper_bound'].values * weight
            
            # Prepare results
            results = pd.DataFrame({
                'date': first_dates,
                'forecast': forecast_values,
                'lower_bound': lower_values,
                'upper_bound': upper_values,
                'model': 'ensemble'
            })
            
            results.set_index('date', inplace=True)
            return results
            
        except Exception as e:
            logger.error(f"Error generating ensemble forecast: {e}")
            raise
    
    def calculate_forecast_metrics(self, forecast_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for generated forecast"""
        try:
            metrics = {
                'forecast_periods': len(forecast_df),
                'forecast_start': forecast_df.index.min().strftime('%Y-%m-%d'),
                'forecast_end': forecast_df.index.max().strftime('%Y-%m-%d'),
                'total_forecast': float(forecast_df['forecast'].sum()),
                'mean_forecast': float(forecast_df['forecast'].mean()),
                'std_forecast': float(forecast_df['forecast'].std()),
                'min_forecast': float(forecast_df['forecast'].min()),
                'max_forecast': float(forecast_df['forecast'].max())
            }
            
            # Add uncertainty metrics if available
            if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                metrics['mean_uncertainty_range'] = float(
                    (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
                )
                metrics['max_uncertainty_range'] = float(
                    (forecast_df['upper_bound'] - forecast_df['lower_bound']).max()
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {e}")
            raise
    
    def generate_forecast(self, store_id: int, dept_id: int, periods: int = 12,
                         model_type: str = "prophet", include_confidence: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Generate forecast for specific store and department"""
        try:
            logger.info(f"Generating {periods}-period forecast for Store {store_id}, Dept {dept_id} using {model_type}")
            
            # Load model
            model = self.load_model(store_id, dept_id, model_type)
            
            # Generate forecast based on model type
            if model_type == "prophet":
                forecast_df = self.generate_prophet_forecast(model, periods)
            elif model_type == "arima":
                forecast_df = self.generate_arima_forecast(model, periods)
            elif model_type == "sarima":
                forecast_df = self.generate_sarima_forecast(model, periods)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Remove confidence intervals if not requested
            if not include_confidence:
                forecast_df = forecast_df[['forecast']].copy()
            
            # Calculate metrics
            metrics = self.calculate_forecast_metrics(forecast_df)
            
            # Add metadata
            metadata = self.load_metadata(store_id, dept_id)
            if metadata:
                metrics['model_performance'] = metadata.get('model_performance', {}).get(model_type, {})
                metrics['training_date'] = metadata.get('training_date', 'unknown')
            
            logger.info(f"Forecast generated successfully: {len(forecast_df)} periods")
            return forecast_df, metrics
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
    
    def generate_multiple_forecasts(self, store_id: int, dept_id: int, periods: int = 12,
                                  model_types: List[str] = None) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """Generate forecasts using multiple models"""
        try:
            if model_types is None:
                model_types = ["prophet", "arima", "sarima"]
            
            results = {}
            
            for model_type in model_types:
                try:
                    forecast_df, metrics = self.generate_forecast(
                        store_id, dept_id, periods, model_type
                    )
                    results[model_type] = (forecast_df, metrics)
                    
                except Exception as e:
                    logger.warning(f"Could not generate {model_type} forecast: {e}")
                    results[model_type] = (None, {"error": str(e)})
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating multiple forecasts: {e}")
            raise
    
    def save_forecast(self, forecast_df: pd.DataFrame, store_id: int, dept_id: int,
                     model_type: str, output_dir: str = "../models/forecasts"):
        """Save forecast to file"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"forecast_{model_type}_store_{store_id}_dept_{dept_id}_{timestamp}.csv"
            filepath = output_dir / filename
            
            # Save forecast
            forecast_df.to_csv(filepath)
            
            # Save latest reference
            latest_filename = f"forecast_{model_type}_store_{store_id}_dept_{dept_id}_latest.csv"
            latest_path = output_dir / latest_filename
            forecast_df.to_csv(latest_path)
            
            logger.info(f"Saved forecast to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving forecast: {e}")
            raise
    
    def validate_forecast(self, forecast_df: pd.DataFrame) -> List[str]:
        """Validate forecast for data quality issues"""
        issues = []
        
        # Check for NaN values
        if forecast_df.isnull().any().any():
            nan_cols = forecast_df.columns[forecast_df.isnull().any()].tolist()
            issues.append(f"NaN values found in columns: {nan_cols}")
        
        # Check for negative forecasts (if sales should always be positive)
        if 'forecast' in forecast_df.columns and (forecast_df['forecast'] < 0).any():
            negative_count = (forecast_df['forecast'] < 0).sum()
            issues.append(f"Negative forecasts found: {negative_count} periods")
        
        # Check confidence intervals
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            # Check if lower bound > upper bound
            invalid_bounds = (forecast_df['lower_bound'] > forecast_df['upper_bound']).any()
            if invalid_bounds:
                issues.append("Invalid confidence intervals: lower bound > upper bound")
            
            # Check if forecast is outside bounds
            outside_bounds = ((forecast_df['forecast'] < forecast_df['lower_bound']) | 
                            (forecast_df['forecast'] > forecast_df['upper_bound'])).any()
            if outside_bounds:
                issues.append("Forecast values outside confidence intervals")
        
        # Check for extreme values
        if 'forecast' in forecast_df.columns:
            forecast_mean = forecast_df['forecast'].mean()
            forecast_std = forecast_df['forecast'].std()
            extreme_threshold = forecast_mean + 3 * forecast_std
            
            extreme_values = (forecast_df['forecast'] > extreme_threshold).any()
            if extreme_values:
                issues.append("Extreme forecast values detected")
        
        return issues

def main():
    """Test the inference engine"""
    import sys
    sys.path.append('..')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test inference
    inference = ForecastInference()
    
    store_id = 1
    dept_id = 1
    periods = 12
    
    print(f"Testing inference for Store {store_id}, Department {dept_id}")
    print("=" * 50)
    
    try:
        # Generate forecast
        forecast_df, metrics = inference.generate_forecast(
            store_id, dept_id, periods, "prophet"
        )
        
        print(f"Generated forecast: {len(forecast_df)} periods")
        print(f"Total forecasted sales: ${metrics['total_forecast']:,.0f}")
        print(f"Average weekly forecast: ${metrics['mean_forecast']:,.0f}")
        
        # Validate forecast
        issues = inference.validate_forecast(forecast_df)
        if issues:
            print("\nValidation Issues:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("\n✅ Forecast validation passed")
        
        # Save forecast
        saved_path = inference.save_forecast(forecast_df, store_id, dept_id, "prophet")
        print(f"\n✅ Forecast saved to {saved_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()