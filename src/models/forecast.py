import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ForecastGenerator:
    """Generate forecasts using trained models"""
    
    def __init__(self, models_dir='../models'):
        self.models_dir = Path(models_dir)
    
    def load_model(self, model_type, store_id, dept_id, version='latest'):
        """Load trained model"""
        try:
            if version == 'latest':
                filename = f"{model_type}_store_{store_id}_dept_{dept_id}_latest.pkl"
            else:
                filename = f"{model_type}_store_{store_id}_dept_{dept_id}_{version}.pkl"
            
            model_path = self.models_dir / filename
            model = joblib.load(model_path)
            
            logger.info(f"Loaded {model_type} model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_metadata(self, store_id, dept_id):
        """Load model metadata"""
        try:
            metadata_path = self.models_dir / f"metadata_store_{store_id}_dept_{dept_id}.json"
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded metadata from {metadata_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def generate_prophet_forecast(self, model, periods=12, freq='W'):
        """Generate forecast using Prophet model"""
        try:
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq=freq)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract future predictions only
            last_training_date = model.history['ds'].max()
            future_forecast = forecast[forecast['ds'] > last_training_date].copy()
            
            # Format results
            results = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                columns={
                    'ds': 'date',
                    'yhat': 'forecast',
                    'yhat_lower': 'lower_bound',
                    'yhat_upper': 'upper_bound'
                }
            )
            
            results['model'] = 'prophet'
            results['created_at'] = datetime.now()
            
            logger.info(f"Generated Prophet forecast for {periods} periods")
            return results
            
        except Exception as e:
            logger.error(f"Error generating Prophet forecast: {e}")
            raise
    
    def generate_arima_forecast(self, model, periods=12):
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
            
            # Format results (ARIMA doesn't provide confidence intervals by default)
            results = pd.DataFrame({
                'date': dates,
                'forecast': forecast,
                'lower_bound': forecast - 1.96 * np.std(model.resid),
                'upper_bound': forecast + 1.96 * np.std(model.resid),
                'model': 'arima',
                'created_at': datetime.now()
            })
            
            logger.info(f"Generated ARIMA forecast for {periods} periods")
            return results
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {e}")
            raise
    
    def generate_sarima_forecast(self, model, periods=12):
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
            
            # Get confidence intervals if available
            if hasattr(model, 'get_forecast'):
                forecast_result = model.get_forecast(steps=periods)
                conf_int = forecast_result.conf_int()
                
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            else:
                # Use residual standard deviation
                resid_std = np.std(model.resid) if hasattr(model, 'resid') else forecast.std()
                lower_bound = forecast - 1.96 * resid_std
                upper_bound = forecast + 1.96 * resid_std
            
            # Format results
            results = pd.DataFrame({
                'date': dates,
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model': 'sarima',
                'created_at': datetime.now()
            })
            
            logger.info(f"Generated SARIMA forecast for {periods} periods")
            return results
            
        except Exception as e:
            logger.error(f"Error generating SARIMA forecast: {e}")
            raise
    
    def generate_ensemble_forecast(self, forecasts_list, weights=None):
        """Generate ensemble forecast from multiple models"""
        try:
            if not forecasts_list:
                raise ValueError("No forecasts provided for ensemble")
            
            # Use equal weights if not specified
            if weights is None:
                weights = [1/len(forecasts_list)] * len(forecasts_list)
            
            # Check all forecasts have same dates
            dates = forecasts_list[0]['date']
            for forecast in forecasts_list[1:]:
                if not forecast['date'].equals(dates):
                    raise ValueError("Forecast dates don't match")
            
            # Calculate weighted ensemble
            forecast_values = np.zeros(len(dates))
            lower_values = np.zeros(len(dates))
            upper_values = np.zeros(len(dates))
            
            for forecast, weight in zip(forecasts_list, weights):
                forecast_values += forecast['forecast'].values * weight
                lower_values += forecast['lower_bound'].values * weight
                upper_values += forecast['upper_bound'].values * weight
            
            # Create ensemble results
            results = pd.DataFrame({
                'date': dates,
                'forecast': forecast_values,
                'lower_bound': lower_values,
                'upper_bound': upper_values,
                'model': 'ensemble',
                'created_at': datetime.now(),
                'component_models': [f['model'].iloc[0] for f in forecasts_list],
                'weights': weights
            })
            
            logger.info(f"Generated ensemble forecast from {len(forecasts_list)} models")
            return results
            
        except Exception as e:
            logger.error(f"Error generating ensemble forecast: {e}")
            raise
    
    def generate_all_forecasts(self, store_id, dept_id, periods=12, 
                              include_ensemble=True, ensemble_weights=None):
        """Generate forecasts using all available models"""
        try:
            logger.info(f"Generating forecasts for Store {store_id}, Dept {dept_id}")
            
            forecasts = {}
            
            # Try to load and generate forecast for each model type
            model_types = ['prophet', 'arima', 'sarima']
            
            for model_type in model_types:
                try:
                    model = self.load_model(model_type, store_id, dept_id)
                    
                    if model_type == 'prophet':
                        forecast_df = self.generate_prophet_forecast(model, periods)
                    elif model_type == 'arima':
                        forecast_df = self.generate_arima_forecast(model, periods)
                    elif model_type == 'sarima':
                        forecast_df = self.generate_sarima_forecast(model, periods)
                    
                    forecasts[model_type] = forecast_df
                    logger.info(f"Generated {model_type} forecast successfully")
                    
                except Exception as e:
                    logger.warning(f"Could not generate {model_type} forecast: {e}")
            
            # Generate ensemble forecast if requested and we have at least 2 models
            if include_ensemble and len(forecasts) >= 2:
                try:
                    forecast_list = list(forecasts.values())
                    ensemble_forecast = self.generate_ensemble_forecast(
                        forecast_list, 
                        ensemble_weights
                    )
                    forecasts['ensemble'] = ensemble_forecast
                except Exception as e:
                    logger.warning(f"Could not generate ensemble forecast: {e}")
            
            # Load metadata for context
            try:
                metadata = self.load_metadata(store_id, dept_id)
            except:
                metadata = None
            
            return forecasts, metadata
            
        except Exception as e:
            logger.error(f"Error generating all forecasts: {e}")
            raise
    
    def calculate_forecast_metrics(self, forecasts, historical_data=None):
        """Calculate metrics for generated forecasts"""
        try:
            metrics = {}
            
            for model_name, forecast_df in forecasts.items():
                model_metrics = {
                    'forecast_periods': len(forecast_df),
                    'forecast_start': forecast_df['date'].min().strftime('%Y-%m-%d'),
                    'forecast_end': forecast_df['date'].max().strftime('%Y-%m-%d'),
                    'mean_forecast': forecast_df['forecast'].mean(),
                    'total_forecast': forecast_df['forecast'].sum(),
                    'forecast_std': forecast_df['forecast'].std(),
                    'uncertainty_range': (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
                }
                
                # If historical data is provided, calculate growth rates
                if historical_data is not None:
                    last_historical = historical_data['Weekly_Sales'].iloc[-4:].mean()  # Last month average
                    first_forecast = forecast_df['forecast'].iloc[:4].mean()  # First month forecast
                    
                    if last_historical > 0:
                        growth_rate = (first_forecast / last_historical - 1) * 100
                        model_metrics['projected_growth_rate'] = growth_rate
                
                metrics[model_name] = model_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {e}")
            raise
    
    def save_forecasts(self, forecasts, store_id, dept_id, output_dir='../models/forecasts'):
        """Save forecasts to files"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_paths = {}
            
            for model_name, forecast_df in forecasts.items():
                # Create filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"forecast_{model_name}_store_{store_id}_dept_{dept_id}_{timestamp}.csv"
                filepath = output_dir / filename
                
                # Save forecast
                forecast_df.to_csv(filepath, index=False)
                saved_paths[model_name] = str(filepath)
                
                # Save latest reference
                latest_filename = f"forecast_{model_name}_store_{store_id}_dept_{dept_id}_latest.csv"
                latest_path = output_dir / latest_filename
                forecast_df.to_csv(latest_path, index=False)
            
            # Save combined forecasts
            combined_df = pd.concat([
                forecast_df.assign(model=model_name) 
                for model_name, forecast_df in forecasts.items()
            ])
            
            combined_filename = f"all_forecasts_store_{store_id}_dept_{dept_id}_{timestamp}.csv"
            combined_path = output_dir / combined_filename
            combined_df.to_csv(combined_path, index=False)
            
            logger.info(f"Saved forecasts to {output_dir}")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error saving forecasts: {e}")
            raise
    
    def generate_forecast_report(self, forecasts, metrics, store_id, dept_id, metadata=None):
        """Generate forecast report"""
        report = {
            'store_id': store_id,
            'dept_id': dept_id,
            'generation_date': datetime.now().isoformat(),
            'forecast_horizon': len(next(iter(forecasts.values()))) if forecasts else 0,
            'available_models': list(forecasts.keys()),
            'forecast_metrics': metrics,
            'metadata': metadata,
            'summary': {},
            'recommendations': []
        }
        
        # Generate summary
        if forecasts:
            # Find model with least uncertainty (narrowest confidence intervals)
            uncertainty_scores = {}
            for model_name, forecast_df in forecasts.items():
                uncertainty = (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
                uncertainty_scores[model_name] = uncertainty
            
            least_uncertain = min(uncertainty_scores.items(), key=lambda x: x[1])[0]
            
            report['summary'] = {
                'recommended_model': least_uncertain,
                'forecast_period': {
                    'start': forecasts[least_uncertain]['date'].min().strftime('%Y-%m-%d'),
                    'end': forecasts[least_uncertain]['date'].max().strftime('%Y-%m-%d')
                },
                'total_forecasted_sales': metrics.get(least_uncertain, {}).get('total_forecast', 0),
                'average_weekly_forecast': metrics.get(least_uncertain, {}).get('mean_forecast', 0),
                'uncertainty_level': uncertainty_scores[least_uncertain]
            }
        
        # Generate recommendations
        self._generate_forecast_recommendations(report)
        
        return report
    
    def _generate_forecast_recommendations(self, report):
        """Generate recommendations based on forecast"""
        recommendations = []
        
        if 'forecast_metrics' in report:
            # Check forecast volatility
            for model_name, metrics in report['forecast_metrics'].items():
                cv = metrics.get('forecast_std', 0) / metrics.get('mean_forecast', 1)
                if cv > 0.3:
                    recommendations.append(
                        f"{model_name.upper()} forecast shows high volatility (CV: {cv:.2f}). "
                        "Consider more conservative inventory planning."
                    )
            
            # Check for significant growth/decline
            if 'projected_growth_rate' in report['forecast_metrics'].get('prophet', {}):
                growth = report['forecast_metrics']['prophet']['projected_growth_rate']
                if growth > 20:
                    recommendations.append(
                        f"Forecast projects significant growth ({growth:.1f}%). "
                        "Plan for increased inventory and staffing."
                    )
                elif growth < -10:
                    recommendations.append(
                        f"Forecast projects significant decline ({growth:.1f}%). "
                        "Consider reducing inventory orders."
                    )
        
        report['recommendations'] = recommendations
    
    def save_forecast_report(self, report, store_id, dept_id, output_dir='../monitoring'):
        """Save forecast report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f'forecast_report_store_{store_id}_dept_{dept_id}.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved forecast report to {report_path}")
        return report_path

def main():
    """Main forecast generation pipeline"""
    import sys
    sys.path.append('..')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate forecasts for Store 1, Dept 1
    store_id = 1
    dept_id = 1
    forecast_horizon = 12  # 12 weeks
    
    generator = ForecastGenerator()
    
    print(f"Generating forecasts for Store {store_id}, Department {dept_id}")
    print(f"Forecast horizon: {forecast_horizon} weeks")
    print("=" * 50)
    
    # Generate all forecasts
    forecasts, metadata = generator.generate_all_forecasts(
        store_id, dept_id, 
        periods=forecast_horizon,
        include_ensemble=True
    )
    
    # Calculate metrics
    # Note: In production, you would load historical data for comparison
    metrics = generator.calculate_forecast_metrics(forecasts)
    
    # Generate report
    report = generator.generate_forecast_report(
        forecasts, metrics, store_id, dept_id, metadata
    )
    
    # Save forecasts
    saved_paths = generator.save_forecasts(forecasts, store_id, dept_id)
    
    # Save report
    report_path = generator.save_forecast_report(report, store_id, dept_id)
    
    print("\n" + "="*50)
    print("FORECAST GENERATION COMPLETED")
    print("="*50)
    
    print(f"\nGenerated forecasts for {len(forecasts)} models:")
    for model_name, forecast_df in forecasts.items():
        print(f"  • {model_name.upper()}: {len(forecast_df)} periods")
    
    print(f"\nRecommended model: {report['summary']['recommended_model'].upper()}")
    print(f"Forecast period: {report['summary']['forecast_period']['start']} to {report['summary']['forecast_period']['end']}")
    print(f"Total forecasted sales: ${report['summary']['total_forecasted_sales']:,.0f}")
    print(f"Average weekly forecast: ${report['summary']['average_weekly_forecast']:,.0f}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n✅ Forecasts saved to models/forecasts/")
    print(f"✅ Report saved to {report_path}")

if __name__ == "__main__":
    main()