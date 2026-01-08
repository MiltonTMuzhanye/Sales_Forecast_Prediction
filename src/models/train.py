import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import logging
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train time series forecasting models"""
    
    def __init__(self, models_dir='../models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, features_path, store_id, dept_id):
        """Prepare data for specific store and department"""
        try:
            df = pd.read_parquet(features_path)
            
            # Filter for specific store and department
            mask = (df['Store'] == store_id) & (df['Dept'] == dept_id)
            store_dept_data = df[mask].copy()
            
            # Sort by date
            store_dept_data = store_dept_data.sort_values('Date')
            
            # Create time series
            ts_data = store_dept_data.set_index('Date')['Weekly_Sales']
            
            return store_dept_data, ts_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train_test_split(self, ts_data, test_size=0.2):
        """Split time series into train and test sets"""
        split_idx = int(len(ts_data) * (1 - test_size))
        
        train = ts_data[:split_idx]
        test = ts_data[split_idx:]
        
        logger.info(f"Train size: {len(train)} ({len(train)/len(ts_data)*100:.1f}%)")
        logger.info(f"Test size: {len(test)} ({len(test)/len(ts_data)*100:.1f}%)")
        
        return train, test
    
    def train_arima(self, train_data, order=(1, 1, 1)):
        """Train ARIMA model"""
        try:
            logger.info(f"Training ARIMA{order}...")
            model = ARIMA(train_data, order=order)
            model_fit = model.fit()
            
            logger.info("ARIMA training completed")
            return model_fit
            
        except Exception as e:
            logger.error(f"Error training ARIMA: {e}")
            raise
    
    def train_sarima(self, train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Train SARIMA model"""
        try:
            logger.info(f"Training SARIMA{order}{seasonal_order}...")
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            
            logger.info("SARIMA training completed")
            return model_fit
            
        except Exception as e:
            logger.error(f"Error training SARIMA: {e}")
            raise
    
    def train_prophet(self, store_dept_data, train_data):
        """Train Prophet model"""
        try:
            logger.info("Training Prophet model...")
            
            # Prepare Prophet data
            prophet_data = store_dept_data[['Date', 'Weekly_Sales']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Filter to training period
            train_dates = train_data.index
            prophet_train = prophet_data[prophet_data['ds'].isin(train_dates)].copy()
            
            # Create holidays dataframe
            holidays = store_dept_data[store_dept_data['IsHoliday'] == 1][['Date']].copy()
            holidays.columns = ['ds']
            holidays['holiday'] = 'store_holiday'
            
            # Train model
            model = Prophet(
                holidays=holidays,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            model.fit(prophet_train)
            
            logger.info("Prophet training completed")
            return model
            
        except Exception as e:
            logger.error(f"Error training Prophet: {e}")
            raise
    
    def evaluate_model(self, model, model_type, train_data, test_data, store_dept_data=None):
        """Evaluate model performance"""
        try:
            if model_type == 'arima':
                predictions = model.forecast(steps=len(test_data))
                
            elif model_type == 'sarima':
                predictions = model.forecast(steps=len(test_data))
                
            elif model_type == 'prophet' and store_dept_data is not None:
                # Prepare future dates
                last_train_date = train_data.index.max()
                future_dates = pd.date_range(
                    start=last_train_date + pd.Timedelta(days=7),
                    periods=len(test_data),
                    freq='W-FRI'
                )
                
                future_df = pd.DataFrame({'ds': future_dates})
                forecast = model.predict(future_df)
                predictions = pd.Series(forecast['yhat'].values, index=future_dates)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Align predictions with test data
            predictions = predictions.reindex(test_data.index)
            
            # Calculate metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(test_data, predictions)),
                'mae': mean_absolute_error(test_data, predictions),
                'mape': mean_absolute_percentage_error(test_data, predictions) * 100,
                'bias': np.mean(predictions - test_data),
                'std_error': np.std(predictions - test_data)
            }
            
            logger.info(f"{model_type.upper()} Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric.upper()}: {value:.2f}")
            
            return predictions, metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_type}: {e}")
            raise
    
    def time_series_cv(self, ts_data, model_type='prophet', n_splits=3, test_size=8):
        """Perform time series cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
            
            cv_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(ts_data), 1):
                train_cv = ts_data.iloc[train_idx]
                test_cv = ts_data.iloc[test_idx]
                
                if model_type == 'prophet':
                    # Prepare Prophet data for this fold
                    prophet_train = pd.DataFrame({
                        'ds': train_cv.index,
                        'y': train_cv.values
                    })
                    
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                    model.fit(prophet_train)
                    
                    future = model.make_future_dataframe(periods=test_size, freq='W')
                    forecast = model.predict(future)
                    predictions = forecast.iloc[-test_size:]['yhat'].values
                    
                elif model_type == 'arima':
                    model = ARIMA(train_cv, order=(1, 1, 1))
                    model_fit = model.fit()
                    predictions = model_fit.forecast(steps=test_size)
                
                # Calculate MAPE for this fold
                mape = mean_absolute_percentage_error(test_cv, predictions) * 100
                cv_scores.append(mape)
                
                logger.info(f"Fold {fold}: MAPE = {mape:.1f}%")
            
            cv_results = {
                'mean_mape': np.mean(cv_scores),
                'std_mape': np.std(cv_scores),
                'scores': cv_scores
            }
            
            logger.info(f"CV Results - Mean MAPE: {cv_results['mean_mape']:.1f}% (±{cv_results['std_mape']:.1f}%)")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in time series CV: {e}")
            raise
    
    def save_model(self, model, model_type, store_id, dept_id, metrics=None):
        """Save trained model"""
        try:
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{model_type}_store_{store_id}_dept_{dept_id}_{timestamp}.pkl"
            filepath = self.models_dir / filename
            
            # Save model
            joblib.dump(model, filepath)
            logger.info(f"Saved model to {filepath}")
            
            # Save latest reference
            latest_file = f"{model_type}_store_{store_id}_dept_{dept_id}_latest.pkl"
            latest_path = self.models_dir / latest_file
            joblib.dump(model, latest_path)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def save_metadata(self, metadata, store_id, dept_id):
        """Save model metadata"""
        try:
            filename = f"metadata_store_{store_id}_dept_{dept_id}.json"
            filepath = self.models_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise
    
    def train_all_models(self, features_path, store_id=1, dept_id=1):
        """Train all models for a specific store and department"""
        try:
            logger.info(f"Training models for Store {store_id}, Department {dept_id}")
            
            # Prepare data
            store_dept_data, ts_data = self.prepare_data(features_path, store_id, dept_id)
            
            # Split data
            train_data, test_data = self.train_test_split(ts_data)
            
            # Train models
            arima_model = self.train_arima(train_data)
            sarima_model = self.train_sarima(train_data)
            prophet_model = self.train_prophet(store_dept_data, train_data)
            
            # Evaluate models
            arima_pred, arima_metrics = self.evaluate_model(
                arima_model, 'arima', train_data, test_data
            )
            
            sarima_pred, sarima_metrics = self.evaluate_model(
                sarima_model, 'sarima', train_data, test_data
            )
            
            prophet_pred, prophet_metrics = self.evaluate_model(
                prophet_model, 'prophet', train_data, test_data, store_dept_data
            )
            
            # Cross-validation for best model
            logger.info("Running cross-validation for Prophet...")
            cv_results = self.time_series_cv(ts_data, model_type='prophet')
            
            # Compare models
            model_comparison = {
                'arima': arima_metrics,
                'sarima': sarima_metrics,
                'prophet': prophet_metrics
            }
            
            # Determine best model
            best_model_type = min(model_comparison.items(), key=lambda x: x[1]['mape'])[0]
            logger.info(f"Best model: {best_model_type.upper()} (MAPE: {model_comparison[best_model_type]['mape']:.1f}%)")
            
            # Save models
            arima_path = self.save_model(arima_model, 'arima', store_id, dept_id)
            sarima_path = self.save_model(sarima_model, 'sarima', store_id, dept_id)
            prophet_path = self.save_model(prophet_model, 'prophet', store_id, dept_id)
            
            # Prepare metadata
            metadata = {
                'store_id': store_id,
                'dept_id': dept_id,
                'training_date': datetime.now().isoformat(),
                'data_info': {
                    'total_weeks': len(ts_data),
                    'train_weeks': len(train_data),
                    'test_weeks': len(test_data),
                    'train_period': {
                        'start': train_data.index.min().isoformat(),
                        'end': train_data.index.max().isoformat()
                    },
                    'test_period': {
                        'start': test_data.index.min().isoformat(),
                        'end': test_data.index.max().isoformat()
                    }
                },
                'model_performance': model_comparison,
                'best_model': best_model_type,
                'cross_validation': cv_results,
                'model_paths': {
                    'arima': str(arima_path),
                    'sarima': str(sarima_path),
                    'prophet': str(prophet_path)
                }
            }
            
            # Save metadata
            self.save_metadata(metadata, store_id, dept_id)
            
            # Save predictions for analysis
            predictions_df = pd.DataFrame({
                'Date': test_data.index,
                'Actual': test_data.values,
                'ARIMA_Prediction': arima_pred.values,
                'SARIMA_Prediction': sarima_pred.values,
                'Prophet_Prediction': prophet_pred.values
            })
            
            predictions_path = self.models_dir / f'predictions_store_{store_id}_dept_{dept_id}.csv'
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Saved predictions to {predictions_path}")
            
            return {
                'models': {
                    'arima': arima_model,
                    'sarima': sarima_model,
                    'prophet': prophet_model
                },
                'predictions': {
                    'arima': arima_pred,
                    'sarima': sarima_pred,
                    'prophet': prophet_pred
                },
                'metrics': model_comparison,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise

def main():
    """Main training pipeline"""
    import sys
    sys.path.append('..')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train models
    trainer = ModelTrainer()
    
    # Use default store and department for demonstration
    store_id = 1
    dept_id = 1
    features_path = '../data/features/time_series_features.parquet'
    
    print(f"Training models for Store {store_id}, Department {dept_id}")
    print("=" * 50)
    
    results = trainer.train_all_models(features_path, store_id, dept_id)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*50)
    
    print(f"\nBest Model: {results['metadata']['best_model'].upper()}")
    print(f"Best MAPE: {results['metrics'][results['metadata']['best_model']]['mape']:.1f}%")
    
    print("\nModel Performance Summary:")
    for model_type, metrics in results['metrics'].items():
        print(f"  {model_type.upper()}:")
        print(f"    MAPE: {metrics['mape']:.1f}%")
        print(f"    MAE: ${metrics['mae']:.0f}")
        print(f"    RMSE: ${metrics['rmse']:.0f}")
    
    print(f"\n✅ Models saved to {trainer.models_dir}")
    print(f"✅ Metadata saved to {trainer.models_dir}/metadata_store_{store_id}_dept_{dept_id}.json")

if __name__ == "__main__":
    main()