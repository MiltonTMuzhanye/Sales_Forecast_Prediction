import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from pathlib import Path
import joblib
import mlflow
from ..data.ingestion import DataIngestion
from ..data.validation import DataValidator
from ..data.preprocessing import DataPreprocessor
from ..models.arima_model import ARIMAModel
from ..models.sarima_model import SARIMAModel
from ..models.prophet_model import ProphetModel
from ..models.ensemble_model import EnsembleModel
from ..evaluation.metrics import ModelEvaluator
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.ingestion = DataIngestion(config)
        self.validator = DataValidator(config)
        self.preprocessor = DataPreprocessor(config)
        self.evaluator = ModelEvaluator(config)
        self.models = {}
        self.results = {}
        
    def run(self, store_id: int = 1, dept_id: int = 1) -> Dict:
        """Run the complete training pipeline for a specific store and department"""
        logger.info(f"Starting training pipeline for Store {store_id}, Dept {dept_id}...")
        
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        data = self.ingestion.load_all_data()
        
        # Step 2: Validate data
        logger.info("Step 2: Validating data...")
        self.validator.validate_all(data)
        
        # Step 3: Preprocess data
        logger.info("Step 3: Preprocessing data...")
        processed_data = self.preprocessor.preprocess_all(
            data['train'], data['stores'], data['features']
        )
        
        # Step 4: Filter data for specific store and department
        logger.info(f"Step 4: Filtering data for Store {store_id}, Dept {dept_id}...")
        store_dept_data = processed_data[
            (processed_data['Store'] == store_id) & 
            (processed_data['Dept'] == dept_id)
        ].sort_values('Date')
        
        if len(store_dept_data) == 0:
            logger.error(f"No data found for Store {store_id}, Dept {dept_id}")
            return {}
        
        # Step 5: Prepare time series data
        logger.info("Step 5: Preparing time series data...")
        store_dept_data.set_index('Date', inplace=True)
        ts_data = store_dept_data['Weekly_Sales']
        
        # Split data (80/20 as in notebook)
        train_size = int(len(ts_data) * 0.8)
        train_data_ts = ts_data[:train_size]
        test_data_ts = ts_data[train_size:]
        
        logger.info(f"Train size: {len(train_data_ts)}, Test size: {len(test_data_ts)}")
        
        # Create holidays dataframe
        holidays_df = store_dept_data[store_dept_data['IsHoliday'] == 1].reset_index()[['Date']]
        holidays_df.columns = ['ds']
        holidays_df['holiday'] = 'store_holiday'
        
        # Step 6: Train models
        logger.info("Step 6: Training models...")
        
        # Train ARIMA
        arima = ARIMAModel(self.config)
        arima.train(train_data_ts)
        arima_pred = arima.predict(len(test_data_ts))
        arima_metrics = arima.evaluate(test_data_ts, arima_pred)
        
        # Train SARIMA
        sarima = SARIMAModel(self.config)
        sarima.train(train_data_ts)
        sarima_pred = sarima.predict(len(test_data_ts))
        sarima_metrics = sarima.evaluate(test_data_ts, sarima_pred)
        
        # Train Prophet
        prophet = ProphetModel(self.config)
        prophet_df = store_dept_data.reset_index()[['Date', 'Weekly_Sales']]
        prophet_df.columns = ['ds', 'y']
        prophet.train(prophet_df, holidays_df)
        prophet_pred_df = prophet.predict(len(test_data_ts))
        prophet_pred = prophet_pred_df['yhat'].values
        prophet_metrics = prophet.evaluate(test_data_ts, prophet_pred)
        
        # Train Ensemble
        ensemble = EnsembleModel(self.config)
        ensemble.train(train_data_ts, holidays_df)
        ensemble_pred = ensemble.predict(len(test_data_ts))
        ensemble_metrics = ensemble.evaluate(test_data_ts, ensemble_pred)
        
        # Step 7: Compare models
        logger.info("Step 7: Comparing models...")
        predictions = {
            'ARIMA': arima_pred,
            'SARIMA': sarima_pred,
            'Prophet': prophet_pred,
            'Ensemble': ensemble_pred
        }
        comparison_df = self.evaluator.compare_models(predictions, test_data_ts)
        
        # Store results
        self.results = {
            'store_id': store_id,
            'dept_id': dept_id,
            'models': {
                'ARIMA': {'model': arima, 'predictions': arima_pred, 'metrics': arima_metrics},
                'SARIMA': {'model': sarima, 'predictions': sarima_pred, 'metrics': sarima_metrics},
                'Prophet': {'model': prophet, 'predictions': prophet_pred, 'metrics': prophet_metrics},
                'Ensemble': {'model': ensemble, 'predictions': ensemble_pred, 'metrics': ensemble_metrics}
            },
            'comparison': comparison_df,
            'train_data': train_data_ts,
            'test_data': test_data_ts
        }
        
        logger.info("Training pipeline completed successfully")
        return self.results
    
    def save_models(self, base_path: str = 'artifacts/trained_models/'):
        """Save all trained models"""
        logger.info("Saving models...")
        
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        for model_name, model_data in self.results['models'].items():
            model = model_data['model']
            if hasattr(model, 'save_model'):
                model.save_model(f"{base_path}/{model_name.lower()}_model.pkl")
        
        # Save results
        joblib.dump(self.results, f"{base_path}/training_results.pkl")
        logger.info("All models saved")