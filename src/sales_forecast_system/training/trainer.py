import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path
import joblib
import mlflow
from ..data.ingestion import DataIngestion
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..models.prophet_model import ProphetModel
from ..models.xgboost_model import XGBoostModel
from ..models.lightgbm_model import LightGBMModel
from ..models.hybrid_model import HybridModel
from ..models.baseline import BaselineModels
from ..evaluation.metrics import ModelEvaluator
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class ModelTrainer:
    """Handles model training pipeline"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.preprocessor = DataPreprocessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.evaluator = ModelEvaluator(config)
        self.models = {}
        self.results = {}
        
    def train_prophet(self, data: pd.DataFrame, store_id: int, dept_id: int) -> Dict:
        """Train Prophet model"""
        logger.info(f"Training Prophet for Store {store_id}, Dept {dept_id}")
        
        # Filter data
        store_dept_data = data[(data['Store'] == store_id) & (data['Dept'] == dept_id)].copy()
        store_dept_data = store_dept_data.sort_values('Date')
        
        # Prepare data
        prophet_data = store_dept_data[['Date', 'Weekly_Sales']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Create holidays
        if 'IsHoliday' in store_dept_data.columns:
            holidays_df = store_dept_data[store_dept_data['IsHoliday'] == 1][['Date']].copy()
            holidays_df.columns = ['ds']
            holidays_df['holiday'] = 'store_holiday'
        else:
            holidays_df = None
        
        # Split data
        train_size = int(len(prophet_data) * 0.8)
        train_data = prophet_data.iloc[:train_size]
        test_data = prophet_data.iloc[train_size:]
        
        # Train model
        model = ProphetModel(self.config)
        model.train(train_data, holidays_df)
        
        # Predict
        predictions = model.predict(len(test_data))
        y_pred = predictions['yhat'].values
        y_true = test_data['y'].values
        
        # Evaluate
        metrics = model.evaluate(y_true, y_pred)
        
        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'train_data': train_data,
            'test_data': test_data
        }
    
    def _recursive_forecast_tree_model(
        self,
        model,
        history: pd.DataFrame,
        future: pd.DataFrame
    ) -> np.ndarray:
        """
        Recursively forecast future observations for a tree-based model.

        Each prediction is appended to the historical target series before
        the next forecast row is constructed. This prevents actual future
        target values from entering lag or rolling features.
        """
        logger.info(
            "Starting recursive forecasting for %s future periods.",
            len(future)
        )

        history = history.copy()
        future = future.copy()

        history[self.feature_engineer.date_col] = pd.to_datetime(
            history[self.feature_engineer.date_col]
        )
        future[self.feature_engineer.date_col] = pd.to_datetime(
            future[self.feature_engineer.date_col]
        )

        history = (
            history
            .sort_values(self.feature_engineer.date_col)
            .reset_index(drop=True)
        )

        future = (
            future
            .sort_values(self.feature_engineer.date_col)
            .reset_index(drop=True)
        )

        predictions = []

        for _, future_row in future.iterrows():
            future_row_df = pd.DataFrame(
                [future_row.to_dict()]
            )

            # The future target is unknown. Remove it before feature
            # construction so it cannot become a model input.
            future_row_df[self.feature_engineer.target_col] = np.nan

            combined = pd.concat(
                [history, future_row_df],
                ignore_index=True,
                sort=False
            )

            engineered = self.feature_engineer.engineer_all_features(
                combined,
                include_target_history=True
            )

            current_features = engineered.iloc[[-1]].copy()

            prediction = float(
                model.predict(current_features)[0]
            )

            predictions.append(prediction)

            # Add the prediction to the historical target series.
            # This prediction becomes available history for the next
            # recursive forecasting step.
            future_row_with_prediction = future_row.copy()
            future_row_with_prediction[
                self.feature_engineer.target_col
            ] = prediction

            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        [future_row_with_prediction.to_dict()]
                    )
                ],
                ignore_index=True,
                sort=False
            )

        logger.info(
            "Recursive forecasting completed: %d predictions.",
            len(predictions)
        )

        return np.asarray(predictions)


    def train_xgboost(
        self,
        data: pd.DataFrame,
        store_id: int,
        dept_id: int
    ) -> Dict:
        """Train and recursively evaluate XGBoost."""

        logger.info(
            f"Training XGBoost for Store {store_id}, Dept {dept_id}"
        )

        store_dept_data = data[
            (data['Store'] == store_id) &
            (data['Dept'] == dept_id)
        ].copy()

        store_dept_data = (
            store_dept_data
            .sort_values('Date')
            .reset_index(drop=True)
        )

        train_size = int(len(store_dept_data) * 0.8)

        train_raw = store_dept_data.iloc[:train_size].copy()
        test_data = store_dept_data.iloc[train_size:].copy()

        logger.info(
            "XGBoost chronological split: "
            f"train={len(train_raw)}, test={len(test_data)}"
        )

        # Create target-history features using training history only.
        train_features = self.feature_engineer.engineer_all_features(
            train_raw,
            include_target_history=True
        )

        # Rows before the maximum required lag cannot contain a complete
        # set of lag features. Remove them from model training only.
        lag_columns = [
            f"lag_{lag}"
            for lag in self.feature_engineer.lags
        ]

        train_features = train_features.dropna(
            subset=lag_columns
        ).reset_index(drop=True)

        if train_features.empty:
            raise ValueError(
                "No usable XGBoost training rows remain after "
                "lag-feature preparation."
            )

        logger.info(
            "XGBoost usable training rows after lag filtering: %d",
            len(train_features)
        )

        model = XGBoostModel(self.config)
        model.train(train_features)

        # Recursive forecasting starts from the complete raw training
        # history, not the lag-filtered training dataframe.
        y_pred = self._recursive_forecast_tree_model(
            model=model,
            history=train_raw,
            future=test_data
        )

        y_true = test_data['Weekly_Sales'].to_numpy()

        metrics = model.evaluate(y_true, y_pred)

        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'train_data': train_raw,
            'test_data': test_data
        }

    def train_lightgbm(
        self,
        data: pd.DataFrame,
        store_id: int,
        dept_id: int
    ) -> Dict:
        """Train and recursively evaluate LightGBM."""

        logger.info(
            f"Training LightGBM for Store {store_id}, Dept {dept_id}"
        )

        store_dept_data = data[
            (data['Store'] == store_id) &
            (data['Dept'] == dept_id)
        ].copy()

        store_dept_data = (
            store_dept_data
            .sort_values('Date')
            .reset_index(drop=True)
        )

        train_size = int(len(store_dept_data) * 0.8)

        train_raw = store_dept_data.iloc[:train_size].copy()
        test_data = store_dept_data.iloc[train_size:].copy()

        logger.info(
            "LightGBM chronological split: "
            f"train={len(train_raw)}, test={len(test_data)}"
        )

        # Create target-history features using training history only.
        train_features = self.feature_engineer.engineer_all_features(
            train_raw,
            include_target_history=True
        )

        lag_columns = [
            f"lag_{lag}"
            for lag in self.feature_engineer.lags
        ]

        train_features = train_features.dropna(
            subset=lag_columns
        ).reset_index(drop=True)

        if train_features.empty:
            raise ValueError(
                "No usable LightGBM training rows remain after "
                "lag-feature preparation."
            )

        logger.info(
            "LightGBM usable training rows after lag filtering: %d",
            len(train_features)
        )

        model = LightGBMModel(self.config)
        model.train(train_features)

        # Forecast recursively so each prediction becomes available
        # history for the next forecasting step.
        y_pred = self._recursive_forecast_tree_model(
            model=model,
            history=train_raw,
            future=test_data
        )

        y_true = test_data['Weekly_Sales'].to_numpy()

        metrics = model.evaluate(y_true, y_pred)

        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'train_data': train_raw,
            'test_data': test_data
        }

    def train_all_models(self, data: pd.DataFrame, store_id: int, dept_id: int) -> Dict:
        """Train all models"""
        logger.info(f"Training all models for Store {store_id}, Dept {dept_id}")
        
        results = {}
        
        # Train Prophet
        results['prophet'] = self.train_prophet(data, store_id, dept_id)
        
        # Train XGBoost
        results['xgboost'] = self.train_xgboost(data, store_id, dept_id)
        
        # Train LightGBM
        results['lightgbm'] = self.train_lightgbm(data, store_id, dept_id)
        
        # Train Hybrid
        try:
            store_dept_data = data[
                (data['Store'] == store_id) &
                (data['Dept'] == dept_id)
            ].copy()

            # Ensure chronological ordering for time-series forecasting.
            store_dept_data = store_dept_data.sort_values('Date').reset_index(drop=True)

            # Use the same chronological 80/20 split as the other models.
            train_size = int(len(store_dept_data) * 0.8)

            train_data = store_dept_data.iloc[:train_size].copy()
            test_data = store_dept_data.iloc[train_size:].copy()

            hybrid = HybridModel(self.config)

            # Train only on historical training data.
            hybrid.train(train_data)

            # Forecast exactly the length of the held-out test period.
            y_pred = hybrid.predict(
                train_data,
                periods=len(test_data)
            )

            y_true = test_data['Weekly_Sales'].values

            results['hybrid'] = {
                'model': hybrid,
                'predictions': y_pred,
                'metrics': hybrid.evaluate(y_true, y_pred),
                'train_data': train_data,
                'test_data': test_data
            }

        except Exception as e:
            logger.warning(f"Hybrid model failed: {e}")
        
        # Compare models using the same chronological test period.
        predictions = {}

        for name, result in results.items():
            if 'predictions' not in result:
                continue

            y_pred = np.asarray(
                result['predictions']
            ).flatten()

            predictions[name] = y_pred

        if 'test_data' in results.get('prophet', {}):
            y_true = np.asarray(
                results['prophet']['test_data']['y'].values
            ).flatten()

            # Only compare models whose predictions have the same
            # number of observations as the common test set.
            valid_predictions = {}

            for model_name, y_pred in predictions.items():
                if len(y_pred) != len(y_true):
                    logger.warning(
                        f"Skipping {model_name} from comparison: "
                        f"prediction length={len(y_pred)}, "
                        f"expected={len(y_true)}"
                    )
                    continue

                valid_predictions[model_name] = y_pred

            if valid_predictions:
                comparison = self.evaluator.compare_models(
                    valid_predictions,
                    y_true
                )

                results['comparison'] = comparison

                logger.info(
                    "Model comparison completed:\n%s",
                    comparison
                )
        
        self.results = results
        return results
    
    def save_models(self, base_path: str = 'artifacts/trained_models/') -> None:
        """Save all trained models"""
        logger.info("Saving models...")
        
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        for model_name, result in self.results.items():
            if 'model' in result and hasattr(result['model'], 'save_model'):
                result['model'].save_model(f"{base_path}/{model_name}_model.pkl")
        
        # Save results
        joblib.dump(self.results, f"{base_path}/training_results.pkl")
        logger.info("All models saved")