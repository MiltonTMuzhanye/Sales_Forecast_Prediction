import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path
import joblib
from ..data.ingestion import DataIngestion
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..models.prophet_model import ProphetModel
from ..models.xgboost_model import XGBoostModel
from ..models.lightgbm_model import LightGBMModel
from ..models.hybrid_model import HybridModel
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class ForecastingPipeline:
    """End-to-end forecasting pipeline"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.preprocessor = DataPreprocessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.models = {}
        self.loaded = False
        
    def load_models(self, model_path: str = 'artifacts/trained_models/') -> None:
        """Load all trained models"""
        logger.info("Loading models...")
        
        try:
            prophet = ProphetModel(self.config)
            prophet.load_model(f"{model_path}/prophet_model.pkl")
            self.models['prophet'] = prophet

            xgb = XGBoostModel(self.config)
            xgb.load_model(f"{model_path}/xgboost_model.pkl")
            self.models['xgboost'] = xgb

            lgb = LightGBMModel(self.config)
            lgb.load_model(f"{model_path}/lightgbm_model.pkl")
            self.models['lightgbm'] = lgb

            try:
                hybrid = HybridModel(self.config)
                hybrid.load_model(f"{model_path}/hybrid_model/")
                self.models['hybrid'] = hybrid
            except:
                logger.warning("Hybrid model not found")
            
            self.loaded = True
            logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def forecast(
        self,
        store_id: int,
        dept_id: int,
        periods: int = 12,
        model_name: str = 'prophet'
    ) -> Dict:
        """Generate future forecasts for a specific store and department."""

        logger.info(
            f"Generating {periods}-period forecast for "
            f"Store {store_id}, Dept {dept_id}, model={model_name}"
        )

        if not self.loaded:
            self.load_models()

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        ingestion = DataIngestion(self.config)
        data = ingestion.load_all_data()

        processed_data = self.preprocessor.preprocess_all(
            data['train'],
            data['stores'],
            data['features']
        )

        store_dept_data = processed_data[
            (processed_data['Store'] == store_id) &
            (processed_data['Dept'] == dept_id)
        ].copy()

        if store_dept_data.empty:
            raise ValueError(
                f"No data found for Store {store_id}, Dept {dept_id}"
            )

        store_dept_data = store_dept_data.sort_values('Date').reset_index(
            drop=True
        )

        model = self.models[model_name]

        # ---------------------------------------------------------
        # PROPHET
        # ---------------------------------------------------------
        if model_name == 'prophet':
            prophet_data = store_dept_data[
                ['Date', 'Weekly_Sales']
            ].copy()

            prophet_data.columns = ['ds', 'y']

            holidays_df = None

            if 'IsHoliday' in store_dept_data.columns:
                holiday_rows = store_dept_data[
                    store_dept_data['IsHoliday'].astype(bool)
                ][['Date']].copy()

                if not holiday_rows.empty:
                    holiday_rows.columns = ['ds']
                    holiday_rows['holiday'] = 'store_holiday'
                    holidays_df = holiday_rows

            # The saved Prophet model is trained for this series.
            # Retrain on the complete historical series before forecasting.
            model.train(prophet_data, holidays_df)

            predictions_df = model.predict(periods)

            dates = predictions_df['ds'].tolist()
            predictions = predictions_df['yhat'].tolist()

        # ---------------------------------------------------------
        # XGBOOST / LIGHTGBM
        # ---------------------------------------------------------
        elif model_name in ['xgboost', 'lightgbm']:

            history = store_dept_data.copy()

            last_date = history['Date'].max()

            future = pd.DataFrame({
                'Store': [store_id] * periods,
                'Dept': [dept_id] * periods,
                'Date': pd.date_range(
                    start=last_date + pd.Timedelta(days=7),
                    periods=periods,
                    freq='7D'
                )
            })

            # Use the most recent known exogenous values as the
            # future baseline. This is a deterministic fallback
            # until an external future-feature source is provided.
            static_cols = [
                col for col in history.columns
                if col not in ['Weekly_Sales', 'Date']
            ]

            for col in static_cols:
                if col not in future.columns:
                    future[col] = history[col].iloc[-1]

            history = history.sort_values('Date').reset_index(drop=True)
            future = future.sort_values('Date').reset_index(drop=True)

            predictions = []

            for _, future_row in future.iterrows():

                row = future_row.copy()
                row['Weekly_Sales'] = np.nan

                combined = pd.concat(
                    [history, pd.DataFrame([row])],
                    ignore_index=True,
                    sort=False
                )

                engineered = self.feature_engineer.engineer_all_features(
                    combined,
                    include_target_history=True
                )

                current = engineered.iloc[[-1]].copy()

                prediction = float(model.predict(current)[0])

                predictions.append(prediction)

                row['Weekly_Sales'] = prediction

                history = pd.concat(
                    [history, pd.DataFrame([row])],
                    ignore_index=True,
                    sort=False
                )

            dates = future['Date'].tolist()

        # ---------------------------------------------------------
        # HYBRID
        # ---------------------------------------------------------
        elif model_name == 'hybrid':

            predictions = model.predict(
                store_dept_data,
                periods
            )

            last_date = store_dept_data['Date'].max()

            dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=periods,
                freq='7D'
            ).tolist()

            predictions = np.asarray(predictions).tolist()

        else:
            raise ValueError(
                f"Unsupported forecasting model: {model_name}"
            )

        result = {
            'store_id': store_id,
            'department_id': dept_id,
            'model': model_name,
            'periods': periods,
            'dates': [
                pd.Timestamp(d).isoformat()
                for d in dates
            ],
            'predictions': [
                float(p)
                for p in predictions
            ],
            'timestamp': pd.Timestamp.now().isoformat()
        }

        logger.info(
            f"Forecast completed: {len(predictions)} predictions generated."
        )

        return result

