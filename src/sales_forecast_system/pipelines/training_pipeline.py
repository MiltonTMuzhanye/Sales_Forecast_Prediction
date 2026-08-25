import pandas as pd
from typing import Dict, Optional, List
import logging
from pathlib import Path
import joblib
import mlflow
from ..data.ingestion import DataIngestion
from ..data.validation import DataValidator
from ..data.preprocessing import DataPreprocessor
from ..training.trainer import ModelTrainer
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
        self.trainer = ModelTrainer(config)
        
    def run(self, store_id: int = 1, dept_id: int = 1) -> Dict:
        """Run the complete training pipeline"""
        logger.info(f"Starting training pipeline for Store {store_id}, Dept {dept_id}")
        
        with mlflow.start_run(run_name=f"training_store_{store_id}_dept_{dept_id}"):
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
            
            # Step 4: Train models
            logger.info("Step 4: Training models...")
            results = self.trainer.train_all_models(processed_data, store_id, dept_id)
            
            # Step 5: Save models
            logger.info("Step 5: Saving models...")
            self.trainer.save_models()
            
            # Step 6: Log metrics
            logger.info("Step 6: Logging metrics...")
            if 'comparison' in results:
                for model_name, metrics in results['comparison'].iterrows():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            logger.info("Training pipeline completed successfully")
            return results