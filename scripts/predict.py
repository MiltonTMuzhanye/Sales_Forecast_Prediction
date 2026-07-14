"""
Prediction script for sales forecast
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.sales_forecast.utils.config import Config
from src.sales_forecast.utils.logger import setup_logger
from src.sales_forecast.pipelines.inference_pipeline import InferencePipeline
from src.sales_forecast.data.ingestion import DataIngestion

logger = setup_logger(__name__)

def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description="Make sales forecasts")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--store", type=int, required=True,
                       help="Store ID")
    parser.add_argument("--dept", type=int, required=True,
                       help="Department ID")
    parser.add_argument("--periods", type=int, default=12,
                       help="Number of periods to forecast")
    parser.add_argument("--model", type=str, default="Prophet",
                       choices=["ARIMA", "SARIMA", "Prophet", "Ensemble"],
                       help="Model to use")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = Config(args.config)
        
        # Load data
        ingestion = DataIngestion(config)
        data = ingestion.load_all_data()
        
        # Preprocess data
        from src.sales_forecast.data.preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor(config)
        processed_data = preprocessor.preprocess_all(
            data['train'], data['stores'], data['features']
        )
        
        # Filter for specific store and department
        store_dept_data = processed_data[
            (processed_data['Store'] == args.store) & 
            (processed_data['Dept'] == args.dept)
        ]
        
        if len(store_dept_data) == 0:
            logger.error(f"No data found for Store {args.store}, Dept {args.dept}")
            return 1
        
        # Initialize inference pipeline
        pipeline = InferencePipeline(config)
        
        # Make predictions
        predictions = pipeline.predict(
            store_dept_data, 
            periods=args.periods, 
            model_name=args.model
        )
        
        # Prepare output
        output_data = {
            'store_id': args.store,
            'department_id': args.dept,
            'model': args.model,
            'periods': args.periods,
            'predictions': predictions.to_dict(orient='records')
        }
        
        # Save or print
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            logger.info(f"Predictions saved to {args.output}")
        else:
            print(json.dumps(output_data, indent=2, default=str))
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())