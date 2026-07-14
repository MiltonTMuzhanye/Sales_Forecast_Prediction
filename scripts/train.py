"""
Training script for sales forecast models
"""

import sys
import argparse
from pathlib import Path
import logging
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.sales_forecast.utils.config import Config
from src.sales_forecast.utils.logger import setup_logger
from src.sales_forecast.pipelines.training_pipeline import TrainingPipeline

logger = setup_logger(__name__)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train sales forecast models")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--store", type=int, default=1,
                       help="Store ID")
    parser.add_argument("--dept", type=int, default=1,
                       help="Department ID")
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = Config(args.config)
        
        # Initialize pipeline
        pipeline = TrainingPipeline(config)
        
        # Run pipeline
        results = pipeline.run(store_id=args.store, dept_id=args.dept)
        
        # Save models
        pipeline.save_models()
        
        # Print results
        print("\n" + "="*50)
        print("Training Results")
        print("="*50)
        print(f"Store: {results['store_id']}, Dept: {results['dept_id']}")
        print("\nModel Performance:")
        print(results['comparison'])
        
        # Find best model
        best_model = results['comparison']['rmse'].idxmin()
        print(f"\nBest Model: {best_model}")
        print(f"RMSE: {results['comparison'].loc[best_model, 'rmse']:.2f}")
        print(f"MAE: {results['comparison'].loc[best_model, 'mae']:.2f}")
        print(f"MAPE: {results['comparison'].loc[best_model, 'mape']:.2f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
