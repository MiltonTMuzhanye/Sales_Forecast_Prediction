"""Training script"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.sales_forecasting.utils.config import Config
from src.sales_forecasting.utils.logger import setup_logger
from src.sales_forecasting.pipelines.training_pipeline import TrainingPipeline

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train sales forecasting models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--dept", type=int, default=1)
    
    args = parser.parse_args()
    
    try:
        config = Config(args.config)
        pipeline = TrainingPipeline(config)
        results = pipeline.run(store_id=args.store, dept_id=args.dept)
        
        print("\nTraining Results:")
        if 'comparison' in results:
            print(results['comparison'])
        
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())