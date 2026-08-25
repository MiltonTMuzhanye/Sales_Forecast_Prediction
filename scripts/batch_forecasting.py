"""Batch forecasting script"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.sales_forecasting.utils.config import Config
from src.sales_forecasting.utils.logger import setup_logger
from src.sales_forecasting.pipelines.batch_pipeline import BatchForecastingPipeline

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run batch forecasting")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--periods", type=int, default=12)
    parser.add_argument("--model", type=str, default="prophet")
    
    args = parser.parse_args()
    
    try:
        config = Config(args.config)
        pipeline = BatchForecastingPipeline(config)
        
        results = pipeline.run_batch_forecast(
            periods=args.periods,
            model_name=args.model
        )
        
        print(f"\nBatch forecast completed. {len(results)} forecasts generated.")
        return 0
    except Exception as e:
        logger.error(f"Batch forecast failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())