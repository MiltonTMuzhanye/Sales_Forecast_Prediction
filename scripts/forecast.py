"""Forecasting script"""

import sys
import argparse
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.sales_forecasting.utils.config import Config
from src.sales_forecasting.utils.logger import setup_logger
from src.sales_forecasting.pipelines.forecasting_pipeline import ForecastingPipeline

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate sales forecasts")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--store", type=int, required=True)
    parser.add_argument("--dept", type=int, required=True)
    parser.add_argument("--periods", type=int, default=12)
    parser.add_argument("--model", type=str, default="prophet")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    try:
        config = Config(args.config)
        pipeline = ForecastingPipeline(config)
        
        result = pipeline.forecast(args.store, args.dept, args.periods, args.model)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Forecast saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, default=str))
        
        return 0
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())