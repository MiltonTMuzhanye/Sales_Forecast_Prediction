"""Evaluation script"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.sales_forecasting.utils.config import Config
from src.sales_forecasting.utils.logger import setup_logger
from src.sales_forecasting.evaluation.metrics import ModelEvaluator

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--results", type=str, default="artifacts/trained_models/training_results.pkl")
    
    args = parser.parse_args()
    
    try:
        import joblib
        results = joblib.load(args.results)
        
        print("\nModel Performance Comparison:")
        if 'comparison' in results:
            print(results['comparison'])
        
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())