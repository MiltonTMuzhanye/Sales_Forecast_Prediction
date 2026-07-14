"""
Evaluation script for sales forecast models
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.sales_forecast.utils.config import Config
from src.sales_forecast.utils.logger import setup_logger
from src.sales_forecast.evaluation.metrics import ModelEvaluator

logger = setup_logger(__name__)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate sales forecast models")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--results", type=str, default="artifacts/trained_models/training_results.pkl",
                       help="Path to training results")
    parser.add_argument("--output", type=str, default="reports/metrics/",
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = Config(args.config)
        
        # Load results
        import joblib
        results = joblib.load(args.results)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get comparison DataFrame
        comparison_df = results['comparison']
        
        # Save comparison
        comparison_df.to_csv(output_path / 'model_comparison.csv')
        
        print("\n" + "="*50)
        print("Model Performance Comparison")
        print("="*50)
        print(comparison_df)
        
        # Find best model
        best_model = comparison_df['rmse'].idxmin()
        print(f"\nBest Model: {best_model}")
        print(f"RMSE: {comparison_df.loc[best_model, 'rmse']:.2f}")
        print(f"MAE: {comparison_df.loc[best_model, 'mae']:.2f}")
        print(f"MAPE: {comparison_df.loc[best_model, 'mape']:.2f}%")
        
        # Generate plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE comparison
        comparison_df['rmse'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        comparison_df['mae'].plot(kind='bar', ax=axes[0, 1], color='lightcoral')
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        comparison_df['mape'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_title('MAPE Comparison (%)')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall performance
        comparison_df[['rmse', 'mae', 'mape']].plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Overall Performance Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())