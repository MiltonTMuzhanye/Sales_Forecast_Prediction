import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime
from pathlib import Path
import json
from ..utils.logger import setup_logger
from ..utils.config import Config
from .forecasting_pipeline import ForecastingPipeline

logger = setup_logger(__name__)

class BatchForecastingPipeline:
    """Batch forecasting pipeline for scheduled runs"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.pipeline = ForecastingPipeline(config)
        
    def get_stores_departments(self, data: pd.DataFrame) -> Dict[int, List[int]]:
        """Get all store-department combinations"""
        stores_depts = data.groupby('Store')['Dept'].unique().apply(list).to_dict()
        logger.info(f"Found {len(stores_depts)} stores")
        return stores_depts
    
    def run_batch_forecast(self, stores_depts: Optional[Dict[int, List[int]]] = None,
                          periods: int = 12, model_name: str = 'prophet') -> Dict:
        """Run batch forecast for all stores and departments"""
        logger.info("Starting batch forecast...")
        
        # Load data
        from ..data.ingestion import DataIngestion
        from ..data.preprocessing import DataPreprocessor
        
        ingestion = DataIngestion(self.config)
        preprocessor = DataPreprocessor(self.config)
        
        data = ingestion.load_all_data()
        processed_data = preprocessor.preprocess_all(
            data['train'], data['stores'], data['features']
        )
        
        # Get store-department combinations
        if stores_depts is None:
            stores_depts = self.get_stores_departments(processed_data)
        
        # Load models once
        self.pipeline.load_models()
        
        # Generate forecasts
        all_results = {}
        total = sum(len(depts) for depts in stores_depts.values())
        count = 0
        
        for store_id, depts in stores_depts.items():
            for dept_id in depts:
                count += 1
                logger.info(f"Processing {count}/{total}: Store {store_id}, Dept {dept_id}")
                
                try:
                    result = self.pipeline.forecast(store_id, dept_id, periods, model_name)
                    key = f"{store_id}_{dept_id}"
                    all_results[key] = result
                except Exception as e:
                    logger.error(f"Failed: Store {store_id}, Dept {dept_id}: {e}")
                    all_results[f"{store_id}_{dept_id}"] = {'error': str(e)}
        
        # Save summary
        self.save_batch_summary(all_results)
        
        logger.info(f"Batch forecast completed. {len(all_results)} forecasts generated.")
        return all_results
    
    def save_batch_summary(self, results: Dict) -> None:
        """Save batch forecast summary"""
        output_path = Path(self.config.get('forecast.output_path', 'reports/forecasts/'))
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary
        summary_data = []
        for key, result in results.items():
            if 'error' in result:
                continue
            
            store_id, dept_id = key.split('_')
            summary_data.append({
                'store_id': int(store_id),
                'department_id': int(dept_id),
                'mean_prediction': np.mean(result.get('predictions', [0])),
                'timestamp': result.get('timestamp', '')
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(output_path / f'batch_summary_{timestamp}.csv', index=False)
        
        with open(output_path / f'batch_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Batch summary saved to {output_path}")