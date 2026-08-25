import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.sales_forecasting.utils.config import Config
from src.sales_forecasting.utils.logger import setup_logger
from src.sales_forecasting.utils.exceptions import DataIngestionError
from src.sales_forecasting.data.ingestion import DataIngestion
from src.sales_forecasting.data.validation import DataValidator

logger = setup_logger(__name__)

class DataIngestor:
    """
    Handles data ingestion with validation, backup, and reporting
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.ingestion = DataIngestion(config)
        self.validator = DataValidator(config)
        self.raw_path = Path(self.config.get('data.raw_path', 'data/raw/'))
        self.processed_path = Path(self.config.get('data.processed_path', 'data/processed/'))
        self.external_path = Path(self.config.get('data.external_path', 'data/external/'))
        
        # Create directories if they don't exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.external_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.data = {}
        self.ingestion_report = {}
        
    def ingest_all(self) -> Dict[str, pd.DataFrame]:
        """
        Ingest all data files
        """
        logger.info("Starting data ingestion...")
        
        try:
            # Load all data
            self.data = self.ingestion.load_all_data()
            
            # Create ingestion report
            self.ingestion_report = {
                'timestamp': datetime.now().isoformat(),
                'files_loaded': {
                    'train': len(self.data['train']),
                    'stores': len(self.data['stores']),
                    'features': len(self.data['features'])
                },
                'columns': {
                    'train': self.data['train'].columns.tolist(),
                    'stores': self.data['stores'].columns.tolist(),
                    'features': self.data['features'].columns.tolist()
                },
                'date_ranges': {}
            }
            
            # Get date ranges
            if 'Date' in self.data['train'].columns:
                self.data['train']['Date'] = pd.to_datetime(self.data['train']['Date'])
                self.ingestion_report['date_ranges']['train'] = {
                    'min': self.data['train']['Date'].min().isoformat(),
                    'max': self.data['train']['Date'].max().isoformat()
                }
            
            if 'Date' in self.data['features'].columns:
                self.data['features']['Date'] = pd.to_datetime(self.data['features']['Date'])
                self.ingestion_report['date_ranges']['features'] = {
                    'min': self.data['features']['Date'].min().isoformat(),
                    'max': self.data['features']['Date'].max().isoformat()
                }
            
            logger.info(f"Data ingestion completed. Loaded {sum(self.ingestion_report['files_loaded'].values())} rows")
            return self.data
            
        except Exception as e:
            raise DataIngestionError(f"Failed to ingest data: {e}")
    
    def validate_ingested_data(self) -> bool:
        """
        Validate the ingested data
        """
        logger.info("Validating ingested data...")
        
        if not self.data:
            raise ValueError("No data loaded. Run ingest_all() first.")
        
        try:
            # Run validations
            is_valid = self.validator.validate_all(self.data)
            
            # Add validation results to report
            self.ingestion_report['validation'] = {
                'status': 'passed' if is_valid else 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Additional validations
            validation_details = {}
            
            # Check for duplicates
            for name, df in self.data.items():
                if 'Store' in df.columns and 'Date' in df.columns:
                    duplicates = df.duplicated(subset=['Store', 'Date']).sum()
                    validation_details[f'{name}_duplicates'] = duplicates
                    
                    if duplicates > 0:
                        logger.warning(f"Found {duplicates} duplicate rows in {name} data")
            
            # Check for missing values
            for name, df in self.data.items():
                missing = df.isnull().sum().to_dict()
                validation_details[f'{name}_missing'] = missing
            
            self.ingestion_report['validation_details'] = validation_details
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.ingestion_report['validation'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def save_processed_data(self) -> None:
        """
        Save processed data to disk
        """
        logger.info("Saving processed data...")
        
        try:
            for name, df in self.data.items():
                # Save as CSV
                file_path = self.processed_path / f"{name}_processed.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {name} data to {file_path}")
                
                # Save as Parquet for better performance
                parquet_path = self.processed_path / f"{name}_processed.parquet"
                df.to_parquet(parquet_path, index=False)
                logger.info(f"Saved {name} data to {parquet_path}")
            
            # Save ingestion report
            report_path = self.processed_path / "ingestion_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.ingestion_report, f, indent=2, default=str)
            logger.info(f"Saved ingestion report to {report_path}")
            
        except Exception as e:
            raise DataIngestionError(f"Failed to save processed data: {e}")
    
    def backup_raw_data(self) -> None:
        """
        Create backup of raw data
        """
        logger.info("Creating backup of raw data...")
        
        backup_path = self.raw_path / "backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            for file_path in self.raw_path.glob("*.csv"):
                shutil.copy2(file_path, backup_path / file_path.name)
                logger.info(f"Backed up {file_path.name} to {backup_path}")
            
            self.ingestion_report['backup'] = {
                'path': str(backup_path),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Failed to backup raw data: {e}")
    
    def generate_summary_report(self) -> Dict:
        """
        Generate a summary report of the ingested data
        """
        logger.info("Generating summary report...")
        
        summary = {
            'ingestion_timestamp': datetime.now().isoformat(),
            'files': {},
            'statistics': {},
            'warnings': []
        }
        
        for name, df in self.data.items():
            summary['files'][name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
            
            # Generate statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary['statistics'][name] = {}
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    summary['statistics'][name][col] = {
                        'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                        'std': float(df[col].std()) if not df[col].isnull().all() else None,
                        'min': float(df[col].min()) if not df[col].isnull().all() else None,
                        'max': float(df[col].max()) if not df[col].isnull().all() else None,
                        'null_count': int(df[col].isnull().sum())
                    }
            
            # Check for issues
            if df.isnull().sum().sum() > 0:
                summary['warnings'].append(f"{name} contains missing values")
            
            if len(df) == 0:
                summary['warnings'].append(f"{name} is empty")
        
        # Save summary report
        report_path = self.processed_path / "summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to {report_path}")
        return summary
    
    def run(self, validate_only: bool = False, create_backup: bool = True) -> Dict:
        """
        Run the complete data ingestion pipeline
        
        Args:
            validate_only: Only validate data without processing
            create_backup: Create backup of raw data
        
        Returns:
            Dictionary with ingestion results
        """
        logger.info("="*60)
        logger.info("STARTING DATA INGESTION PIPELINE")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Ingest data
            logger.info("Step 1: Ingesting data...")
            self.ingest_all()
            
            # Step 2: Validate data
            logger.info("Step 2: Validating data...")
            is_valid = self.validate_ingested_data()
            
            if not is_valid:
                logger.warning("Data validation failed. Check the report for details.")
            
            if validate_only:
                logger.info("Validation-only mode. Skipping processing.")
                return self.ingestion_report
            
            # Step 3: Backup raw data
            if create_backup:
                logger.info("Step 3: Creating backup...")
                self.backup_raw_data()
            
            # Step 4: Save processed data
            logger.info("Step 4: Saving processed data...")
            self.save_processed_data()
            
            # Step 5: Generate summary report
            logger.info("Step 5: Generating summary report...")
            summary = self.generate_summary_report()
            
            # Add to report
            self.ingestion_report['summary'] = summary
            self.ingestion_report['duration_seconds'] = (datetime.now() - start_time).total_seconds()
            self.ingestion_report['status'] = 'completed'
            
            # Save final report
            report_path = self.processed_path / "ingestion_complete_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.ingestion_report, f, indent=2, default=str)
            
            logger.info("="*60)
            logger.info("DATA INGESTION COMPLETED SUCCESSFULLY")
            logger.info(f"Duration: {self.ingestion_report['duration_seconds']:.2f} seconds")
            logger.info("="*60)
            
            return self.ingestion_report
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            self.ingestion_report['status'] = 'failed'
            self.ingestion_report['error'] = str(e)
            
            # Save error report
            report_path = self.processed_path / "ingestion_error_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.ingestion_report, f, indent=2, default=str)
            
            raise

def main():
    """
    Main function to run data ingestion from command line
    """
    parser = argparse.ArgumentParser(
        description="Ingest and validate data for sales forecasting system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic ingestion
    python scripts/ingest_data.py
    
    # Validate only
    python scripts/ingest_data.py --validate-only
    
    # Use custom config
    python scripts/ingest_data.py --config configs/my_config.yaml
    
    # Skip backup
    python scripts/ingest_data.py --no-backup
    
    # Custom directories
    python scripts/ingest_data.py --data-dir data/custom/ --output-dir data/processed/
        """
    )
    
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file (default: configs/config.yaml)")
    
    parser.add_argument("--data-dir", type=str, 
                       help="Directory containing raw data files")
    
    parser.add_argument("--output-dir", type=str,
                       help="Directory to save processed data")
    
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate data without processing")
    
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backup of raw data")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = Config(args.config)
        
        # Override directories if specified
        if args.data_dir:
            config.update('data.raw_path', args.data_dir)
        
        if args.output_dir:
            config.update('data.processed_path', args.output_dir)
        
        # Set logging level
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize ingestor
        ingestor = DataIngestor(config)
        
        # Check if data files exist
        missing_files = []
        required_files = ['train.csv', 'stores.csv', 'features.csv']
        
        for file in required_files:
            file_path = ingestor.raw_path / file
            if not file_path.exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"Missing required files: {missing_files}")
            logger.info(f"Please place the following files in {ingestor.raw_path}:")
            for file in missing_files:
                logger.info(f"  - {file}")
            
            if not args.validate_only:
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return 0
        
        # Run ingestion
        results = ingestor.run(
            validate_only=args.validate_only,
            create_backup=not args.no_backup
        )
        
        # Print summary
        print("\n" + "="*60)
        print("DATA INGESTION SUMMARY")
        print("="*60)
        
        print(f"\nStatus: {results.get('status', 'unknown')}")
        print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
        
        if 'files_loaded' in results:
            print("\nFiles Loaded:")
            for name, count in results['files_loaded'].items():
                print(f"  - {name}: {count} rows")
        
        if 'validation' in results:
            print(f"\nValidation: {results['validation'].get('status', 'unknown')}")
        
        if 'warnings' in results.get('summary', {}):
            print("\nWarnings:")
            for warning in results['summary']['warnings']:
                print(f"  ⚠️ {warning}")
        
        print("\n" + "="*60)
        print(f"Processed data saved to: {ingestor.processed_path}")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Data ingestion interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())