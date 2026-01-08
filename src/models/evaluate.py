import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate forecasting model performance"""
    
    def __init__(self, models_dir='../models'):
        self.models_dir = Path(models_dir)
    
    def load_predictions(self, store_id, dept_id):
        """Load model predictions"""
        try:
            predictions_path = self.models_dir / f'predictions_store_{store_id}_dept_{dept_id}.csv'
            predictions_df = pd.read_csv(predictions_path)
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
            
            logger.info(f"Loaded predictions: {predictions_df.shape}")
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            raise
    
    def calculate_error_metrics(self, actual, predicted, model_name):
        """Calculate comprehensive error metrics"""
        errors = predicted - actual
        
        metrics = {
            'model': model_name,
            'mape': mean_absolute_percentage_error(actual, predicted) * 100,
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mse': mean_squared_error(actual, predicted),
            'bias': np.mean(errors),
            'std_error': np.std(errors),
            'mad': np.median(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
            'mean_absolute_scaled_error': np.mean(np.abs(errors)) / np.mean(np.abs(np.diff(actual))),
            'symmetric_mape': 100 * np.mean(2 * np.abs(errors) / (np.abs(actual) + np.abs(predicted)))
        }
        
        # Directional accuracy
        direction_actual = np.sign(np.diff(actual))
        direction_pred = np.sign(np.diff(predicted))
        direction_match = direction_actual == direction_pred
        metrics['direction_accuracy'] = np.mean(direction_match) * 100 if len(direction_match) > 0 else 0
        
        return metrics
    
    def analyze_error_distribution(self, errors):
        """Analyze the distribution of errors"""
        analysis = {
            'mean': np.mean(errors),
            'median': np.median(errors),
            'std': np.std(errors),
            'skewness': pd.Series(errors).skew(),
            'kurtosis': pd.Series(errors).kurtosis(),
            'q1': np.percentile(errors, 25),
            'q3': np.percentile(errors, 75),
            'iqr': np.percentile(errors, 75) - np.percentile(errors, 25),
            'range': np.max(errors) - np.min(errors)
        }
        
        # Normality test (simplified)
        from scipy import stats
        analysis['is_normal'] = stats.shapiro(errors[:5000])[1] > 0.05 if len(errors) > 3 else False
        
        return analysis
    
    def analyze_error_by_horizon(self, predictions_df):
        """Analyze how error changes with forecast horizon"""
        results = []
        
        for horizon in range(1, min(13, len(predictions_df))):
            horizon_data = predictions_df.iloc[:horizon]
            
            metrics = {}
            for model in ['ARIMA', 'SARIMA', 'Prophet']:
                if f'{model}_Prediction' in horizon_data.columns:
                    actual = horizon_data['Actual']
                    predicted = horizon_data[f'{model}_Prediction']
                    
                    metrics[model.lower()] = {
                        'mape': mean_absolute_percentage_error(actual, predicted) * 100,
                        'bias': np.mean(predicted - actual),
                        'std_error': np.std(predicted - actual)
                    }
            
            results.append({
                'horizon': horizon,
                'metrics': metrics
            })
        
        return pd.DataFrame(results)
    
    def analyze_error_by_condition(self, predictions_df, features_df=None):
        """Analyze error under different conditions"""
        analysis = {}
        
        # Holiday vs Non-holiday
        if 'IsHoliday' in predictions_df.columns:
            holiday_mask = predictions_df['IsHoliday'] == 1
            
            analysis['holiday'] = {}
            for model in ['ARIMA', 'SARIMA', 'Prophet']:
                pred_col = f'{model}_Prediction'
                if pred_col in predictions_df.columns:
                    holiday_errors = predictions_df.loc[holiday_mask, pred_col] - predictions_df.loc[holiday_mask, 'Actual']
                    non_holiday_errors = predictions_df.loc[~holiday_mask, pred_col] - predictions_df.loc[~holiday_mask, 'Actual']
                    
                    analysis['holiday'][model.lower()] = {
                        'holiday_mape': mean_absolute_percentage_error(
                            predictions_df.loc[holiday_mask, 'Actual'],
                            predictions_df.loc[holiday_mask, pred_col]
                        ) * 100,
                        'non_holiday_mape': mean_absolute_percentage_error(
                            predictions_df.loc[~holiday_mask, 'Actual'],
                            predictions_df.loc[~holiday_mask, pred_col]
                        ) * 100,
                        'difference': None  # Will be calculated
                    }
        
        # Sales volume categories
        if 'Actual' in predictions_df.columns:
            sales_quantiles = predictions_df['Actual'].quantile([0.25, 0.5, 0.75])
            
            analysis['sales_volume'] = {}
            for i, (q_name, q_value) in enumerate(zip(['low', 'medium', 'high', 'very_high'], 
                                                    [0, sales_quantiles[0.25], sales_quantiles[0.5], sales_quantiles[0.75]])):
                if i == 0:
                    mask = predictions_df['Actual'] <= sales_quantiles[0.25]
                elif i == 3:
                    mask = predictions_df['Actual'] > sales_quantiles[0.75]
                else:
                    lower = list(sales_quantiles.values)[i-1]
                    upper = list(sales_quantiles.values)[i]
                    mask = (predictions_df['Actual'] > lower) & (predictions_df['Actual'] <= upper)
                
                analysis['sales_volume'][q_name] = {}
                for model in ['ARIMA', 'SARIMA', 'Prophet']:
                    pred_col = f'{model}_Prediction'
                    if pred_col in predictions_df.columns and mask.any():
                        actual_subset = predictions_df.loc[mask, 'Actual']
                        pred_subset = predictions_df.loc[mask, pred_col]
                        
                        if len(actual_subset) > 0:
                            analysis['sales_volume'][q_name][model.lower()] = {
                                'mape': mean_absolute_percentage_error(actual_subset, pred_subset) * 100,
                                'count': len(actual_subset)
                            }
        
        return analysis
    
    def detect_forecast_bias(self, predictions_df, model_name):
        """Detect and analyze forecast bias"""
        if f'{model_name}_Prediction' not in predictions_df.columns:
            return None
        
        errors = predictions_df[f'{model_name}_Prediction'] - predictions_df['Actual']
        
        from scipy import stats
        
        bias_analysis = {
            'mean_bias': np.mean(errors),
            'median_bias': np.median(errors),
            'percent_positive': (errors > 0).mean() * 100,
            'percent_negative': (errors < 0).mean() * 100,
            'percent_zero': (errors == 0).mean() * 100,
            't_test_pvalue': stats.ttest_1samp(errors, 0).pvalue,
            'is_significant_bias': None,
            'bias_trend': None
        }
        
        # Check for significant bias
        bias_analysis['is_significant_bias'] = bias_analysis['t_test_pvalue'] < 0.05
        
        # Check for trend in bias over time
        if len(errors) > 2:
            bias_trend = np.polyfit(range(len(errors)), errors, 1)[0]
            bias_analysis['bias_trend'] = bias_trend
        
        return bias_analysis
    
    def generate_performance_report(self, predictions_df, store_id, dept_id):
        """Generate comprehensive performance report"""
        report = {
            'store_id': store_id,
            'dept_id': dept_id,
            'evaluation_date': datetime.now().isoformat(),
            'summary': {},
            'model_performance': {},
            'error_analysis': {},
            'recommendations': []
        }
        
        # Calculate metrics for each model
        for model in ['ARIMA', 'SARIMA', 'Prophet']:
            pred_col = f'{model}_Prediction'
            if pred_col in predictions_df.columns:
                metrics = self.calculate_error_metrics(
                    predictions_df['Actual'],
                    predictions_df[pred_col],
                    model
                )
                report['model_performance'][model.lower()] = metrics
        
        # Determine best model
        if report['model_performance']:
            best_model = min(
                report['model_performance'].items(),
                key=lambda x: x[1]['mape']
            )
            report['summary']['best_model'] = best_model[0]
            report['summary']['best_mape'] = best_model[1]['mape']
        
        # Analyze error distribution for best model
        if 'summary' in report and 'best_model' in report['summary']:
            best_model = report['summary']['best_model']
            errors = predictions_df[f'{best_model.capitalize()}_Prediction'] - predictions_df['Actual']
            report['error_analysis']['distribution'] = self.analyze_error_distribution(errors)
            
            # Bias analysis
            report['error_analysis']['bias'] = self.detect_forecast_bias(predictions_df, best_model.capitalize())
        
        # Analyze error by horizon
        report['error_analysis']['by_horizon'] = self.analyze_error_by_horizon(predictions_df).to_dict('records')
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if 'model_performance' in report:
            # Check if any model has MAPE > 30%
            for model_name, metrics in report['model_performance'].items():
                if metrics['mape'] > 30:
                    recommendations.append(
                        f"{model_name.upper()} model shows high error (MAPE: {metrics['mape']:.1f}%). Consider retraining or trying different parameters."
                    )
            
            # Check for bias
            if 'error_analysis' in report and 'bias' in report['error_analysis']:
                bias = report['error_analysis']['bias']
                if bias and bias['is_significant_bias']:
                    direction = "over" if bias['mean_bias'] > 0 else "under"
                    recommendations.append(
                        f"Model shows significant {direction}-forecasting bias. Consider bias correction techniques."
                    )
        
        # Check error increase with horizon
        if 'error_analysis' in report and 'by_horizon' in report['error_analysis']:
            horizon_data = report['error_analysis']['by_horizon']
            if len(horizon_data) >= 2:
                first_horizon = horizon_data[0]['metrics'].get('prophet', {}).get('mape', 0)
                last_horizon = horizon_data[-1]['metrics'].get('prophet', {}).get('mape', 0)
                
                if last_horizon > first_horizon * 1.5:
                    recommendations.append(
                        f"Forecast accuracy degrades significantly with horizon. Consider shorter forecast horizons or horizon-specific models."
                    )
        
        report['recommendations'] = recommendations
    
    def create_visualizations(self, predictions_df, store_id, dept_id, output_dir='../monitoring'):
        """Create evaluation visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Forecast vs Actual
        axes[0, 0].plot(predictions_df['Date'], predictions_df['Actual'], 
                        label='Actual', linewidth=2, color='#2E86AB')
        
        colors = {'ARIMA': '#F18F01', 'SARIMA': '#C73E1D', 'Prophet': '#A23B72'}
        for model in ['ARIMA', 'SARIMA', 'Prophet']:
            pred_col = f'{model}_Prediction'
            if pred_col in predictions_df.columns:
                axes[0, 0].plot(predictions_df['Date'], predictions_df[pred_col],
                               label=f'{model} Forecast', linewidth=1.5, 
                               color=colors.get(model, 'gray'), linestyle='--')
        
        axes[0, 0].set_title(f'Forecast vs Actual - Store {store_id}, Dept {dept_id}', fontsize=12)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Weekly Sales ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error Distribution
        if 'Prophet_Prediction' in predictions_df.columns:
            errors = predictions_df['Prophet_Prediction'] - predictions_df['Actual']
            axes[0, 1].hist(errors, bins=30, color='#2E86AB', edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[0, 1].set_title('Error Distribution (Prophet)', fontsize=12)
            axes[0, 1].set_xlabel('Forecast Error ($)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: MAPE Comparison
        mape_values = []
        model_names = []
        for model in ['ARIMA', 'SARIMA', 'Prophet']:
            pred_col = f'{model}_Prediction'
            if pred_col in predictions_df.columns:
                mape = mean_absolute_percentage_error(
                    predictions_df['Actual'], 
                    predictions_df[pred_col]
                ) * 100
                mape_values.append(mape)
                model_names.append(model)
        
        axes[1, 0].bar(model_names, mape_values, 
                      color=[colors.get(m, 'gray') for m in model_names])
        axes[1, 0].set_title('Model Performance Comparison (MAPE)', fontsize=12)
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(mape_values):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Cumulative Error
        if 'Prophet_Prediction' in predictions_df.columns:
            cumulative_error = (predictions_df['Prophet_Prediction'] - predictions_df['Actual']).cumsum()
            axes[1, 1].plot(predictions_df['Date'], cumulative_error, 
                           linewidth=2, color='#A23B72')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title('Cumulative Forecast Error', fontsize=12)
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Cumulative Error ($)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        viz_path = output_dir / f'evaluation_visualizations_store_{store_id}_dept_{dept_id}.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualizations to {viz_path}")
    
    def save_report(self, report, store_id, dept_id, output_dir='../monitoring'):
        """Save evaluation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f'evaluation_report_store_{store_id}_dept_{dept_id}.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved evaluation report to {report_path}")
        return report_path

def main():
    """Main evaluation pipeline"""
    import sys
    sys.path.append('..')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Evaluate models for Store 1, Dept 1
    store_id = 1
    dept_id = 1
    
    evaluator = ModelEvaluator()
    
    print(f"Evaluating models for Store {store_id}, Department {dept_id}")
    print("=" * 50)
    
    # Load predictions
    predictions_df = evaluator.load_predictions(store_id, dept_id)
    
    # Generate performance report
    report = evaluator.generate_performance_report(predictions_df, store_id, dept_id)
    
    # Create visualizations
    evaluator.create_visualizations(predictions_df, store_id, dept_id)
    
    # Save report
    evaluator.save_report(report, store_id, dept_id)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETED")
    print("="*50)
    
    print(f"\nBest Model: {report['summary']['best_model'].upper()}")
    print(f"Best MAPE: {report['summary']['best_mape']:.1f}%")
    
    print("\nModel Performance:")
    for model_name, metrics in report['model_performance'].items():
        print(f"  {model_name.upper()}:")
        print(f"    MAPE: {metrics['mape']:.1f}%")
        print(f"    MAE: ${metrics['mae']:.0f}")
        print(f"    Bias: ${metrics['bias']:.0f}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n✅ Report saved to monitoring/evaluation_report_store_{store_id}_dept_{dept_id}.json")
    print(f"✅ Visualizations saved to monitoring/evaluation_visualizations_store_{store_id}_dept_{dept_id}.png")

if __name__ == "__main__":
    main()
