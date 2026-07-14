import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path
import json

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class ErrorAnalyzer:
    """Analyze forecast errors and identify root causes"""
    
    def __init__(self, data_dir='../data'):
        self.data_dir = Path(data_dir)
    
    def load_forecast_data(self, store_id, dept_id):
        """Load forecast and actual data"""
        try:
            # Load predictions
            predictions_path = Path('../models') / f'predictions_store_{store_id}_dept_{dept_id}.csv'
            predictions_df = pd.read_csv(predictions_path)
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
            
            # Load features for context
            features_path = self.data_dir / 'features' / 'time_series_features.parquet'
            features_df = pd.read_parquet(features_path)
            
            # Filter for specific store and department
            mask = (features_df['Store'] == store_id) & (features_df['Dept'] == dept_id)
            store_features = features_df[mask].copy()
            store_features['Date'] = pd.to_datetime(store_features['Date'])
            
            # Merge predictions with features
            analysis_df = pd.merge(
                predictions_df,
                store_features,
                on='Date',
                how='left',
                suffixes=('', '_feature')
            )
            
            logger.info(f"Loaded forecast data: {analysis_df.shape}")
            return analysis_df
            
        except Exception as e:
            logger.error(f"Error loading forecast data: {e}")
            raise
    
    def calculate_error_components(self, analysis_df, model_name='Prophet'):
        """Calculate different components of forecast error"""
        if f'{model_name}_Prediction' not in analysis_df.columns:
            raise ValueError(f"Model {model_name} not found in data")
        
        actual = analysis_df['Actual']
        predicted = analysis_df[f'{model_name}_Prediction']
        errors = predicted - actual
        
        # Error decomposition
        error_components = {
            'total_error': errors,
            'absolute_error': np.abs(errors),
            'percentage_error': (errors / actual) * 100,
            'absolute_percentage_error': np.abs(errors / actual) * 100,
            'squared_error': errors ** 2
        }
        
        # Directional errors
        error_components['over_forecast'] = errors[errors > 0]
        error_components['under_forecast'] = errors[errors < 0]
        
        # Sign of error
        error_components['error_sign'] = np.sign(errors)
        
        # Error magnitude categories
        pct_error = np.abs(errors / actual) * 100
        error_components['error_category'] = pd.cut(
            pct_error,
            bins=[0, 10, 20, 30, 50, 100, np.inf],
            labels=['Excellent (<10%)', 'Good (10-20%)', 'Fair (20-30%)', 
                   'Poor (30-50%)', 'Very Poor (50-100%)', 'Extreme (>100%)']
        )
        
        return error_components
    
    def analyze_error_patterns(self, analysis_df, model_name='Prophet'):
        """Analyze patterns in forecast errors"""
        error_components = self.calculate_error_components(analysis_df, model_name)
        
        patterns = {
            'summary': {
                'total_observations': len(analysis_df),
                'mean_absolute_percentage_error': np.mean(error_components['absolute_percentage_error']),
                'mean_absolute_error': np.mean(error_components['absolute_error']),
                'root_mean_squared_error': np.sqrt(np.mean(error_components['squared_error'])),
                'mean_bias': np.mean(error_components['total_error']),
                'std_bias': np.std(error_components['total_error'])
            },
            'distribution': {
                'over_forecast_count': len(error_components['over_forecast']),
                'under_forecast_count': len(error_components['under_forecast']),
                'over_forecast_percentage': len(error_components['over_forecast']) / len(analysis_df) * 100,
                'under_forecast_percentage': len(error_components['under_forecast']) / len(analysis_df) * 100
            },
            'magnitude_distribution': pd.Series(error_components['error_category']).value_counts().to_dict(),
            'temporal_patterns': {},
            'conditional_patterns': {}
        }
        
        # Analyze temporal patterns
        if 'Date' in analysis_df.columns:
            analysis_df['Error'] = error_components['total_error']
            analysis_df['Abs_Pct_Error'] = error_components['absolute_percentage_error']
            
            # By month
            analysis_df['Month'] = analysis_df['Date'].dt.month
            monthly_errors = analysis_df.groupby('Month').agg({
                'Error': ['mean', 'std', 'count'],
                'Abs_Pct_Error': 'mean'
            })
            patterns['temporal_patterns']['monthly'] = monthly_errors.to_dict()
            
            # By year
            analysis_df['Year'] = analysis_df['Date'].dt.year
            yearly_errors = analysis_df.groupby('Year').agg({
                'Error': ['mean', 'std', 'count'],
                'Abs_Pct_Error': 'mean'
            })
            patterns['temporal_patterns']['yearly'] = yearly_errors.to_dict()
        
        # Analyze conditional patterns
        if 'IsHoliday' in analysis_df.columns:
            holiday_errors = analysis_df.groupby('IsHoliday').agg({
                'Error': 'mean',
                'Abs_Pct_Error': 'mean'
            })
            patterns['conditional_patterns']['holiday'] = holiday_errors.to_dict()
        
        # Analyze by sales volume
        if 'Actual' in analysis_df.columns:
            sales_quantiles = analysis_df['Actual'].quantile([0.25, 0.5, 0.75])
            analysis_df['Sales_Quantile'] = pd.qcut(
                analysis_df['Actual'], 
                q=[0, 0.25, 0.5, 0.75, 1],
                labels=['Very Low', 'Low', 'Medium', 'High']
            )
            
            quantile_errors = analysis_df.groupby('Sales_Quantile').agg({
                'Error': 'mean',
                'Abs_Pct_Error': 'mean',
                'Actual': 'count'
            })
            patterns['conditional_patterns']['sales_volume'] = quantile_errors.to_dict()
        
        return patterns
    
    def identify_error_clusters(self, analysis_df, model_name='Prophet', n_clusters=3):
        """Identify clusters of similar errors"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for clustering
            features = []
            feature_names = []
            
            # Temporal features
            if 'Date' in analysis_df.columns:
                features.append(analysis_df['Date'].dt.month.values.reshape(-1, 1))
                feature_names.append('month')
                features.append(analysis_df['Date'].dt.week.values.reshape(-1, 1))
                feature_names.append('week')
            
            # Sales features
            if 'Actual' in analysis_df.columns:
                features.append(analysis_df['Actual'].values.reshape(-1, 1))
                feature_names.append('sales_volume')
            
            # Error magnitude
            error_components = self.calculate_error_components(analysis_df, model_name)
            features.append(error_components['absolute_percentage_error'].values.reshape(-1, 1))
            feature_names.append('error_magnitude')
            
            # Combine features
            X = np.hstack(features)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Analyze clusters
            analysis_df['Error_Cluster'] = clusters
            cluster_analysis = {}
            
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_data = analysis_df[cluster_mask]
                
                cluster_stats = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(analysis_df) * 100,
                    'mean_error': cluster_data['Error'].mean() if 'Error' in cluster_data.columns else None,
                    'mean_abs_pct_error': cluster_data['Abs_Pct_Error'].mean() if 'Abs_Pct_Error' in cluster_data.columns else None
                }
                
                # Add feature means for interpretation
                for i, feature_name in enumerate(feature_names):
                    cluster_stats[f'mean_{feature_name}'] = cluster_data.iloc[:, i].mean() if i < len(cluster_data.columns) else None
                
                cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats
            
            # Calculate cluster characteristics
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            for i, center in enumerate(cluster_centers):
                cluster_analysis[f'cluster_{i}']['center'] = {
                    feature_names[j]: center[j] for j in range(len(feature_names))
                }
            
            return cluster_analysis
            
        except Exception as e:
            logger.warning(f"Could not perform clustering analysis: {e}")
            return {}
    
    def analyze_feature_importance_for_errors(self, analysis_df, model_name='Prophet'):
        """Analyze which features are most predictive of forecast errors"""
        try:
            # Calculate errors
            error_components = self.calculate_error_components(analysis_df, model_name)
            analysis_df['Abs_Pct_Error'] = error_components['absolute_percentage_error']
            
            # Select numeric features for analysis
            numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target-related columns
            exclude_cols = ['Actual', 'Abs_Pct_Error', 'Error'] + \
                          [col for col in analysis_df.columns if 'Prediction' in col or 'lag' in col]
            
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(feature_cols) < 2:
                logger.warning("Not enough features for importance analysis")
                return {}
            
            # Prepare data
            X = analysis_df[feature_cols].fillna(0)
            y = analysis_df['Abs_Pct_Error']
            
            # Remove features with zero variance
            X = X.loc[:, X.std() > 0]
            
            if len(X.columns) < 2:
                logger.warning("Not enough features with variance for importance analysis")
                return {}
            
            # Train random forest to predict error magnitude
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importances
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Calculate correlations with error
            correlations = []
            for col in X.columns:
                corr = analysis_df[col].corr(analysis_df['Abs_Pct_Error'])
                correlations.append(corr)
            
            importances['correlation'] = correlations
            
            return importances.to_dict('records')
            
        except Exception as e:
            logger.warning(f"Could not perform feature importance analysis: {e}")
            return {}
    
    def detect_systematic_biases(self, analysis_df, model_name='Prophet'):
        """Detect systematic biases in forecasts"""
        error_components = self.calculate_error_components(analysis_df, model_name)
        
        biases = {
            'overall_bias': {
                'mean_error': np.mean(error_components['total_error']),
                'median_error': np.median(error_components['total_error']),
                'percent_positive': (error_components['total_error'] > 0).mean() * 100,
                'percent_negative': (error_components['total_error'] < 0).mean() * 100,
                'is_significant_bias': None
            },
            'conditional_biases': {},
            'temporal_biases': {}
        }
        
        # Test for significant bias
        t_stat, p_value = stats.ttest_1samp(error_components['total_error'], 0)
        biases['overall_bias']['is_significant_bias'] = p_value < 0.05
        biases['overall_bias']['t_test_pvalue'] = p_value
        
        # Check for bias in specific conditions
        if 'IsHoliday' in analysis_df.columns:
            holiday_bias = analysis_df.groupby('IsHoliday')['Error'].mean()
            biases['conditional_biases']['holiday'] = holiday_bias.to_dict()
        
        # Check for monthly bias
        if 'Date' in analysis_df.columns:
            analysis_df['Month'] = analysis_df['Date'].dt.month
            monthly_bias = analysis_df.groupby('Month')['Error'].mean()
            biases['temporal_biases']['monthly'] = monthly_bias.to_dict()
            
            # Check for bias trend over time
            analysis_df['Time_Index'] = range(len(analysis_df))
            time_corr = analysis_df['Time_Index'].corr(analysis_df['Error'])
            biases['temporal_biases']['time_trend_correlation'] = time_corr
        
        return biases
    
    def generate_root_cause_analysis(self, analysis_df, model_name='Prophet'):
        """Generate root cause analysis for forecast errors"""
        try:
            # Perform all analyses
            error_patterns = self.analyze_error_patterns(analysis_df, model_name)
            error_clusters = self.identify_error_clusters(analysis_df, model_name)
            feature_importance = self.analyze_feature_importance_for_errors(analysis_df, model_name)
            systematic_biases = self.detect_systematic_biases(analysis_df, model_name)
            
            # Combine analyses
            root_causes = {
                'error_patterns': error_patterns,
                'error_clusters': error_clusters,
                'feature_importance': feature_importance,
                'systematic_biases': systematic_biases,
                'identified_issues': [],
                'recommended_actions': []
            }
            
            # Identify specific issues
            self._identify_issues(root_causes)
            
            # Generate recommendations
            self._generate_recommendations(root_causes)
            
            return root_causes
            
        except Exception as e:
            logger.error(f"Error in root cause analysis: {e}")
            raise
    
    def _identify_issues(self, root_causes):
        """Identify specific issues from analysis"""
        issues = []
        
        # Check for significant bias
        if root_causes['systematic_biases']['overall_bias']['is_significant_bias']:
            mean_error = root_causes['systematic_biases']['overall_bias']['mean_error']
            direction = "over" if mean_error > 0 else "under"
            issues.append({
                'issue': 'Systematic forecasting bias',
                'severity': 'High',
                'description': f'Model consistently {direction}-forecasts by ${abs(mean_error):.0f} on average',
                'evidence': f'Mean error: ${mean_error:.0f}, p-value: {root_causes["systematic_biases"]["overall_bias"]["t_test_pvalue"]:.4f}'
            })
        
        # Check holiday performance
        if 'holiday' in root_causes['error_patterns']['conditional_patterns']:
            holiday_error = root_causes['error_patterns']['conditional_patterns']['holiday'].get(1, {}).get('Abs_Pct_Error', {}).get('mean', 0)
            non_holiday_error = root_causes['error_patterns']['conditional_patterns']['holiday'].get(0, {}).get('Abs_Pct_Error', {}).get('mean', 0)
            
            if holiday_error > non_holiday_error * 1.5:
                issues.append({
                    'issue': 'Poor holiday forecast accuracy',
                    'severity': 'Medium',
                    'description': f'Holiday forecasts are {holiday_error/non_holiday_error:.1f}x less accurate than non-holiday forecasts',
                    'evidence': f'Holiday MAPE: {holiday_error:.1f}%, Non-holiday MAPE: {non_holiday_error:.1f}%'
                })
        
        # Check for high error periods
        if 'monthly' in root_causes['error_patterns']['temporal_patterns']:
            monthly_errors = root_causes['error_patterns']['temporal_patterns']['monthly']
            worst_month = max(monthly_errors.items(), key=lambda x: x[1].get('Abs_Pct_Error', {}).get('mean', 0))
            best_month = min(monthly_errors.items(), key=lambda x: x[1].get('Abs_Pct_Error', {}).get('mean', 0))
            
            if worst_month[1].get('Abs_Pct_Error', {}).get('mean', 0) > best_month[1].get('Abs_Pct_Error', {}).get('mean', 0) * 2:
                issues.append({
                    'issue': 'Seasonal forecast inconsistency',
                    'severity': 'Medium',
                    'description': f'Forecast accuracy varies significantly by month',
                    'evidence': f'Worst month ({worst_month[0]}): {worst_month[1].get("Abs_Pct_Error", {}).get("mean", 0):.1f}%, Best month ({best_month[0]}): {best_month[1].get("Abs_Pct_Error", {}).get("mean", 0):.1f}%'
                })
        
        root_causes['identified_issues'] = issues
    
    def _generate_recommendations(self, root_causes):
        """Generate recommendations based on identified issues"""
        recommendations = []
        
        for issue in root_causes['identified_issues']:
            if issue['issue'] == 'Systematic forecasting bias':
                direction = "over" if "over" in issue['description'] else "under"
                recommendations.append(
                    f"Implement bias correction for {direction}-forecasting. "
                    f"Consider adjusting the model intercept or using error correction techniques."
                )
            
            elif issue['issue'] == 'Poor holiday forecast accuracy':
                recommendations.append(
                    "Develop separate holiday forecasting models or incorporate "
                    "holiday-specific features to improve accuracy during peak periods."
                )
            
            elif issue['issue'] == 'Seasonal forecast inconsistency':
                recommendations.append(
                    "Enhance seasonal modeling by incorporating month-specific "
                    "parameters or using more sophisticated seasonal decomposition methods."
                )
        
        # Additional recommendations based on feature importance
        if root_causes['feature_importance']:
            top_features = root_causes['feature_importance'][:3]
            if top_features:
                recommendations.append(
                    f"Focus on improving features related to: {', '.join([f['feature'] for f in top_features])}. "
                    f"These show the highest correlation with forecast errors."
                )
        
        root_causes['recommended_actions'] = recommendations
    
    def create_error_visualizations(self, analysis_df, model_name='Prophet', 
                                  store_id=None, dept_id=None, output_dir='../monitoring'):
        """Create visualizations for error analysis"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        error_components = self.calculate_error_components(analysis_df, model_name)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Error over time
        axes[0, 0].plot(analysis_df['Date'], error_components['total_error'], 
                        color='#2E86AB', linewidth=2)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].fill_between(analysis_df['Date'], 0, error_components['total_error'],
                               where=error_components['total_error'] > 0,
                               color='red', alpha=0.3, label='Over-forecast')
        axes[0, 0].fill_between(analysis_df['Date'], 0, error_components['total_error'],
                               where=error_components['total_error'] < 0,
                               color='blue', alpha=0.3, label='Under-forecast')
        axes[0, 0].set_title('Forecast Error Over Time', fontsize=12)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Error ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        axes[0, 1].hist(error_components['total_error'], bins=30,
                       color='#F18F01', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Error Distribution', fontsize=12)
        axes[0, 1].set_xlabel('Forecast Error ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Percentage error by month
        if 'Date' in analysis_df.columns:
            analysis_df['Month'] = analysis_df['Date'].dt.month
            monthly_error = analysis_df.groupby('Month')['Abs_Pct_Error'].mean()
            axes[0, 2].bar(monthly_error.index, monthly_error.values,
                          color='#C73E1D', alpha=0.7)
            axes[0, 2].set_title('Average Percentage Error by Month', fontsize=12)
            axes[0, 2].set_xlabel('Month')
            axes[0, 2].set_ylabel('MAPE (%)')
            axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Error by sales volume
        if 'Actual' in analysis_df.columns:
            analysis_df['Sales_Bin'] = pd.qcut(analysis_df['Actual'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
            sales_error = analysis_df.groupby('Sales_Bin')['Abs_Pct_Error'].mean()
            axes[1, 0].bar(sales_error.index, sales_error.values,
                          color='#A23B72', alpha=0.7)
            axes[1, 0].set_title('Error by Sales Volume', fontsize=12)
            axes[1, 0].set_xlabel('Sales Volume Quartile')
            axes[1, 0].set_ylabel('MAPE (%)')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Holiday vs Non-holiday error
        if 'IsHoliday' in analysis_df.columns:
            holiday_error = analysis_df.groupby('IsHoliday')['Abs_Pct_Error'].mean()
            axes[1, 1].bar(['Non-Holiday', 'Holiday'], holiday_error.values,
                          color=['#2E86AB', '#F18F01'], alpha=0.7)
            axes[1, 1].set_title('Error: Holiday vs Non-Holiday', fontsize=12)
            axes[1, 1].set_ylabel('MAPE (%)')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Cumulative error
        cumulative_error = error_components['total_error'].cumsum()
        axes[1, 2].plot(analysis_df['Date'], cumulative_error,
                       color='#2E86AB', linewidth=2)
        axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Cumulative Forecast Error', fontsize=12)
        axes[1, 2].set_xlabel('Date')
        axes[1, 2].set_ylabel('Cumulative Error ($)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add title
        title = f'Error Analysis - {model_name}'
        if store_id and dept_id:
            title += f' (Store {store_id}, Dept {dept_id})'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        viz_path = output_dir / f'error_analysis_{model_name.lower()}'
        if store_id and dept_id:
            viz_path = output_dir / f'error_analysis_{model_name.lower()}_store_{store_id}_dept_{dept_id}.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved error analysis visualizations to {viz_path}")
    
    def save_analysis_report(self, root_causes, store_id, dept_id, model_name='Prophet',
                            output_dir='../monitoring'):
        """Save error analysis report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f'error_analysis_{model_name.lower()}_store_{store_id}_dept_{dept_id}.json'
        
        with open(report_path, 'w') as f:
            json.dump(root_causes, f, indent=2)
        
        logger.info(f"Saved error analysis report to {report_path}")
        return report_path

def main():
    """Main error analysis pipeline"""
    import sys
    sys.path.append('..')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Analyze errors for Store 1, Dept 1, Prophet model
    store_id = 1
    dept_id = 1
    model_name = 'Prophet'
    
    analyzer = ErrorAnalyzer()
    
    print(f"Analyzing forecast errors for Store {store_id}, Department {dept_id}")
    print(f"Model: {model_name}")
    print("=" * 50)
    
    # Load forecast data
    analysis_df = analyzer.load_forecast_data(store_id, dept_id)
    
    # Generate root cause analysis
    root_causes = analyzer.generate_root_cause_analysis(analysis_df, model_name)
    
    # Create visualizations
    analyzer.create_error_visualizations(analysis_df, model_name, store_id, dept_id)
    
    # Save analysis report
    report_path = analyzer.save_analysis_report(root_causes, store_id, dept_id, model_name)
    
    print("\n" + "="*50)
    print("ERROR ANALYSIS COMPLETED")
    print("="*50)
    
    print(f"\nOverall Performance:")
    print(f"  MAPE: {root_causes['error_patterns']['summary']['mean_absolute_percentage_error']:.1f}%")
    print(f"  MAE: ${root_causes['error_patterns']['summary']['mean_absolute_error']:.0f}")
    print(f"  Bias: ${root_causes['error_patterns']['summary']['mean_bias']:.0f}")
    
    print(f"\nError Distribution:")
    print(f"  Over-forecast: {root_causes['error_patterns']['distribution']['over_forecast_percentage']:.1f}%")
    print(f"  Under-forecast: {root_causes['error_patterns']['distribution']['under_forecast_percentage']:.1f}%")
    
    if root_causes['identified_issues']:
        print(f"\nIdentified Issues:")
        for i, issue in enumerate(root_causes['identified_issues'], 1):
            print(f"  {i}. {issue['issue']} ({issue['severity']})")
            print(f"      {issue['description']}")
    
    if root_causes['recommended_actions']:
        print(f"\nRecommended Actions:")
        for i, action in enumerate(root_causes['recommended_actions'], 1):
            print(f"  {i}. {action}")
    
    print(f"\n✅ Analysis report saved to {report_path}")
    print(f"✅ Visualizations saved to monitoring/error_analysis_{model_name.lower()}_store_{store_id}_dept_{dept_id}.png")

if __name__ == "__main__":
    main()