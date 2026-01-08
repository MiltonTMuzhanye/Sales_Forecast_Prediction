import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate data quality and integrity"""
    
    def __init__(self):
        self.validation_results = {}
    
    def check_date_continuity(self, df, date_col='Date', group_cols=['Store', 'Dept']):
        """Check for missing dates in time series"""
        results = {}
        
        for _, group in df.groupby(group_cols):
            # Create complete date range
            min_date = group[date_col].min()
            max_date = group[date_col].max()
            expected_dates = pd.date_range(start=min_date, end=max_date, freq='W-FRI')
            
            # Find missing dates
            actual_dates = set(group[date_col])
            missing_dates = set(expected_dates) - actual_dates
            
            if missing_dates:
                key = tuple(group[group_cols].iloc[0][group_cols])
                results[key] = {
                    'missing_count': len(missing_dates),
                    'total_expected': len(expected_dates),
                    'completeness': (len(expected_dates) - len(missing_dates)) / len(expected_dates),
                    'missing_dates': sorted(missing_dates)[:10]  # Show first 10
                }
        
        self.validation_results['date_continuity'] = results
        return results
    
    def check_sales_consistency(self, df, sales_col='Weekly_Sales'):
        """Check for negative or zero sales where not expected"""
        results = {}
        
        # Check for negative sales
        negative_sales = df[df[sales_col] < 0]
        if len(negative_sales) > 0:
            results['negative_sales'] = {
                'count': len(negative_sales),
                'percentage': len(negative_sales) / len(df) * 100,
                'examples': negative_sales[[sales_col, 'Store', 'Dept', 'Date']].head().to_dict('records')
            }
        
        # Check for zero sales
        zero_sales = df[df[sales_col] == 0]
        if len(zero_sales) > 0:
            results['zero_sales'] = {
                'count': len(zero_sales),
                'percentage': len(zero_sales) / len(df) * 100,
                'examples': zero_sales[[sales_col, 'Store', 'Dept', 'Date']].head().to_dict('records')
            }
        
        # Check for extreme values
        median = df[sales_col].median()
        std = df[sales_col].std()
        extreme = df[np.abs(df[sales_col] - median) > 5 * std]
        if len(extreme) > 0:
            results['extreme_values'] = {
                'count': len(extreme),
                'percentage': len(extreme) / len(df) * 100,
                'threshold': f'{median:.0f} ± {5*std:.0f}'
            }
        
        self.validation_results['sales_consistency'] = results
        return results
    
    def check_feature_ranges(self, df):
        """Check that features are within reasonable ranges"""
        results = {}
        
        expected_ranges = {
            'Temperature': (-50, 120),  # Fahrenheit
            'Fuel_Price': (0, 10),      # Dollars per gallon
            'CPI': (100, 300),          # Consumer Price Index
            'Unemployment': (0, 25),    # Percentage
            'MarkDown1': (0, None),     # Non-negative
            'MarkDown2': (0, None),
            'MarkDown3': (0, None),
            'MarkDown4': (0, None),
            'MarkDown5': (0, None)
        }
        
        for feature, (min_val, max_val) in expected_ranges.items():
            if feature in df.columns:
                issues = {}
                
                if min_val is not None:
                    below_min = df[df[feature] < min_val]
                    if len(below_min) > 0:
                        issues['below_min'] = {
                            'count': len(below_min),
                            'min_found': below_min[feature].min(),
                            'expected_min': min_val
                        }
                
                if max_val is not None:
                    above_max = df[df[feature] > max_val]
                    if len(above_max) > 0:
                        issues['above_max'] = {
                            'count': len(above_max),
                            'max_found': above_max[feature].max(),
                            'expected_max': max_val
                        }
                
                if issues:
                    results[feature] = issues
        
        self.validation_results['feature_ranges'] = results
        return results
    
    def check_holiday_consistency(self, df):
        """Check holiday flag consistency"""
        results = {}
        
        if 'IsHoliday' in df.columns:
            # Check if holidays align with known dates
            known_holidays = {
                'Super Bowl': '02-*',
                'Labor Day': '09-*',
                'Thanksgiving': '11-*',
                'Christmas': '12-*'
            }
            
            holiday_dates = df[df['IsHoliday'] == 1]['Date']
            holiday_months = holiday_dates.dt.month.value_counts().sort_index()
            
            results['holiday_distribution'] = {
                'total_holidays': len(holiday_dates),
                'unique_dates': len(holiday_dates.unique()),
                'by_month': holiday_months.to_dict()
            }
        
        self.validation_results['holiday_consistency'] = results
        return results
    
    def check_store_department_coverage(self, df):
        """Check that all stores and departments have consistent data"""
        results = {}
        
        # Store coverage
        store_counts = df['Store'].value_counts()
        store_stats = {
            'total_stores': len(store_counts),
            'min_weeks': store_counts.min(),
            'max_weeks': store_counts.max(),
            'avg_weeks': store_counts.mean(),
            'stores_with_insufficient_data': len(store_counts[store_counts < store_counts.quantile(0.25)])
        }
        
        # Department coverage
        dept_counts = df['Dept'].value_counts()
        dept_stats = {
            'total_depts': len(dept_counts),
            'min_weeks': dept_counts.min(),
            'max_weeks': dept_counts.max(),
            'avg_weeks': dept_counts.mean()
        }
        
        # Store-department combinations
        combos = df.groupby(['Store', 'Dept']).size()
        combo_stats = {
            'total_combinations': len(combos),
            'min_weeks_per_combo': combos.min(),
            'max_weeks_per_combo': combos.max(),
            'avg_weeks_per_combo': combos.mean()
        }
        
        results['store_coverage'] = store_stats
        results['department_coverage'] = dept_stats
        results['combination_coverage'] = combo_stats
        
        self.validation_results['coverage'] = results
        return results
    
    def run_all_checks(self, df):
        """Run all validation checks"""
        logger.info("Starting data validation...")
        
        self.check_date_continuity(df)
        self.check_sales_consistency(df)
        self.check_feature_ranges(df)
        self.check_holiday_consistency(df)
        self.check_store_department_coverage(df)
        
        logger.info("Data validation completed")
        return self.validation_results
    
    def generate_report(self):
        """Generate validation report"""
        report = {
            'summary': {},
            'issues': [],
            'recommendations': []
        }
        
        total_issues = 0
        critical_issues = 0
        
        # Analyze date continuity
        if 'date_continuity' in self.validation_results:
            for key, result in self.validation_results['date_continuity'].items():
                if result['completeness'] < 0.95:
                    total_issues += 1
                    critical_issues += 1
                    report['issues'].append({
                        'type': 'critical',
                        'description': f'Missing dates for {key}: {result["missing_count"]} missing weeks',
                        'impact': 'Time series analysis will be affected',
                        'action': 'Investigate and impute missing dates'
                    })
        
        # Analyze sales consistency
        if 'sales_consistency' in self.validation_results:
            sales_issues = self.validation_results['sales_consistency']
            if 'negative_sales' in sales_issues:
                total_issues += 1
                report['issues'].append({
                    'type': 'warning',
                    'description': f'Negative sales found: {sales_issues["negative_sales"]["count"]} records',
                    'impact': 'May indicate data entry errors or returns',
                    'action': 'Review negative sales transactions'
                })
            
            if 'extreme_values' in sales_issues:
                total_issues += 1
                report['issues'].append({
                    'type': 'warning',
                    'description': f'Extreme sales values: {sales_issues["extreme_values"]["count"]} records',
                    'impact': 'Could skew statistical analysis',
                    'action': 'Review extreme values for validity'
                })
        
        # Generate summary
        report['summary'] = {
            'total_checks': len(self.validation_results),
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'data_quality_score': max(0, 100 - total_issues * 10 - critical_issues * 20)
        }
        
        # Add recommendations
        if critical_issues > 0:
            report['recommendations'].append('Address critical data quality issues before modeling')
        if total_issues > 5:
            report['recommendations'].append('Consider data cleaning and imputation steps')
        
        report['recommendations'].append('Regularly monitor data quality metrics')
        report['recommendations'].append('Implement automated data validation pipeline')
        
        return report

def main():
    """Main validation pipeline"""
    import sys
    sys.path.append('..')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load cleaned data
    try:
        df = pd.read_parquet('../data/processed/cleaned_sales.parquet')
        print(f"Loaded data: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run validation
    validator = DataValidator()
    validation_results = validator.run_all_checks(df)
    
    # Generate report
    report = validator.generate_report()
    
    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60)
    
    print(f"\nSummary:")
    print(f"  Data Quality Score: {report['summary']['data_quality_score']}/100")
    print(f"  Total Issues: {report['summary']['total_issues']}")
    print(f"  Critical Issues: {report['summary']['critical_issues']}")
    
    if report['issues']:
        print(f"\nIssues Found:")
        for issue in report['issues']:
            print(f"  [{issue['type'].upper()}] {issue['description']}")
            print(f"      Impact: {issue['impact']}")
            print(f"      Action: {issue['action']}")
            print()
    else:
        print("\n✅ No issues found. Data quality is good.")
    
    print("Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save report
    import json
    with open('../monitoring/data_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved to monitoring/data_validation_report.json")

if __name__ == "__main__":
    main()