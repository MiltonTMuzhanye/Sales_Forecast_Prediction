import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

def ensure_directory(path: str) -> None:
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def date_range(start_date: str, periods: int, freq: str = 'W') -> List[str]:
    """Generate date range"""
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    return dates.strftime('%Y-%m-%d').tolist()

def save_json(data: Dict, filepath: str) -> None:
    """Save dictionary as JSON"""
    ensure_directory(str(Path(filepath).parent))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_seasonal_indices(data: pd.Series, period: int = 52) -> np.ndarray:
    """Calculate seasonal indices"""
    if len(data) < period:
        return np.ones(period)
    
    # Reshape data into seasonal periods
    n_periods = len(data) // period
    if n_periods < 1:
        return np.ones(period)
    
    seasonal_data = data.values[:n_periods * period].reshape(n_periods, period)
    seasonal_means = np.mean(seasonal_data, axis=0)
    overall_mean = np.mean(seasonal_data)
    
    if overall_mean == 0:
        return np.ones(period)
    
    return seasonal_means / overall_mean

def remove_outliers(df: pd.DataFrame, column: str, 
                   method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """Remove outliers from DataFrame"""
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_copy = df_copy[(df_copy[column] >= lower_bound) & 
                         (df_copy[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df_copy[column] - df_copy[column].mean()) / df_copy[column].std())
        df_copy = df_copy[z_scores < threshold]
    
    return df_copy

def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Create time-based features"""
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    df_copy['year'] = df_copy[date_col].dt.year
    df_copy['month'] = df_copy[date_col].dt.month
    df_copy['day'] = df_copy[date_col].dt.day
    df_copy['dayofweek'] = df_copy[date_col].dt.dayofweek
    df_copy['quarter'] = df_copy[date_col].dt.quarter
    df_copy['week'] = df_copy[date_col].dt.isocalendar().week
    df_copy['dayofyear'] = df_copy[date_col].dt.dayofyear
    
    return df_copy