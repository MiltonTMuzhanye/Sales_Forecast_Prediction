import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import matplotlib.pyplot as plt
from scipy import stats
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class ResidualAnalyzer:
    """Analyzes residuals from time series forecasts"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                         plot: bool = True) -> Dict:
        """Comprehensive residual analysis"""
        logger.info("Analyzing residuals...")
        
        residuals = y_pred - y_true
        percent_residuals = (residuals / (np.abs(y_true) + 1e-8)) * 100
        
        analysis = {
            'residuals': residuals,
            'percent_residuals': percent_residuals,
            'statistics': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals)
            },
            'percentiles': {
                '5%': np.percentile(percent_residuals, 5),
                '25%': np.percentile(percent_residuals, 25),
                '50%': np.percentile(percent_residuals, 50),
                '75%': np.percentile(percent_residuals, 75),
                '95%': np.percentile(percent_residuals, 95)
            }
        }
        
        # Normality test
        if len(residuals) > 8:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            analysis['normality_test'] = {
                'shapiro_stat': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        if plot:
            self.plot_residual_analysis(y_true, residuals, percent_residuals)
        
        return analysis
    
    def plot_residual_analysis(self, y_true: np.ndarray, residuals: np.ndarray,
                              percent_residuals: np.ndarray) -> None:
        """Generate residual analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(y_true, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residual Distribution
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Percentage Error Distribution
        axes[1, 1].hist(percent_residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Percentage Error (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Percentage Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()