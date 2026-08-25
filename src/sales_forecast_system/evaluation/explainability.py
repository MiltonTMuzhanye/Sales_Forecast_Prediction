import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging
import matplotlib.pyplot as plt
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class ModelExplainer:
    """Model explainability tools"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = None
        self.feature_names = None
        
    def set_model(self, model: Any, feature_names: List[str]) -> None:
        """Set model and feature names"""
        self.model = model
        self.feature_names = feature_names
        logger.info(f"Model set with {len(feature_names)} features")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        if self.model is None:
            raise ValueError("Model not set")
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                return df
            else:
                raise ValueError("Model does not support feature importance")
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            raise
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance"""
        df = self.get_feature_importance()
        df_top = df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(df_top['feature'], df_top['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()