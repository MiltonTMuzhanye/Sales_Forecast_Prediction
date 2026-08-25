import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files"""
        config = {}
        
        # Load main config
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config.update(yaml.safe_load(f))
        
        # Load model config
        model_config_path = Path("configs/model.yaml")
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                config['model'] = yaml.safe_load(f)
        
        # Load data config
        data_config_path = Path("configs/data.yaml")
        if data_config_path.exists():
            with open(data_config_path, 'r') as f:
                config['data'] = yaml.safe_load(f)
        
        # Load forecast config
        forecast_config_path = Path("configs/forecast_config.yaml")
        if forecast_config_path.exists():
            with open(forecast_config_path, 'r') as f:
                config['forecast'] = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, path: str = None) -> None:
        """Save configuration to file"""
        if path is None:
            path = self.config_path
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)