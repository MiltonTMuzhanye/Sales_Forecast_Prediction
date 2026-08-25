import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import ModelTrainingError

logger = setup_logger(__name__)

class LSTMModel:
    """LSTM model wrapper for time series forecasting"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback = self.config.get('model.models.lstm.lookback', 12)
        self.epochs = self.config.get('model.models.lstm.epochs', 50)
        self.batch_size = self.config.get('model.models.lstm.batch_size', 32)
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: int) -> None:
        """Build LSTM model"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(input_shape, 1)),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            self.model = model
            logger.info("LSTM model built successfully")
            
        except ImportError:
            logger.warning("TensorFlow not installed. LSTM model not available.")
            raise ModelTrainingError("TensorFlow required for LSTM model")
    
    def train(self, train_data: np.ndarray) -> None:
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        
        try:
            scaled_data = self.scaler.fit_transform(train_data.reshape(-1, 1))
            X, y = self._create_sequences(scaled_data)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            if self.model is None:
                self._build_model(X.shape[1])
            
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, 
                          verbose=0, validation_split=0.1)
            logger.info("LSTM model trained successfully")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train LSTM model: {e}")
    
    def predict(self, test_data: np.ndarray, steps: int) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        logger.info(f"Making predictions for {steps} steps...")
        
        try:
            scaled_test = self.scaler.transform(test_data.reshape(-1, 1))
            current_sequence = scaled_test[-self.lookback:].reshape(1, self.lookback, 1)
            
            predictions = []
            for _ in range(steps):
                next_val = self.model.predict(current_sequence, verbose=0)[0, 0]
                predictions.append(next_val)
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_val
            
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            return predictions.flatten()
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to make predictions: {e}")
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate model"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }