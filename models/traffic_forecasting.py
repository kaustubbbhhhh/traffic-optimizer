"""
LSTM-based Traffic Forecasting Module
Predicts future traffic patterns based on historical data
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
import pickle

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not installed. Using statistical forecasting.")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
sys.path.append('..')
from config import (LSTM_SEQUENCE_LENGTH, LSTM_PREDICTION_HORIZON,
                   LSTM_HIDDEN_UNITS, LSTM_EPOCHS, LSTM_BATCH_SIZE,
                   MODELS_DIR, DATA_DIR)


@dataclass
class ForecastResult:
    """Traffic forecast result"""
    timestamps: List[datetime]
    predictions: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    density_levels: List[str]


class TrafficForecaster:
    """
    LSTM-based traffic flow forecasting
    Predicts vehicle counts for future time periods
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the traffic forecaster
        
        Args:
            model_path: Path to pre-trained LSTM model
        """
        self.model_path = model_path or str(MODELS_DIR / "traffic_lstm.h5")
        self.scaler_path = str(MODELS_DIR / "traffic_scaler.pkl")
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = LSTM_SEQUENCE_LENGTH
        self.prediction_horizon = LSTM_PREDICTION_HORIZON
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if exists"""
        if TF_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                if os.path.exists(self.scaler_path):
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                print(f"✓ LSTM model loaded: {self.model_path}")
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                self.model = None
    
    def _build_model(self, input_shape: Tuple[int, int]) -> 'Sequential':
        """
        Build LSTM model architecture
        
        Args:
            input_shape: (sequence_length, num_features)
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(LSTM_HIDDEN_UNITS, return_sequences=True, 
                 input_shape=input_shape),
            Dropout(0.2),
            LSTM(LSTM_HIDDEN_UNITS // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.prediction_horizon)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame, 
                     target_col: str = 'vehicle_count') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data: DataFrame with timestamp and vehicle_count columns
            target_col: Name of target column
        
        Returns:
            X (sequences), y (targets)
        """
        # Extract features
        df = data.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Select features
        features = df[['vehicle_count', 'hour', 'day_of_week', 'is_weekend']].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length - self.prediction_horizon + 1):
            X.append(scaled_features[i:i + self.sequence_length])
            y.append(scaled_features[i + self.sequence_length:
                                    i + self.sequence_length + self.prediction_horizon, 0])
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, 
              validation_split: float = 0.2,
              verbose: int = 1) -> dict:
        """
        Train the LSTM model
        
        Args:
            data: Training data with timestamp and vehicle_count
            validation_split: Fraction of data for validation
            verbose: Training verbosity
        
        Returns:
            Training history
        """
        if not TF_AVAILABLE:
            print("TensorFlow not available. Cannot train LSTM model.")
            return {}
        
        X, y = self.prepare_data(data)
        
        if len(X) == 0:
            print("Not enough data to train model")
            return {}
        
        # Build model
        self.model = self._build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True)
        ]
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Save scaler
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return history.history
    
    def forecast(self, recent_data: pd.DataFrame) -> ForecastResult:
        """
        Generate traffic forecast
        
        Args:
            recent_data: Recent traffic data (last sequence_length hours)
        
        Returns:
            ForecastResult with predictions
        """
        if self.model is not None and TF_AVAILABLE:
            return self._lstm_forecast(recent_data)
        else:
            return self._statistical_forecast(recent_data)
    
    def _lstm_forecast(self, recent_data: pd.DataFrame) -> ForecastResult:
        """LSTM-based forecast"""
        # Prepare input sequence
        df = recent_data.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        features = df[['vehicle_count', 'hour', 'day_of_week', 'is_weekend']].values
        scaled = self.scaler.transform(features)
        
        # Use last sequence_length points
        input_seq = scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Predict
        predictions = self.model.predict(input_seq, verbose=0)[0]
        
        # Inverse transform (only vehicle count)
        predictions_full = np.zeros((len(predictions), 4))
        predictions_full[:, 0] = predictions
        predictions = self.scaler.inverse_transform(predictions_full)[:, 0]
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        
        # Generate timestamps
        last_time = recent_data['timestamp'].iloc[-1]
        timestamps = [last_time + timedelta(hours=i+1) 
                     for i in range(self.prediction_horizon)]
        
        # Confidence intervals (simple estimation)
        std = np.std(recent_data['vehicle_count'].values) * 0.5
        confidence_lower = predictions - 1.96 * std
        confidence_upper = predictions + 1.96 * std
        
        # Density levels
        density_levels = [self._get_density(p) for p in predictions]
        
        return ForecastResult(
            timestamps=timestamps,
            predictions=predictions,
            confidence_lower=np.maximum(confidence_lower, 0),
            confidence_upper=confidence_upper,
            density_levels=density_levels
        )
    
    def _statistical_forecast(self, recent_data: pd.DataFrame) -> ForecastResult:
        """
        Simple statistical forecast (fallback when LSTM unavailable)
        Uses moving average with hourly patterns
        """
        df = recent_data.copy()
        
        # Calculate hourly averages
        df['hour'] = df['timestamp'].dt.hour
        hourly_avg = df.groupby('hour')['vehicle_count'].mean()
        
        # Moving average
        ma = df['vehicle_count'].rolling(window=6, min_periods=1).mean().iloc[-1]
        
        # Generate predictions
        last_time = df['timestamp'].iloc[-1]
        timestamps = []
        predictions = []
        
        for i in range(self.prediction_horizon):
            next_time = last_time + timedelta(hours=i+1)
            timestamps.append(next_time)
            
            # Blend moving average with hourly pattern
            hour = next_time.hour
            hourly_factor = hourly_avg.get(hour, ma) / hourly_avg.mean() if hourly_avg.mean() > 0 else 1
            pred = ma * hourly_factor
            predictions.append(max(pred, 0))
        
        predictions = np.array(predictions)
        std = df['vehicle_count'].std() * 0.5
        
        return ForecastResult(
            timestamps=timestamps,
            predictions=predictions,
            confidence_lower=np.maximum(predictions - 1.96 * std, 0),
            confidence_upper=predictions + 1.96 * std,
            density_levels=[self._get_density(p) for p in predictions]
        )
    
    def _get_density(self, vehicle_count: float) -> str:
        """Get density level from vehicle count"""
        from config import DENSITY_THRESHOLDS
        
        if vehicle_count < DENSITY_THRESHOLDS["low"]:
            return "low"
        elif vehicle_count < DENSITY_THRESHOLDS["medium"]:
            return "medium"
        elif vehicle_count < DENSITY_THRESHOLDS["high"]:
            return "high"
        else:
            return "critical"


def generate_sample_data(days: int = 30) -> pd.DataFrame:
    """
    Generate synthetic traffic data for testing/training
    
    Args:
        days: Number of days of data to generate
    
    Returns:
        DataFrame with timestamp and vehicle_count
    """
    np.random.seed(42)
    
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, periods=days * 24, freq='H')
    
    data = []
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        
        # Base traffic pattern (rush hours, weekends)
        if day_of_week < 5:  # Weekday
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                base = 35 + np.random.normal(0, 8)
            elif 10 <= hour <= 16:  # Daytime
                base = 20 + np.random.normal(0, 5)
            elif 22 <= hour or hour <= 5:  # Night
                base = 5 + np.random.normal(0, 2)
            else:
                base = 15 + np.random.normal(0, 4)
        else:  # Weekend
            if 10 <= hour <= 18:
                base = 18 + np.random.normal(0, 6)
            else:
                base = 8 + np.random.normal(0, 3)
        
        data.append({
            'timestamp': ts,
            'vehicle_count': max(0, int(base))
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the forecaster
    print("Generating sample data...")
    data = generate_sample_data(days=7)
    print(f"Generated {len(data)} hours of data")
    
    forecaster = TrafficForecaster()
    
    # Test forecast
    print("\nGenerating forecast...")
    forecast = forecaster.forecast(data)
    
    print("\nForecast Results:")
    for ts, pred, level in zip(forecast.timestamps, 
                               forecast.predictions, 
                               forecast.density_levels):
        print(f"  {ts.strftime('%Y-%m-%d %H:%M')}: {pred:.1f} vehicles ({level})")
