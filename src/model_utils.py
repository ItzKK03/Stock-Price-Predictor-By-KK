"""
Model Utilities Module.

Handles model building, loading, saving, and data preparation for LSTM.
Includes input validation and error handling.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Tuple, List, Optional, Any, Dict
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.logger import get_logger


logger = get_logger(__name__)


# --- Constants ---
MODEL_PATH: str = 'stock_predictor.keras'
SCALER_X_PATH: str = 'scaler_x.joblib'
SCALER_Y_PATH: str = 'scaler_y.joblib'
SCALER_FEATURES_PATH: str = 'scaler_features.json'


def validate_input_shape(input_shape: Tuple[int, int]) -> bool:
    """
    Validate that input shape is valid for LSTM model.

    Args:
        input_shape: Tuple of (time_steps, n_features).

    Returns:
        True if valid, False otherwise.
    """
    if not input_shape or len(input_shape) != 2:
        return False
    time_steps, n_features = input_shape
    return time_steps > 0 and n_features > 0


def build_model(
    input_shape: Tuple[int, int],
    lstm_units: int = 100,
    dropout_rate: float = 0.2,
    dense_units: int = 25,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Build and compile an LSTM model for time series prediction.

    Args:
        input_shape: Tuple of (time_steps, n_features) for LSTM input.
        lstm_units: Number of units in LSTM layers (default: 100).
        dropout_rate: Dropout rate for regularization (default: 0.2).
        dense_units: Number of units in intermediate dense layer (default: 25).
        learning_rate: Learning rate for Adam optimizer (default: 0.001).

    Returns:
        Compiled Keras Sequential model.

    Raises:
        ValueError: If input parameters are invalid.
    """
    # Input validation
    if not validate_input_shape(input_shape):
        logger.error(f"Invalid input_shape: {input_shape}")
        raise ValueError(
            "input_shape must be a tuple of (time_steps, n_features) "
            "where both values are positive"
        )

    if lstm_units <= 0:
        logger.error(f"Invalid lstm_units: {lstm_units}")
        raise ValueError("lstm_units must be positive")

    if not 0 <= dropout_rate <= 1:
        logger.error(f"Invalid dropout_rate: {dropout_rate}")
        raise ValueError("dropout_rate must be between 0 and 1")

    if dense_units <= 0:
        logger.error(f"Invalid dense_units: {dense_units}")
        raise ValueError("dense_units must be positive")

    if learning_rate <= 0:
        logger.error(f"Invalid learning_rate: {learning_rate}")
        raise ValueError("learning_rate must be positive")

    logger.info(
        f"Building LSTM model: input_shape={input_shape}, units={lstm_units}, "
        f"dropout={dropout_rate}, lr={learning_rate}"
    )

    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )

    logger.info(f"Model built successfully with input shape: {input_shape}")
    return model


def save_model(model: Sequential, path: str) -> None:
    """
    Save a Keras model to disk.

    Args:
        model: Keras Sequential model to save.
        path: File path to save the model.

    Raises:
        ValueError: If model is None or path is empty.
    """
    if model is None:
        logger.error("Cannot save None model")
        raise ValueError("Model cannot be None")

    if not path:
        logger.error("Cannot save model to empty path")
        raise ValueError("Path cannot be empty")

    logger.info(f"Saving model to {path}")
    model.save(path)
    logger.info("Model saved successfully")


def load_model(path: str) -> Sequential:
    """
    Load a Keras model from disk.

    Args:
        path: File path to load the model from.

    Returns:
        Loaded Keras Sequential model.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If path is empty.
    """
    if not path:
        logger.error("Cannot load model from empty path")
        raise ValueError("Path cannot be empty")

    if not os.path.exists(path):
        logger.error(f"Model file not found at {path}")
        raise FileNotFoundError(f"Model file not found at {path}")

    logger.info(f"Loading model from {path}")
    return keras_load_model(path)


def save_scaler(
    scaler: Any,
    scaler_path: str,
    features: List[str]
) -> None:
    """
    Save a scaler and its associated features to disk.

    Args:
        scaler: Fitted sklearn scaler object.
        scaler_path: File path to save the scaler (.joblib format).
        features: List of feature names used with this scaler.

    Raises:
        ValueError: If scaler is None or path is empty.
    """
    if scaler is None:
        logger.error("Cannot save None scaler")
        raise ValueError("Scaler cannot be None")

    if not scaler_path:
        logger.error("Cannot save scaler to empty path")
        raise ValueError("Scaler path cannot be empty")

    if not features or not isinstance(features, list):
        logger.error(f"Invalid features: {features}")
        raise ValueError("Features must be a non-empty list")

    logger.info(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)

    # Save features in a separate .json file
    features_path = scaler_path.replace('.joblib', '_features.json')
    with open(features_path, 'w') as f:
        json.dump(features, f)
    logger.info(f"Features saved to {features_path}")


def load_scaler(scaler_path: str) -> Tuple[Any, Optional[List[str]]]:
    """
    Load a scaler and its associated features from disk.

    Args:
        scaler_path: File path to load the scaler from.

    Returns:
        Tuple of (scaler, features) where features may be None if not found.

    Raises:
        FileNotFoundError: If scaler file doesn't exist.
        ValueError: If path is empty.
    """
    if not scaler_path:
        logger.error("Cannot load scaler from empty path")
        raise ValueError("Scaler path cannot be empty")

    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found at {scaler_path}")
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    logger.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Load associated features
    features_path = scaler_path.replace('.joblib', '_features.json')
    features: Optional[List[str]] = None
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = json.load(f)
        logger.debug(f"Loaded {len(features)} features")
    else:
        logger.warning(f"Features file not found at {features_path}")

    return scaler, features


def create_dataset(
    dataset_x: np.ndarray,
    dataset_y: np.ndarray,
    time_step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time-series data into samples for LSTM.

    Creates sliding windows of size time_step for sequence prediction.

    Args:
        dataset_x: Scaled feature data as numpy array.
        dataset_y: Scaled target data as numpy array.
        time_step: Number of time steps to use for each sample (default: 1).

    Returns:
        Tuple of (X, y) where:
            - X has shape (n_samples, time_step, n_features)
            - y has shape (n_samples,)

    Raises:
        ValueError: If inputs are invalid.
    """
    # Input validation
    if dataset_x is None or dataset_y is None:
        logger.error("dataset_x and dataset_y cannot be None")
        raise ValueError("dataset_x and dataset_y cannot be None")

    if not isinstance(dataset_x, np.ndarray) or not isinstance(dataset_y, np.ndarray):
        logger.error("dataset_x and dataset_y must be numpy arrays")
        raise ValueError("dataset_x and dataset_y must be numpy arrays")

    if time_step <= 0:
        logger.error(f"Invalid time_step: {time_step}")
        raise ValueError("time_step must be positive")

    if len(dataset_x) != len(dataset_y):
        logger.error(f"Dataset length mismatch: X={len(dataset_x)}, y={len(dataset_y)}")
        raise ValueError("dataset_x and dataset_y must have the same length")

    if len(dataset_x) <= time_step:
        logger.error(
            f"Dataset too small: {len(dataset_x)} samples, need more than time_step={time_step}"
        )
        raise ValueError(f"Dataset must have more than time_step ({time_step}) samples")

    logger.debug(f"Creating dataset with time_step={time_step}, samples={len(dataset_x)}")

    dataX: List[np.ndarray] = []
    dataY: List[float] = []

    for i in range(len(dataset_x) - time_step):
        a = dataset_x[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset_y[i + time_step, 0])

    result_x = np.array(dataX)
    result_y = np.array(dataY)

    logger.info(f"Created dataset with {len(result_x)} samples")
    return result_x, result_y
