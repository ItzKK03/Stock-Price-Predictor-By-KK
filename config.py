"""
Configuration Module.

Centralized configuration for the Stock Price Predictor application.
All settings can be overridden via environment variables using a .env file.

Usage:
    from config import settings
    ticker = settings.TICKER
    model_path = settings.model_path
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_env_str(key: str, default: str) -> str:
    """Get string environment variable with fallback to default."""
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable with fallback to default."""
    value = os.getenv(key, str(default))
    return int(value) if value else default


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable with fallback to default."""
    value = os.getenv(key, str(default))
    return float(value) if value else default


def _get_env_list(key: str, default: List[str]) -> List[str]:
    """Get list environment variable (comma-separated) with fallback to default."""
    value = os.getenv(key)
    if value:
        return [item.strip() for item in value.split(',')]
    return default


# Base directory
BASE_DIR: Path = Path(__file__).parent.absolute()


class DataConfig:
    """Configuration for data collection and processing."""

    # Stock settings
    TICKER: str = _get_env_str("TICKER", "AAPL")
    START_DATE: str = _get_env_str("START_DATE", "2015-01-01")
    END_DATE: str = _get_env_str("END_DATE", "2025-12-31")

    # Time series settings
    TIME_STEP: int = _get_env_int("TIME_STEP", 60)

    # Feature settings
    FEATURES: List[str] = _get_env_list(
        "FEATURES",
        ['Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'MA20', 'MA50', 'Sentiment']
    )
    TARGET_FEATURE: str = _get_env_str("TARGET_FEATURE", "Close")


class ModelConfig:
    """Configuration for model architecture."""

    # LSTM architecture
    LSTM_UNITS: int = _get_env_int("LSTM_UNITS", 100)
    DROPOUT_RATE: float = _get_env_float("DROPOUT_RATE", 0.2)
    DENSE_UNITS: int = _get_env_int("DENSE_UNITS", 25)
    LEARNING_RATE: float = _get_env_float("LEARNING_RATE", 0.001)

    # File paths
    MODEL_PATH: str = _get_env_str("MODEL_PATH", str(BASE_DIR / "stock_predictor.keras"))
    SCALER_X_PATH: str = _get_env_str("SCALER_X_PATH", str(BASE_DIR / "scaler_x.joblib"))
    SCALER_Y_PATH: str = _get_env_str("SCALER_Y_PATH", str(BASE_DIR / "scaler_y.joblib"))
    SCALER_FEATURES_PATH: str = _get_env_str(
        "SCALER_FEATURES_PATH",
        str(BASE_DIR / "scaler_features.json")
    )

    @property
    def model_path(self) -> str:
        """Get model path."""
        return self.MODEL_PATH

    @property
    def scaler_x_path(self) -> str:
        """Get X scaler path."""
        return self.SCALER_X_PATH

    @property
    def scaler_y_path(self) -> str:
        """Get Y scaler path."""
        return self.SCALER_Y_PATH

    @property
    def scaler_features_path(self) -> str:
        """Get features JSON path."""
        return self.SCALER_FEATURES_PATH


class TrainingConfig:
    """Configuration for model training."""

    # Training hyperparameters
    EPOCHS: int = _get_env_int("EPOCHS", 100)
    BATCH_SIZE: int = _get_env_int("BATCH_SIZE", 64)
    VALIDATION_SPLIT: float = _get_env_float("VALIDATION_SPLIT", 0.2)
    EARLY_STOPPING_PATIENCE: int = _get_env_int("EARLY_STOPPING_PATIENCE", 10)

    # Scaler settings
    SCALER_FEATURE_RANGE: tuple = (0, 1)


class AppConfig:
    """Configuration for the Streamlit web application."""

    # UI settings
    PAGE_TITLE: str = _get_env_str("PAGE_TITLE", "AI Stock Price Predictor")
    PAGE_ICON: str = _get_env_str("PAGE_ICON", "📈")
    LAYOUT: str = _get_env_str("LAYOUT", "wide")

    # Display settings
    HISTORY_DAYS: int = _get_env_int("HISTORY_DAYS", 90)
    DATA_BUFFER_DAYS: int = _get_env_int("DATA_BUFFER_DAYS", 100)

    # Sentiment model
    SENTIMENT_MODEL_NAME: str = _get_env_str("SENTIMENT_MODEL_NAME", "ProsusAI/finbert")


class Settings:
    """
    Main settings class that aggregates all configuration sections.

    Usage:
        from config import settings

        # Access data config
        ticker = settings.TICKER
        time_step = settings.TIME_STEP

        # Access model config
        model_path = settings.model_path
        lstm_units = settings.LSTM_UNITS

        # Access training config
        epochs = settings.EPOCHS
        batch_size = settings.BATCH_SIZE
    """

    # Data configuration
    TICKER: str = DataConfig.TICKER
    START_DATE: str = DataConfig.START_DATE
    END_DATE: str = DataConfig.END_DATE
    TIME_STEP: int = DataConfig.TIME_STEP
    FEATURES: List[str] = DataConfig.FEATURES
    TARGET_FEATURE: str = DataConfig.TARGET_FEATURE

    # Model configuration
    MODEL_PATH: str = ModelConfig.MODEL_PATH
    SCALER_X_PATH: str = ModelConfig.SCALER_X_PATH
    SCALER_Y_PATH: str = ModelConfig.SCALER_Y_PATH
    SCALER_FEATURES_PATH: str = ModelConfig.SCALER_FEATURES_PATH
    LSTM_UNITS: int = ModelConfig.LSTM_UNITS
    DROPOUT_RATE: float = ModelConfig.DROPOUT_RATE
    DENSE_UNITS: int = ModelConfig.DENSE_UNITS
    LEARNING_RATE: float = ModelConfig.LEARNING_RATE

    # Training configuration
    EPOCHS: int = TrainingConfig.EPOCHS
    BATCH_SIZE: int = TrainingConfig.BATCH_SIZE
    VALIDATION_SPLIT: float = TrainingConfig.VALIDATION_SPLIT
    EARLY_STOPPING_PATIENCE: int = TrainingConfig.EARLY_STOPPING_PATIENCE

    # App configuration
    PAGE_TITLE: str = AppConfig.PAGE_TITLE
    PAGE_ICON: str = AppConfig.PAGE_ICON
    LAYOUT: str = AppConfig.LAYOUT
    HISTORY_DAYS: int = AppConfig.HISTORY_DAYS
    DATA_BUFFER_DAYS: int = AppConfig.DATA_BUFFER_DAYS
    SENTIMENT_MODEL_NAME: str = AppConfig.SENTIMENT_MODEL_NAME

    # Convenience properties
    @property
    def model_path(self) -> str:
        """Get model path."""
        return self.MODEL_PATH

    @property
    def scaler_x_path(self) -> str:
        """Get X scaler path."""
        return self.SCALER_X_PATH

    @property
    def scaler_y_path(self) -> str:
        """Get Y scaler path."""
        return self.SCALER_Y_PATH

    @property
    def scaler_features_path(self) -> str:
        """Get features JSON path."""
        return self.SCALER_FEATURES_PATH


# Global settings instance for easy importing
settings: Settings = Settings()
