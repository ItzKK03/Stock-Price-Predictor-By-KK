"""
Pytest Configuration and Shared Fixtures.

This file defines fixtures that are shared across multiple test modules.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Any


@pytest.fixture
def sample_stock_data() -> pd.DataFrame:
    """
    Create sample stock data for testing.

    Returns:
        DataFrame with OHLCV data for 100 days.
    """
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Generate realistic-looking stock data
    base_price = 150.0
    returns = np.random.randn(100) * 0.02  # 2% daily volatility
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(100) * 0.001),
        'High': prices * (1 + np.abs(np.random.randn(100) * 0.01)),
        'Low': prices * (1 - np.abs(np.random.randn(100) * 0.01)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    df.index.name = 'Date'
    return df


@pytest.fixture
def sample_data_with_indicators(sample_stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample stock data with technical indicators.

    Returns:
        DataFrame with OHLCV + RSI, MA20, MA50.
    """
    df = sample_stock_data.copy()

    # Simple RSI calculation (simplified for testing)
    df['RSI'] = 50 + np.random.randn(100) * 10
    df['RSI'] = df['RSI'].clip(0, 100)

    # Moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    return df


@pytest.fixture
def sample_sentiment_data() -> pd.DataFrame:
    """
    Create sample sentiment data for testing.

    Returns:
        DataFrame with Date index and Sentiment column.
    """
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    sentiment = np.random.randn(50)

    df = pd.DataFrame({'Sentiment': sentiment}, index=dates)
    df.index.name = 'Date'
    return df


@pytest.fixture
def sample_scaled_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample scaled data for LSTM testing.

    Returns:
        Tuple of (X_scaled, y_scaled) as numpy arrays.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, 1)

    return X, y


@pytest.fixture
def sample_time_series_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample time-series data in LSTM format.

    Returns:
        Tuple of (X, y) where X has shape (samples, time_steps, features).
    """
    np.random.seed(42)
    samples = 50
    time_steps = 10
    features = 5

    X = np.random.rand(samples, time_steps, features)
    y = np.random.rand(samples, 1)

    return X, y


@pytest.fixture
def sample_predictions() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample actual and predicted values for evaluation testing.

    Returns:
        Tuple of (y_true, y_pred).
    """
    np.random.seed(42)
    n = 100

    y_true = 100 + np.cumsum(np.random.randn(n))
    y_pred = y_true + np.random.randn(n) * 5

    return y_true, y_pred


@pytest.fixture
def mock_news_data() -> list:
    """
    Create mock news data for testing.

    Returns:
        List of news article dictionaries.
    """
    return [
        {
            'title': 'Apple Reports Record Q4 Earnings',
            'publisher': 'Financial Times',
            'link': 'https://example.com/news1',
            'providerPublishTime': 1704067200,
            'type': 'STORY',
            'thumbnail': {'resolutions': []},
            'relatedTickers': ['AAPL']
        },
        {
            'title': 'Tech Stocks Rally on Positive Outlook',
            'publisher': 'Reuters',
            'link': 'https://example.com/news2',
            'providerPublishTime': 1704153600,
            'type': 'STORY',
            'thumbnail': {'resolutions': []},
            'relatedTickers': ['AAPL', 'GOOGL']
        },
        {
            'title': 'Market Analysis: AAPL Price Target Raised',
            'publisher': 'Bloomberg',
            'link': 'https://example.com/news3',
            'providerPublishTime': 1704240000,
            'type': 'STORY',
            'thumbnail': {'resolutions': []},
            'relatedTickers': ['AAPL']
        }
    ]


@pytest.fixture
def temp_model_path(tmp_path: Any) -> str:
    """
    Create a temporary path for model files.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        String path to temporary file.
    """
    return str(tmp_path / "test_model.keras")


@pytest.fixture
def temp_scaler_path(tmp_path: Any) -> str:
    """
    Create a temporary path for scaler files.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        String path to temporary file.
    """
    return str(tmp_path / "test_scaler.joblib")
