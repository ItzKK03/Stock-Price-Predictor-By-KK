"""
Feature Engineering Module.

Handles calculation of technical indicators for stock data.
Includes input validation and error handling.
"""

import pandas as pd
import pandas_ta as ta
from typing import Optional
from src.logger import get_logger


logger = get_logger(__name__)


def validate_ohlc_data(df: pd.DataFrame) -> tuple:
    """
    Validate that DataFrame has required OHLCV columns.

    Args:
        df: DataFrame to validate.

    Returns:
        Tuple of (is_valid, missing_columns).
    """
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    return len(missing) == 0, missing


def add_technical_indicators(
    df: pd.DataFrame,
    rsi_length: int = 14,
    ma20_length: int = 20,
    ma50_length: int = 50,
    skip_validation: bool = False
) -> pd.DataFrame:
    """
    Add technical indicators to a DataFrame.

    Calculates RSI, 20-day SMA, and 50-day SMA using the pandas_ta library.

    Args:
        df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume).
        rsi_length: Period for RSI calculation (default: 14).
        ma20_length: Period for short-term moving average (default: 20).
        ma50_length: Period for long-term moving average (default: 50).
        skip_validation: Skip OHLCV validation (default: False).

    Returns:
        DataFrame with added columns: 'RSI', 'MA20', 'MA50'.

    Raises:
        ValueError: If required columns are missing or df is empty.
    """
    # Input validation
    if df is None:
        logger.error("DataFrame is None")
        raise ValueError("DataFrame cannot be None")

    if df.empty:
        logger.warning("Empty DataFrame provided, returning as-is")
        return df

    # Validate parameters
    if rsi_length <= 0:
        logger.error(f"Invalid rsi_length: {rsi_length}")
        raise ValueError("rsi_length must be positive")

    if ma20_length <= 0:
        logger.error(f"Invalid ma20_length: {ma20_length}")
        raise ValueError("ma20_length must be positive")

    if ma50_length <= 0:
        logger.error(f"Invalid ma50_length: {ma50_length}")
        raise ValueError("ma50_length must be positive")

    # Validate required columns (unless skipped)
    if not skip_validation:
        # At minimum we need Close for all indicators
        if 'Close' not in df.columns:
            logger.error("Missing required 'Close' column")
            raise ValueError("DataFrame must have 'Close' column")

    logger.debug(
        f"Adding technical indicators: RSI({rsi_length}), "
        f"SMA({ma20_length}), SMA({ma50_length})"
    )

    df = df.copy()

    try:
        # Calculate RSI
        df.ta.rsi(close=df['Close'], length=rsi_length, append=True)

        # Calculate 20-day Moving Average
        df.ta.sma(close=df['Close'], length=ma20_length, append=True)

        # Calculate 50-day Moving Average
        df.ta.sma(close=df['Close'], length=ma50_length, append=True)

        # Rename columns for clarity
        df.rename(columns={
            'RSI_14': 'RSI',
            'SMA_20': 'MA20',
            'SMA_50': 'MA50'
        }, inplace=True)

        # Check if indicators were calculated successfully
        indicators_added = ['RSI', 'MA20', 'MA50']
        missing_indicators = [col for col in indicators_added if col not in df.columns]

        if missing_indicators:
            logger.warning(f"Some indicators could not be calculated: {missing_indicators}")

        logger.debug("Technical indicators added successfully")
        return df

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        raise


def add_custom_indicator(
    df: pd.DataFrame,
    indicator_func: callable,
    column_name: str,
    **kwargs
) -> pd.DataFrame:
    """
    Add a custom technical indicator to the DataFrame.

    Args:
        df: DataFrame with price data.
        indicator_func: Function that calculates the indicator.
        column_name: Name for the new column.
        **kwargs: Additional arguments passed to indicator_func.

    Returns:
        DataFrame with the new indicator column.

    Raises:
        ValueError: If indicator calculation fails.
    """
    if df is None or df.empty:
        logger.error("Cannot add indicator to empty DataFrame")
        raise ValueError("DataFrame cannot be empty")

    logger.debug(f"Adding custom indicator: {column_name}")

    df = df.copy()

    try:
        result = indicator_func(df, **kwargs)
        df[column_name] = result
        logger.debug(f"Custom indicator '{column_name}' added successfully")
        return df
    except Exception as e:
        logger.error(f"Error adding custom indicator '{column_name}': {e}")
        raise
