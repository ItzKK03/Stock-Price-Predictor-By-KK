"""
Data Preprocessing Module.

Handles merging, cleaning, and preparing stock data with sentiment analysis.
This module consolidates duplicated preprocessing logic from training and inference.
Includes comprehensive input validation and error handling.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
from src.logger import get_logger


logger = get_logger(__name__)


def validate_dataframe(df: Any, name: str = "DataFrame") -> None:
    """
    Validate that input is a non-empty pandas DataFrame.

    Args:
        df: Object to validate.
        name: Name of the DataFrame for error messages.

    Raises:
        ValueError: If df is not a DataFrame or is empty.
    """
    if df is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(df)}")

    if df.empty:
        raise ValueError(f"{name} cannot be empty")


def merge_stock_sentiment(
    stock_data: pd.DataFrame,
    daily_sentiment: Optional[pd.DataFrame],
    fill_sentiment: float = 0.0
) -> pd.DataFrame:
    """
    Merge stock data with daily sentiment data.

    Handles index alignment, date column creation, and NaN filling.

    Args:
        stock_data: DataFrame with OHLCV data and technical indicators.
        daily_sentiment: DataFrame with Date index and 'Sentiment' column.
                         Can be None, in which case stock_data is returned
                         with Sentiment column filled with fill_sentiment.
        fill_sentiment: Value to fill missing sentiment (default: 0.0 = neutral).

    Returns:
        Merged DataFrame with Date as index and all features.

    Raises:
        ValueError: If stock_data is None, empty, or not a DataFrame.
    """
    # Validate stock_data
    validate_dataframe(stock_data, "stock_data")
    logger.debug(f"Merging stock data ({len(stock_data)} rows)")

    # Handle None or empty sentiment
    if daily_sentiment is None or daily_sentiment.empty:
        logger.warning("No sentiment data provided, proceeding with stock data only")
        result = stock_data.copy()
        result['Sentiment'] = fill_sentiment
        return result

    # Validate daily_sentiment
    if not isinstance(daily_sentiment, pd.DataFrame):
        logger.error(f"daily_sentiment must be DataFrame, got {type(daily_sentiment)}")
        raise ValueError("daily_sentiment must be a pandas DataFrame")

    stock_data = stock_data.copy()

    # Ensure 'Date' is a column for merging
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)

    stock_data['Date'] = stock_data.index.date

    # Handle sentiment DataFrame date column
    daily_sentiment = daily_sentiment.copy()
    if isinstance(daily_sentiment.index, pd.DatetimeIndex):
        daily_sentiment['Date'] = daily_sentiment.index.date
    else:
        logger.warning("Sentiment index is not DatetimeIndex, using NaT")
        daily_sentiment['Date'] = pd.NaT

    # Reset index to avoid merge ambiguity
    stock_data = stock_data.reset_index(drop=True)
    daily_sentiment = daily_sentiment.reset_index(drop=True)

    # Convert Date columns to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

    # Merge data (left join to keep all stock data)
    data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')

    # Set 'Date' as index for time-series processing
    data = data.set_index('Date')

    # Fill NaN sentiment with default value (neutral)
    data['Sentiment'] = data['Sentiment'].fillna(fill_sentiment)

    # Drop any other NaNs (e.g., from indicators at the start)
    initial_count = len(data)
    data = data.dropna()
    dropped_count = initial_count - len(data)

    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} rows with NaN values after merge")

    if data.empty:
        logger.error("No data available after merging and cleaning")
        raise ValueError("No data available after merging and cleaning")

    logger.debug(f"Merged data shape: {data.shape}")
    return data


def validate_features(
    data: pd.DataFrame,
    required_features: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that all required features are present in the data.

    Args:
        data: DataFrame to validate.
        required_features: List of required column names.

    Returns:
        Tuple of (is_valid, missing_features).
        - is_valid: True if all features present, False otherwise.
        - missing_features: List of features not found in data.

    Raises:
        ValueError: If data is empty or required_features is empty.
    """
    if data is None or data.empty:
        raise ValueError("data cannot be empty")

    if not required_features:
        raise ValueError("required_features cannot be empty")

    if not isinstance(required_features, list):
        raise ValueError("required_features must be a list")

    missing = [f for f in required_features if f not in data.columns]
    is_valid = len(missing) == 0

    if not is_valid:
        logger.error(f"Missing features: {missing}")

    return is_valid, missing


def prepare_features(
    data: pd.DataFrame,
    features: List[str],
    target_feature: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare feature matrices X and y from the data.

    Args:
        data: DataFrame with all features.
        features: List of feature column names for X.
        target_feature: Target column name for y.

    Returns:
        Tuple of (X, y, features) where:
            - X is DataFrame with feature columns
            - y is DataFrame with target column
            - features is the list of feature names

    Raises:
        ValueError: If required features are missing or inputs are invalid.
    """
    # Input validation
    validate_dataframe(data, "data")

    if not features:
        raise ValueError("features cannot be empty")

    if not isinstance(features, list):
        raise ValueError("features must be a list")

    if not target_feature:
        raise ValueError("target_feature cannot be empty")

    logger.debug(f"Preparing features: {features}")

    # Validate features
    is_valid, missing = validate_features(data, features)
    if not is_valid:
        raise ValueError(f"Missing features: {missing}")

    # Validate target
    if target_feature not in data.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in data")

    # Create feature matrices
    X = data[features]
    y = data[[target_feature]]

    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y, features


def fill_missing_values(
    data: pd.DataFrame,
    fill_method: str = 'ffill_bfill'
) -> pd.DataFrame:
    """
    Fill missing values in the data.

    Args:
        data: DataFrame with potentially missing values.
        fill_method: Method to fill NaN values.
            - 'ffill_bfill': Forward fill then backward fill
            - 'ffill': Forward fill only
            - 'bfill': Backward fill only
            - 'drop': Drop rows with NaN

    Returns:
        DataFrame with missing values filled.

    Raises:
        ValueError: If data is None or fill_method is invalid.
    """
    if data is None:
        raise ValueError("data cannot be None")

    if data.empty:
        return data

    valid_methods = ['ffill_bfill', 'ffill', 'bfill', 'drop']
    if fill_method not in valid_methods:
        logger.warning(f"Unknown fill method '{fill_method}', using ffill_bfill")
        fill_method = 'ffill_bfill'

    initial_nan_count = data.isna().sum().sum()

    if fill_method == 'ffill_bfill':
        data = data.ffill().bfill()
    elif fill_method == 'ffill':
        data = data.ffill()
    elif fill_method == 'bfill':
        data = data.bfill()
    elif fill_method == 'drop':
        data = data.dropna()

    remaining_nan_count = data.isna().sum().sum()

    if remaining_nan_count > 0:
        logger.warning(f"Still have {remaining_nan_count} NaN values after filling")

    filled_count = initial_nan_count - remaining_nan_count
    if filled_count > 0:
        logger.info(f"Filled {filled_count} NaN values using '{fill_method}'")

    return data


def create_time_index(
    data: pd.DataFrame,
    date_column: str = 'Date'
) -> pd.DataFrame:
    """
    Ensure data has a proper DatetimeIndex.

    Args:
        data: DataFrame with date column or index.
        date_column: Name of date column if not already index.

    Returns:
        DataFrame with DatetimeIndex.

    Raises:
        ValueError: If data is None.
    """
    if data is None:
        raise ValueError("data cannot be None")

    data = data.copy()

    if isinstance(data.index, pd.DatetimeIndex):
        logger.debug("Index is already DatetimeIndex")
        return data

    if date_column in data.columns:
        data.index = pd.to_datetime(data[date_column])
        data = data.drop(columns=[date_column])
        logger.debug(f"Created DatetimeIndex from column '{date_column}'")
    else:
        logger.warning(f"Date column '{date_column}' not found, using existing index")
        data.index = pd.to_datetime(data.index)

    return data
