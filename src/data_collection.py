"""
Data Collection Module.

Handles fetching stock data and news from Yahoo Finance API.
Includes retry logic with exponential backoff for robustness.
"""

import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional
from src.logger import get_logger
import time
import random


logger = get_logger(__name__)


def _retry_with_backoff(
    func: callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: bool = True
) -> Any:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute (should be callable with no args).
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Base delay in seconds between retries (default: 1.0).
        max_delay: Maximum delay in seconds (default: 10.0).
        jitter: Whether to add random jitter to delay (default: True).

    Returns:
        Result of the function call.

    Raises:
        Exception: If all retries fail, raises the last exception.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)

                # Add jitter to prevent thundering herd
                if jitter:
                    delay = delay * (0.5 + random.random())

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


def get_stock_data(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Uses retry logic with exponential backoff to handle transient failures.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.
        max_retries: Maximum retry attempts for API failures (default: 3).

    Returns:
        DataFrame with OHLCV data indexed by date.

    Raises:
        ValueError: If ticker is empty or no data is found.
        Exception: If all retry attempts fail.
    """
    # Input validation
    if not ticker or not isinstance(ticker, str):
        logger.error(f"Invalid ticker: {ticker}")
        raise ValueError("Ticker must be a non-empty string")

    if not start or not isinstance(start, str):
        logger.error(f"Invalid start date: {start}")
        raise ValueError("Start date must be a non-empty string")

    if not end or not isinstance(end, str):
        logger.error(f"Invalid end date: {end}")
        raise ValueError("End date must be a non-empty string")

    logger.info(f"Fetching data for {ticker} from {start} to {end}...")

    def _fetch() -> pd.DataFrame:
        """Inner function for retry wrapper."""
        data = yf.download(ticker, start=start, end=end, group_by='column', progress=False)

        # --- FIX for MultiIndex Columns ---
        if isinstance(data.columns, pd.MultiIndex):
            logger.debug("Flattening MultiIndex columns...")
            data.columns = data.columns.get_level_values(0)
        # --- END FIX ---

        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        return data

    try:
        data = _retry_with_backoff(_fetch, max_retries=max_retries)
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker} after retries: {e}")
        raise

    # Ensure index is datetime
    data.index = pd.to_datetime(data.index)

    logger.info(f"Fetched {len(data)} rows of data for {ticker}")
    return data


def get_stock_news(
    ticker: str,
    max_retries: int = 2
) -> List[Dict[str, Any]]:
    """
    Fetch recent news headlines for a stock.

    Uses retry logic to handle transient API failures.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
        max_retries: Maximum retry attempts (default: 2).

    Returns:
        List of news articles as dictionaries.
        Returns empty list if no news found or on error.
    """
    # Input validation
    if not ticker or not isinstance(ticker, str):
        logger.error(f"Invalid ticker: {ticker}")
        return []

    logger.debug(f"Fetching news for {ticker}...")

    def _fetch() -> List[Dict[str, Any]]:
        """Inner function for retry wrapper."""
        stock = yf.Ticker(ticker)
        news = stock.news
        return news if news else []

    try:
        news = _retry_with_backoff(_fetch, max_retries=max_retries)
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return []

    if not news:
        logger.warning(f"No news found for {ticker}")
        return []

    logger.debug(f"Fetched {len(news)} news articles for {ticker}")
    return news
