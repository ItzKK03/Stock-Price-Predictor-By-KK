"""
Tests for Data Collection Module.

Tests get_stock_data and get_stock_news functions.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_collection import get_stock_data, get_stock_news, _retry_with_backoff


class TestRetryWithBackoff:
    """Tests for the retry with backoff helper function."""

    def test_successful_first_attempt(self):
        """Test function that succeeds on first attempt."""
        result = _retry_with_backoff(lambda: 42, max_retries=3)
        assert result == 42

    def test_successful_after_retry(self):
        """Test function that succeeds after one failure."""
        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = _retry_with_backoff(flaky_func, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count[0] == 2

    def test_all_retries_fail(self):
        """Test function that fails on all attempts."""
        def always_fail():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            _retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

    def test_jitter_applied(self):
        """Test that jitter is applied to delay."""
        # This test just verifies the function accepts jitter parameter
        result = _retry_with_backoff(lambda: 1, jitter=True)
        assert result == 1


class TestGetStockData:
    """Tests for get_stock_data function."""

    def test_invalid_ticker_empty(self):
        """Test with empty ticker string."""
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            get_stock_data("", "2024-01-01", "2024-01-31")

    def test_invalid_ticker_none(self):
        """Test with None as ticker."""
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            get_stock_data(None, "2024-01-01", "2024-01-31")  # type: ignore

    def test_invalid_start_date(self):
        """Test with empty start date."""
        with pytest.raises(ValueError, match="Start date must be a non-empty string"):
            get_stock_data("AAPL", "", "2024-01-31")

    def test_invalid_end_date(self):
        """Test with empty end date."""
        with pytest.raises(ValueError, match="End date must be a non-empty string"):
            get_stock_data("AAPL", "2024-01-01", "")

    @patch('src.data_collection.yf.download')
    def test_successful_fetch(self, mock_download: MagicMock, sample_stock_data: pd.DataFrame):
        """Test successful data fetch."""
        mock_download.return_value = sample_stock_data

        result = get_stock_data("AAPL", "2024-01-01", "2024-01-31")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert 'Close' in result.columns
        mock_download.assert_called_once()

    @patch('src.data_collection.yf.download')
    def test_empty_data_raises_error(self, mock_download: MagicMock):
        """Test that empty data raises ValueError."""
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No data returned"):
            get_stock_data("INVALID", "2024-01-01", "2024-01-31")

    @patch('src.data_collection.yf.download')
    def test_multiindex_columns_flattened(self, mock_download: MagicMock, sample_stock_data: pd.DataFrame):
        """Test that MultiIndex columns are flattened."""
        # Create MultiIndex columns
        multi_columns = pd.MultiIndex.from_tuples([
            ('Close', 'AAPL'), ('Open', 'AAPL'), ('High', 'AAPL')
        ])
        sample_stock_data.columns = multi_columns[:3]
        mock_download.return_value = sample_stock_data

        result = get_stock_data("AAPL", "2024-01-01", "2024-01-31")

        assert not isinstance(result.columns, pd.MultiIndex)
        assert 'Close' in result.columns


class TestGetStockNews:
    """Tests for get_stock_news function."""

    def test_invalid_ticker(self):
        """Test with invalid ticker."""
        result = get_stock_news("")
        assert result == []

        result = get_stock_news(None)  # type: ignore
        assert result == []

    @patch('src.data_collection.yf.Ticker')
    def test_successful_fetch(self, mock_ticker: MagicMock, mock_news_data: list):
        """Test successful news fetch."""
        mock_stock = MagicMock()
        mock_stock.news = mock_news_data
        mock_ticker.return_value = mock_stock

        result = get_stock_news("AAPL")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]['title'] == 'Apple Reports Record Q4 Earnings'

    @patch('src.data_collection.yf.Ticker')
    def test_no_news(self, mock_ticker: MagicMock):
        """Test when no news is available."""
        mock_stock = MagicMock()
        mock_stock.news = []
        mock_ticker.return_value = mock_stock

        result = get_stock_news("AAPL")

        assert result == []

    @patch('src.data_collection.yf.Ticker')
    def test_api_error_returns_empty_list(self, mock_ticker: MagicMock):
        """Test that API errors return empty list instead of raising."""
        mock_stock = MagicMock()
        mock_stock.news = None
        mock_ticker.return_value = mock_stock

        result = get_stock_news("AAPL")

        assert result == []
