"""
Tests for Feature Engineering Module.

Tests add_technical_indicators and related functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    add_technical_indicators,
    validate_ohlc_data,
    add_custom_indicator
)


class TestValidateOhlcData:
    """Tests for validate_ohlc_data helper function."""

    def test_valid_data(self, sample_stock_data: pd.DataFrame):
        """Test validation with valid OHLCV data."""
        is_valid, missing = validate_ohlc_data(sample_stock_data)
        assert is_valid is True
        assert missing == []

    def test_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({'Close': [1, 2, 3]})

        is_valid, missing = validate_ohlc_data(df)

        assert is_valid is False
        assert 'Open' in missing
        assert 'High' in missing
        assert 'Low' in missing
        assert 'Volume' in missing

    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()

        is_valid, missing = validate_ohlc_data(df)

        assert is_valid is False


class TestAddTechnicalIndicators:
    """Tests for add_technical_indicators function."""

    def test_none_dataframe_raises_error(self):
        """Test that None DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="DataFrame cannot be None"):
            add_technical_indicators(None)

    def test_empty_dataframe_returns_as_is(self):
        """Test that empty DataFrame is returned unchanged."""
        df = pd.DataFrame()
        result = add_technical_indicators(df)
        assert result.empty

    def test_missing_close_column_raises_error(self):
        """Test that missing Close column raises ValueError."""
        df = pd.DataFrame({'Open': [1, 2, 3], 'High': [1, 2, 3]})

        with pytest.raises(ValueError, match="must have 'Close' column"):
            add_technical_indicators(df)

    def test_invalid_rsi_length(self, sample_stock_data: pd.DataFrame):
        """Test that invalid RSI length raises ValueError."""
        with pytest.raises(ValueError, match="rsi_length must be positive"):
            add_technical_indicators(sample_stock_data, rsi_length=0)

    def test_invalid_ma_length(self, sample_stock_data: pd.DataFrame):
        """Test that invalid MA length raises ValueError."""
        with pytest.raises(ValueError, match="ma20_length must be positive"):
            add_technical_indicators(sample_stock_data, ma20_length=-1)

    def test_adds_all_indicators(self, sample_stock_data: pd.DataFrame):
        """Test that all indicators are added to DataFrame."""
        result = add_technical_indicators(sample_stock_data)

        assert 'RSI' in result.columns
        assert 'MA20' in result.columns
        assert 'MA50' in result.columns

    def test_indicator_values_are_numeric(self, sample_stock_data: pd.DataFrame):
        """Test that indicator values are numeric."""
        result = add_technical_indicators(sample_stock_data)

        assert np.issubdtype(result['RSI'].dtype, np.number)
        assert np.issubdtype(result['MA20'].dtype, np.number)
        assert np.issubdtype(result['MA50'].dtype, np.number)

    def test_rsi_in_valid_range(self, sample_stock_data: pd.DataFrame):
        """Test that RSI values are in valid 0-100 range."""
        result = add_technical_indicators(sample_stock_data)

        # First 14 values may be NaN due to RSI calculation
        rsi_values = result['RSI'].dropna()

        if len(rsi_values) > 0:
            assert rsi_values.min() >= 0
            assert rsi_values.max() <= 100

    def test_custom_parameters(self, sample_stock_data: pd.DataFrame):
        """Test with custom indicator parameters."""
        result = add_technical_indicators(
            sample_stock_data,
            rsi_length=10,
            ma20_length=15,
            ma50_length=30
        )

        assert 'RSI' in result.columns
        assert 'MA20' in result.columns
        assert 'MA50' in result.columns


class TestAddCustomIndicator:
    """Tests for add_custom_indicator function."""

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        def dummy_func(df, **kwargs):
            return pd.Series([1, 2, 3])

        df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            add_custom_indicator(df, dummy_func, 'custom')

    def test_adds_custom_indicator(self, sample_stock_data: pd.DataFrame):
        """Test adding a custom indicator."""
        def custom_indicator(df, **kwargs):
            return df['Close'] * 2

        result = add_custom_indicator(sample_stock_data, custom_indicator, 'doubled_close')

        assert 'doubled_close' in result.columns
        pd.testing.assert_series_equal(
            result['doubled_close'],
            sample_stock_data['Close'] * 2
        )

    def test_indicator_error_propagates(self, sample_stock_data: pd.DataFrame):
        """Test that indicator function errors are propagated."""
        def failing_indicator(df, **kwargs):
            raise RuntimeError("Intentional failure")

        with pytest.raises(RuntimeError, match="Intentional failure"):
            add_custom_indicator(sample_stock_data, failing_indicator, 'fail')
