"""
Tests for Data Preprocessing Module.

Tests merge_stock_sentiment, prepare_features, and helper functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    merge_stock_sentiment,
    validate_features,
    prepare_features,
    fill_missing_values,
    create_time_index,
    validate_dataframe
)


class TestValidateDataFrame:
    """Tests for validate_dataframe helper function."""

    def test_none_dataframe_raises_error(self):
        """Test that None DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_dataframe(None, "test_df")

    def test_non_dataframe_raises_error(self):
        """Test that non-DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            validate_dataframe([1, 2, 3], "test_df")  # type: ignore

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_dataframe(df, "test_df")

    def test_valid_dataframe_passes(self, sample_stock_data: pd.DataFrame):
        """Test that valid DataFrame passes validation."""
        # Should not raise
        validate_dataframe(sample_stock_data, "test_df")


class TestMergeStockSentiment:
    """Tests for merge_stock_sentiment function."""

    def test_none_stock_data_raises_error(self):
        """Test that None stock_data raises ValueError."""
        with pytest.raises(ValueError, match="stock_data cannot be None"):
            merge_stock_sentiment(None, None)

    def test_empty_stock_data_raises_error(self):
        """Test that empty stock_data raises ValueError."""
        with pytest.raises(ValueError, match="stock_data cannot be empty"):
            merge_stock_sentiment(pd.DataFrame(), None)

    def test_non_dataframe_stock_data_raises_error(self):
        """Test that non-DataFrame stock_data raises ValueError."""
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            merge_stock_sentiment([1, 2, 3], None)  # type: ignore

    def test_merge_with_none_sentiment(self, sample_stock_data: pd.DataFrame):
        """Test merging with None sentiment adds default value."""
        result = merge_stock_sentiment(sample_stock_data, None, fill_sentiment=0.5)

        assert 'Sentiment' in result.columns
        assert result['Sentiment'].iloc[0] == 0.5

    def test_merge_with_empty_sentiment(self, sample_stock_data: pd.DataFrame):
        """Test merging with empty sentiment DataFrame."""
        empty_sentiment = pd.DataFrame(columns=['Sentiment'])
        result = merge_stock_sentiment(sample_stock_data, empty_sentiment)

        assert 'Sentiment' in result.columns

    def test_merge_successful(self, sample_stock_data: pd.DataFrame, sample_sentiment_data: pd.DataFrame):
        """Test successful merge of stock and sentiment data."""
        # Make sure dates overlap
        sentiment = sample_sentiment_data.iloc[:50]

        result = merge_stock_sentiment(sample_stock_data, sentiment)

        assert 'Sentiment' in result.columns
        assert len(result) > 0
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_merge_drops_nan(self, sample_stock_data: pd.DataFrame, sample_sentiment_data: pd.DataFrame):
        """Test that merge drops NaN values."""
        sentiment = sample_sentiment_data.iloc[:50]

        initial_count = len(sample_stock_data)
        result = merge_stock_sentiment(sample_stock_data, sentiment)

        # Some rows may be dropped due to NaN
        assert len(result) <= initial_count


class TestValidateFeatures:
    """Tests for validate_features function."""

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be empty"):
            validate_features(pd.DataFrame(), ['feature1'])

    def test_empty_features_raises_error(self, sample_stock_data: pd.DataFrame):
        """Test that empty features list raises ValueError."""
        with pytest.raises(ValueError, match="required_features cannot be empty"):
            validate_features(sample_stock_data, [])

    def test_non_list_features_raises_error(self, sample_stock_data: pd.DataFrame):
        """Test that non-list features raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            validate_features(sample_stock_data, "feature1")  # type: ignore

    def test_all_features_present(self, sample_stock_data: pd.DataFrame):
        """Test validation when all features are present."""
        is_valid, missing = validate_features(sample_stock_data, ['Close', 'Open', 'High'])

        assert is_valid is True
        assert missing == []

    def test_missing_features(self, sample_stock_data: pd.DataFrame):
        """Test validation when features are missing."""
        is_valid, missing = validate_features(sample_stock_data, ['Close', 'NonExistent'])

        assert is_valid is False
        assert 'NonExistent' in missing


class TestPrepareFeatures:
    """Tests for prepare_features function."""

    def test_missing_features_raises_error(self, sample_stock_data: pd.DataFrame):
        """Test that missing features raise ValueError."""
        with pytest.raises(ValueError, match="Missing features"):
            prepare_features(sample_stock_data, ['Close', 'NonExistent'], 'Close')

    def test_missing_target_raises_error(self, sample_stock_data: pd.DataFrame):
        """Test that missing target raises ValueError."""
        with pytest.raises(ValueError, match="not found in data"):
            prepare_features(sample_stock_data, ['Close'], 'NonExistent')

    def test_returns_correct_structure(self, sample_stock_data: pd.DataFrame):
        """Test that function returns correct structure."""
        X, y, features = prepare_features(sample_stock_data, ['Close', 'Volume'], 'Close')

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert isinstance(features, list)

    def test_feature_columns_correct(self, sample_stock_data: pd.DataFrame):
        """Test that X has correct feature columns."""
        X, y, features = prepare_features(sample_stock_data, ['Close', 'Volume'], 'Close')

        assert list(X.columns) == ['Close', 'Volume']

    def test_target_column_correct(self, sample_stock_data: pd.DataFrame):
        """Test that y has correct target column."""
        X, y, features = prepare_features(sample_stock_data, ['Close'], 'Close')

        assert list(y.columns) == ['Close']


class TestFillMissingValues:
    """Tests for fill_missing_values function."""

    def test_none_data_raises_error(self):
        """Test that None data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be None"):
            fill_missing_values(None)

    def test_empty_dataframe_returns_as_is(self):
        """Test that empty DataFrame is returned unchanged."""
        df = pd.DataFrame()
        result = fill_missing_values(df)
        assert result.empty

    def test_invalid_method_uses_default(self, sample_stock_data: pd.DataFrame):
        """Test that invalid method uses default ffill_bfill."""
        # Add some NaN values
        df = sample_stock_data.copy()
        df.iloc[0, 0] = np.nan

        result = fill_missing_values(df, fill_method='invalid_method')

        # Should not have NaN in first cell after filling
        assert not pd.isna(result.iloc[0, 0])

    def test_ffill_method(self):
        """Test forward fill method."""
        df = pd.DataFrame({'value': [1, np.nan, np.nan, 4, np.nan]})

        result = fill_missing_values(df, fill_method='ffill')

        assert result['value'].iloc[1] == 1
        assert result['value'].iloc[2] == 1
        assert result['value'].iloc[3] == 4

    def test_bfill_method(self):
        """Test backward fill method."""
        df = pd.DataFrame({'value': [np.nan, 1, np.nan, np.nan, 4]})

        result = fill_missing_values(df, fill_method='bfill')

        assert result['value'].iloc[0] == 1
        assert result['value'].iloc[2] == 4

    def test_drop_method(self):
        """Test drop method."""
        df = pd.DataFrame({'value': [1, np.nan, 3, np.nan, 5]})

        result = fill_missing_values(df, fill_method='drop')

        assert len(result) == 3
        assert not result.isna().any().any()


class TestCreateTimeIndex:
    """Tests for create_time_index function."""

    def test_none_data_raises_error(self):
        """Test that None data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be None"):
            create_time_index(None)

    def test_already_datetimeindex(self, sample_stock_data: pd.DataFrame):
        """Test DataFrame already with DatetimeIndex."""
        result = create_time_index(sample_stock_data)

        assert isinstance(result.index, pd.DatetimeIndex)

    def test_creates_index_from_column(self):
        """Test creating index from date column."""
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Close': [1, 2, 3]
        })

        result = create_time_index(df, date_column='Date')

        assert isinstance(result.index, pd.DatetimeIndex)
        assert 'Date' not in result.columns  # Column should be dropped

    def test_missing_date_column_uses_index(self):
        """Test that missing date column uses existing index."""
        df = pd.DataFrame({'Close': [1, 2, 3]})
        df.index = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])

        result = create_time_index(df, date_column='NonExistent')

        assert isinstance(result.index, pd.DatetimeIndex)
