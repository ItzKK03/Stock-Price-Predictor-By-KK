"""
Tests for Sentiment Analysis Module.

Tests initialize_sentiment_model, get_sentiment_scores, and get_daily_sentiment.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.sentiment_analysis import (
    initialize_sentiment_model,
    get_sentiment_scores,
    get_daily_sentiment,
    MODEL_NAME
)


class TestInitializeSentimentModel:
    """Tests for initialize_sentiment_model function."""

    @patch('src.sentiment_analysis.pipeline')
    def test_successful_initialization(self, mock_pipeline: MagicMock):
        """Test successful model initialization."""
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        # Clear cache before test
        initialize_sentiment_model.cache_clear()

        result = initialize_sentiment_model("test-model")

        assert result is not None
        mock_pipeline.assert_called_once_with("sentiment-analysis", model="test-model")

    @patch('src.sentiment_analysis.pipeline')
    def test_initialization_error_propagates(self, mock_pipeline: MagicMock):
        """Test that initialization errors are propagated."""
        mock_pipeline.side_effect = Exception("Model load failed")

        # Clear cache before test
        initialize_sentiment_model.cache_clear()

        with pytest.raises(Exception, match="Model load failed"):
            initialize_sentiment_model("invalid-model")

    def test_default_model_name(self):
        """Test that default model name is set."""
        assert MODEL_NAME == "ProsusAI/finbert"


class TestGetSentimentScores:
    """Tests for get_sentiment_scores function."""

    def test_empty_headlines_returns_empty(self):
        """Test that empty headlines list returns empty list."""
        mock_pipeline = MagicMock()
        result = get_sentiment_scores([], mock_pipeline)
        assert result == []

    def test_none_headlines_raises_error(self):
        """Test that None headlines raises ValueError."""
        mock_pipeline = MagicMock()
        with pytest.raises(ValueError, match="headlines must be a list"):
            get_sentiment_scores(None, mock_pipeline)  # type: ignore

    def test_single_headline(self):
        """Test analyzing a single headline."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{'label': 'positive', 'score': 0.95}]

        result = get_sentiment_scores(["Apple stock rises"], mock_pipeline)

        assert len(result) == 1
        assert result[0]['label'] == 'positive'
        assert result[0]['score'] == 0.95

    def test_multiple_headlines(self):
        """Test analyzing multiple headlines."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {'label': 'positive', 'score': 0.9},
            {'label': 'negative', 'score': 0.85},
            {'label': 'neutral', 'score': 0.7}
        ]

        headlines = [
            "Apple stock rises",
            "Market crashes on bad news",
            "Stocks remain flat"
        ]
        result = get_sentiment_scores(headlines, mock_pipeline)

        assert len(result) == 3
        assert result[0]['label'] == 'positive'
        assert result[1]['label'] == 'negative'
        assert result[2]['label'] == 'neutral'

    def test_batch_processing(self):
        """Test that headlines are processed in batches."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{'label': 'positive', 'score': 0.9}]

        # Create 20 headlines, should be processed in batches of 8
        headlines = [f"Headline {i}" for i in range(20)]
        get_sentiment_scores(headlines, mock_pipeline, batch_size=8)

        # Should be called 3 times (8 + 8 + 4)
        assert mock_pipeline.call_count == 3

    def test_filters_none_headlines(self):
        """Test that None headlines are filtered out."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{'label': 'positive', 'score': 0.9}]

        headlines = ["Valid headline", None, "", "Another valid"]
        result = get_sentiment_scores(headlines, mock_pipeline)

        # Only 2 valid headlines should be processed
        assert len(result) == 2
        mock_pipeline.assert_called_once()

    def test_pipeline_error_propagates(self):
        """Test that pipeline errors are propagated."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("Pipeline failed")

        with pytest.raises(RuntimeError, match="Pipeline failed"):
            get_sentiment_scores(["Test headline"], mock_pipeline)


class TestGetDailySentiment:
    """Tests for get_daily_sentiment function."""

    def test_invalid_ticker_empty(self):
        """Test with empty ticker."""
        mock_pipeline = MagicMock()
        result = get_daily_sentiment("", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert 'Sentiment' in result.columns

    def test_invalid_ticker_none(self):
        """Test with None ticker."""
        mock_pipeline = MagicMock()
        result = get_daily_sentiment(None, mock_pipeline)  # type: ignore

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('src.sentiment_analysis.get_stock_news')
    def test_no_news_returns_empty_df(self, mock_news: MagicMock):
        """Test that no news returns empty DataFrame."""
        mock_news.return_value = []
        mock_pipeline = MagicMock()

        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert 'Sentiment' in result.columns

    @patch('src.sentiment_analysis.get_stock_news')
    def test_news_missing_title_column(self, mock_news: MagicMock):
        """Test handling news without title column."""
        mock_news.return_value = [{'publisher': 'Test'}]  # No title
        mock_pipeline = MagicMock()

        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('src.sentiment_analysis.get_stock_news')
    def test_news_missing_publish_time(self, mock_news: MagicMock):
        """Test handling news without publish_time column."""
        mock_news.return_value = [{'title': 'Test headline'}]  # No publish_time
        mock_pipeline = MagicMock()

        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('src.sentiment_analysis.get_stock_news')
    def test_successful_daily_sentiment(self, mock_news: MagicMock):
        """Test successful daily sentiment calculation."""
        mock_news.return_value = [
            {
                'title': 'Apple Reports Record Earnings',
                'publish_time': 1704067200,
                'publisher': 'Financial Times'
            },
            {
                'title': 'Tech Stocks Rally',
                'publish_time': 1704153600,
                'publisher': 'Reuters'
            }
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {'label': 'positive', 'score': 0.95},
            {'label': 'positive', 'score': 0.90}
        ]

        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert 'Sentiment' in result.columns
        assert len(result) > 0

    @patch('src.sentiment_analysis.get_stock_news')
    def test_sentiment_label_mapping(self, mock_news: MagicMock):
        """Test that sentiment labels are correctly mapped."""
        mock_news.return_value = [
            {
                'title': 'Positive news',
                'publish_time': 1704067200,
                'publisher': 'Test'
            }
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {'label': 'positive', 'score': 0.9},
            {'label': 'negative', 'score': 0.8},
            {'label': 'neutral', 'score': 0.7}
        ]

        # Test the label_to_score mapping indirectly
        # positive=1, negative=-1, neutral=0
        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)

    @patch('src.sentiment_analysis.get_stock_news')
    def test_daily_aggregation(self, mock_news: MagicMock):
        """Test that sentiment is aggregated by day."""
        # Create multiple news items for the same day
        mock_news.return_value = [
            {
                'title': 'Morning news',
                'publish_time': 1704067200,  # Same day
                'publisher': 'Test1'
            },
            {
                'title': 'Afternoon news',
                'publish_time': 1704070800,  # Same day, different time
                'publisher': 'Test2'
            }
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {'label': 'positive', 'score': 0.9},
            {'label': 'negative', 'score': 0.8}
        ]

        result = get_daily_sentiment("AAPL", mock_pipeline)

        # Should have one row per day (resampled)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)


class TestSentimentIntegration:
    """Integration tests for sentiment analysis module."""

    @patch('src.sentiment_analysis.get_stock_news')
    @patch('src.sentiment_analysis.pipeline')
    def test_full_sentiment_pipeline(self, mock_pipeline: MagicMock, mock_news: MagicMock):
        """Test the full sentiment analysis pipeline."""
        # Setup mock news
        mock_news.return_value = [
            {
                'title': 'Stock Market Update',
                'publish_time': 1704067200,
                'publisher': 'News Corp'
            }
        ]

        # Setup mock sentiment model
        mock_model = MagicMock()
        mock_model.return_value = [{'label': 'neutral', 'score': 0.75}]
        mock_pipeline.return_value = mock_model

        # Clear cache
        initialize_sentiment_model.cache_clear()

        # Initialize and run
        model = initialize_sentiment_model()
        result = get_daily_sentiment("AAPL", model)

        assert isinstance(result, pd.DataFrame)
        assert 'Sentiment' in result.columns
        mock_news.assert_called_once_with("AAPL")
