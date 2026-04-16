"""
Stock Price Predictor Package.

A comprehensive AI-powered stock price prediction system using LSTM,
technical indicators, and NLP sentiment analysis.

Modules:
    data_collection: Fetch stock data and news from Yahoo Finance.
    feature_engineering: Calculate technical indicators (RSI, MA).
    sentiment_analysis: Analyze news sentiment using FinBERT.
    model_utils: Build, train, and load LSTM models.
    data_preprocessing: Merge, clean, and prepare data for modeling.
    evaluation: Model evaluation metrics (MAE, RMSE, MAPE, directional accuracy).
    logger: Centralized logging configuration.

Usage:
    from src.data_collection import get_stock_data
    from src.feature_engineering import add_technical_indicators
    from src.sentiment_analysis import get_daily_sentiment
    from src.model_utils import build_model, create_dataset
    from src.data_preprocessing import merge_stock_sentiment
    from src.evaluation import evaluate_predictions, generate_evaluation_report
    from src.logger import get_logger

Version:
    1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Stock Price Predictor Team'
