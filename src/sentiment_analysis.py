"""
Sentiment Analysis Module.

Handles sentiment analysis of stock news using FinBERT model.
Includes caching for model initialization to improve performance.
"""

from functools import lru_cache
from transformers import pipeline
import pandas as pd
from typing import List, Dict, Any, Optional
from src.logger import get_logger

from .data_collection import get_stock_news

logger = get_logger(__name__)

# Use a model specifically trained on financial news
MODEL_NAME: str = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def initialize_sentiment_model(model_name: str = MODEL_NAME) -> Any:
    """
    Initialize and return the FinBERT sentiment analysis pipeline.
    Uses LRU cache to ensure the model is only loaded once per process.

    Args:
        model_name: Hugging Face model name (default: ProsusAI/finbert).

    Returns:
        Hugging Face transformers pipeline for sentiment analysis.

    Raises:
        Exception: If model initialization fails.
    """
    logger.info(f"Initializing sentiment model: {model_name}...")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        logger.info("Sentiment model initialized successfully.")
        return sentiment_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize sentiment model: {e}")
        raise


def get_sentiment_scores(
    headlines: List[str],
    sentiment_pipeline: Any,
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """
    Analyze a list of headlines and return sentiment scores.
    Processes headlines in batches to avoid memory issues with large lists.

    Args:
        headlines: List of news headline strings.
        sentiment_pipeline: Initialized Hugging Face sentiment pipeline.
        batch_size: Number of headlines to process per batch (default: 8).

    Returns:
        List of dictionaries with 'label' and 'score' for each headline.

    Raises:
        ValueError: If headlines is empty or not a list.
    """
    # Input validation
    if not headlines:
        logger.warning("Empty headlines list provided")
        return []

    if not isinstance(headlines, list):
        logger.error(f"headlines must be a list, got {type(headlines)}")
        raise ValueError("headlines must be a list")

    # Filter out None or empty strings
    valid_headlines = [h for h in headlines if h and isinstance(h, str)]
    if not valid_headlines:
        logger.warning("No valid headlines after filtering")
        return []

    logger.debug(f"Analyzing sentiment for {len(valid_headlines)} headlines in batches of {batch_size}")

    all_sentiments: List[Dict[str, Any]] = []

    try:
        # Process in batches
        for i in range(0, len(valid_headlines), batch_size):
            batch = valid_headlines[i:i + batch_size]
            batch_sentiments = sentiment_pipeline(batch)
            all_sentiments.extend(batch_sentiments)

        logger.debug(f"Sentiment analysis complete for {len(all_sentiments)} headlines")
        return all_sentiments

    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        raise


def get_daily_sentiment(
    ticker: str,
    sentiment_pipeline: Any
) -> pd.DataFrame:
    """
    Fetch news, analyze sentiment, and aggregate by day.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        sentiment_pipeline: Initialized Hugging Face sentiment pipeline.

    Returns:
        DataFrame with Date as index and 'Sentiment' column containing
        weighted sentiment scores. Returns empty DataFrame if no news found.

    Raises:
        ValueError: If ticker is invalid.
    """
    # Input validation
    if not ticker or not isinstance(ticker, str):
        logger.error(f"Invalid ticker: {ticker}")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    logger.debug(f"Fetching daily sentiment for {ticker}")

    news = get_stock_news(ticker)

    # Handle empty news
    if not news:
        logger.warning("No news returned, returning empty sentiment DataFrame")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    # Robust normalization of news data to handle changing API structures
    normalized_news = []
    for item in news:
        if not isinstance(item, dict):
            continue
            
        content = item.get('content', item) if isinstance(item.get('content'), dict) else {}
        
        title = item.get('title') or content.get('title') or item.get('headline')
        pub_time = item.get('publish_time') or item.get('providerPublishTime') or content.get('pubDate') or item.get('publishedAt')
        
        if title and pub_time:
            normalized_news.append({'title': title, 'publish_time': pub_time})

    df = pd.DataFrame(normalized_news)

    # Validate required columns
    if 'title' not in df.columns:
        logger.warning("News found, but could not extract 'title' column from the API response structure.")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    # Clean and validate titles
    df = df.dropna(subset=['title'])
    df['title'] = df['title'].astype(str)

    if df.empty:
        logger.warning("News titles were empty after cleaning")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    if 'publish_time' not in df.columns:
        logger.warning("News found, but could not extract 'publish_time' column")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    # Process dates safely, handling both Unix timestamps and standard date strings
    try:
        if pd.api.types.is_numeric_dtype(df['publish_time']):
            df['publish_time'] = pd.to_datetime(df['publish_time'], unit='s')
        else:
            df['publish_time'] = pd.to_datetime(df['publish_time'])
            
        if df['publish_time'].dt.tz is not None:
            df['publish_time'] = df['publish_time'].dt.tz_localize(None)
    except Exception as e:
        logger.error(f"Error parsing dates: {e}")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    df = df.set_index('publish_time')

    # Analyze sentiment
    headlines = df['title'].tolist()
    logger.debug(f"Analyzing {len(headlines)} news headlines")
    sentiments = get_sentiment_scores(headlines, sentiment_pipeline)

    df['sentiment_label'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]

    # Convert labels to numerical scores
    label_to_score: Dict[str, int] = {'positive': 1, 'negative': -1, 'neutral': 0}
    df['numeric_sentiment'] = df['sentiment_label'].map(label_to_score)

    # Weight score by confidence
    df['weighted_sentiment'] = df['numeric_sentiment'] * df['sentiment_score']

    # Resample by day and calculate mean
    daily_sentiment = df['weighted_sentiment'].resample('D').mean()

    daily_sentiment_df = daily_sentiment.to_frame(name='Sentiment')
    daily_sentiment_df.index.name = 'Date'

    logger.info(f"Generated sentiment for {len(daily_sentiment_df)} days")
    return daily_sentiment_df