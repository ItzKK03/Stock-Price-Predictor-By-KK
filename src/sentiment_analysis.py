from transformers import pipeline
import pandas as pd
from .data_collection import get_stock_news

# Use a model specifically trained on financial news
MODEL_NAME = "ProsusAI/finbert"

def initialize_sentiment_model():
    """
    Initializes and returns the FinBERT sentiment analysis pipeline.
    """
    print("Initializing sentiment model...")
    sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
    print("Sentiment model initialized.")
    return sentiment_pipeline

def get_sentiment_scores(headlines, sentiment_pipeline):
    """
    Analyzes a list of headlines and returns sentiment scores.
    """
    # The pipeline can process a list of texts directly
    sentiments = sentiment_pipeline(headlines)
    return sentiments

def get_daily_sentiment(ticker, sentiment_pipeline):
    """
    Fetches news, analyzes sentiment, and aggregates it by day.
    Returns a DataFrame with Date (index) and Sentiment (score).
    """
    news = get_stock_news(ticker)
    
    # --- FIX for KeyError ---
    # If get_stock_news returns an empty list, 'news' will be False.
    if not news:
        print("No news returned. Returning empty sentiment DataFrame.")
        # Return an empty DataFrame with the expected structure
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')
    # --- END FIX ---

    df = pd.DataFrame(news)
    
    # Ensure 'title' exists and is a string, drop rows where it's missing (NaN)
    # This check is now safe because we know 'df' is not empty.
    if 'title' not in df.columns:
        print("News found, but no 'title' column. Returning empty sentiment DataFrame.")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')
        
    df = df.dropna(subset=['title'])
    df['title'] = df['title'].astype(str)
    
    # Check if all news was filtered out
    if df.empty:
        print("News titles were empty. Returning empty sentiment DataFrame.")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')
    
    # Ensure publish_time is datetime and set as index
    # Check if 'publish_time' column exists
    if 'publish_time' not in df.columns:
        print("News found, but no 'publish_time' column. Cannot calculate daily sentiment.")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')
        
    df['publish_time'] = pd.to_datetime(df['publish_time']).dt.tz_localize(None)
    df = df.set_index('publish_time')

    # Analyze sentiment
    headlines = df['title'].tolist()
    sentiments = get_sentiment_scores(headlines, sentiment_pipeline)
    
    df['sentiment_label'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]

    # Convert labels to numerical scores: positive=1, negative=-1, neutral=0
    label_to_score = {'positive': 1, 'negative': -1, 'neutral': 0}
    df['numeric_sentiment'] = df['sentiment_label'].map(label_to_score)
    
    # Weight score by its confidence: e.g., 0.9 positive -> 1 * 0.9 = 0.9
    # e.g., 0.8 negative -> -1 * 0.8 = -0.8
    df['weighted_sentiment'] = df['numeric_sentiment'] * df['sentiment_score']

    # Resample by day ('D') and calculate the mean sentiment
    daily_sentiment = df['weighted_sentiment'].resample('D').mean()
    
    daily_sentiment_df = daily_sentiment.to_frame(name='Sentiment')
    daily_sentiment_df.index.name = 'Date'
    
    return daily_sentiment_df

