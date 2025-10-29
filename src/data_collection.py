import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start, end):
    """
    Fetches historical stock data from Yahoo Finance.
    Handles column MultiIndex.
    """
    print(f"Fetching data for {ticker} from {start} to {end}...")
    
    # Download data. group_by='column' should help, but we'll add a manual fix.
    data = yf.download(ticker, start=start, end=end, group_by='column')
    
    # --- FIX for MultiIndex Columns ---
    # yfinance sometimes returns a MultiIndex for columns (e.g., ('Close', 'AAPL'))
    # This checks if the columns are a MultiIndex and flattens them.
    if isinstance(data.columns, pd.MultiIndex):
        print("Flattening MultiIndex columns...")
        # Keep the main column name (e.g., 'Close') and discard the ticker part
        data.columns = data.columns.get_level_values(0)
    # --- END FIX ---

    if data.empty:
        raise ValueError(f"No data found for {ticker} between {start} and {end}.")
    
    # Ensure index is datetime (it should be, but good to check)
    data.index = pd.to_datetime(data.index)
    
    return data

def get_stock_news(ticker):
    """
    Fetches recent news headlines for a stock.
    """
    print(f"Fetching news for {ticker}...")
    stock = yf.Ticker(ticker)
    try:
        # 'news' is the correct attribute
        news = stock.news
        if not news:
            print(f"No news found for {ticker}.")
            return []
        # 'news' returns a list of dictionaries
        return news
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

