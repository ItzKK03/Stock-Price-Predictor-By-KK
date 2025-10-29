import pandas as pd
import pandas_ta as ta

def add_technical_indicators(df):
    """
    Adds technical indicators (RSI, MA20, MA50) to the DataFrame.
    """
    # Ensure the DataFrame is not empty
    if df.empty:
        return df

    # --- FIX ---
    # Explicitly pass the 'Close' price Series to the functions.
    # This avoids the internal .str.match error in pandas_ta
    # by bypassing its column-finding logic.

    # Calculate RSI
    df.ta.rsi(close=df['Close'], length=14, append=True)
    
    # Calculate 20-day Moving Average
    df.ta.sma(close=df['Close'], length=20, append=True)
    
    # Calculate 50-day Moving Average
    df.ta.sma(close=df['Close'], length=50, append=True)

    # Rename columns for clarity (pandas_ta creates names like SMA_20)
    df.rename(columns={
        'RSI_14': 'RSI',
        'SMA_20': 'MA20',
        'SMA_50': 'MA50'
    }, inplace=True)
    
    return df

