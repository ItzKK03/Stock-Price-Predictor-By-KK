import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import timedelta, date
import plotly.graph_objects as go # Import Plotly

# Import project-specific modules
from src.data_collection import get_stock_data, get_stock_news
from src.feature_engineering import add_technical_indicators
from src.sentiment_analysis import initialize_sentiment_model, get_daily_sentiment
from src.model_utils import (
    load_model, 
    load_scaler, 
    create_dataset
)

# --- App Configuration ---
st.set_page_config(
    page_title="AI Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# --- Constants ---
TICKER = 'AAPL'
TIME_STEP = 60 # Must match the training TIME_STEP

# --- Define File Paths ---
MODEL_PATH = 'stock_predictor.keras'
SCALER_X_PATH = 'scaler_x.joblib'
SCALER_Y_PATH = 'scaler_y.joblib'
SCALER_FEATURES_PATH = 'scaler_features.json'


# --- Model Loading ---
@st.cache_resource
def load_assets():
    """
    Loads all necessary model assets (model, scalers, feature list).
    Uses Streamlit's cache to load only once.
    """
    try:
        model = load_model(MODEL_PATH)
        scaler_x, features = load_scaler(SCALER_X_PATH)
        scaler_y, _ = load_scaler(SCALER_Y_PATH)
        
        # Load feature list from json (fallback)
        if features is None:
            if not os.path.exists(SCALER_FEATURES_PATH):
                st.error(f"Fatal Error: {SCALER_FEATURES_PATH} not found.")
                return None, None, None, None
            with open(SCALER_FEATURES_PATH, 'r') as f:
                features = json.load(f)
                
        return model, scaler_x, scaler_y, features
    
    except Exception as e:
        st.error(f"An error occurred while loading model assets: {e}")
        return None, None, None, None

@st.cache_resource
def load_sentiment_pipeline():
    """Loads the sentiment analysis model."""
    return initialize_sentiment_model()

# --- NEW: Plotting Function ---
def plot_prediction_chart(history_df, predicted_price, ticker):
    """
    Creates a Plotly chart for historical data and the prediction.
    """
    # Get the last 90 days of history
    history = history_df.tail(90).copy()
    
    # Create the date for the prediction (last date + 1)
    prediction_date = history.index.max() + pd.Timedelta(days=1)
    
    # Create a DataFrame for the prediction
    pred_df = pd.DataFrame({
        'Date': [prediction_date], 
        'Close': [predicted_price],
        'Type': ['Prediction']
    })
    
    # Label the history
    history['Type'] = 'Historical'

    # Combine for plotting
    plot_df = pd.concat([history.reset_index(), pred_df], ignore_index=True)
    
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(go.Scatter(
        x=plot_df[plot_df['Type'] == 'Historical']['Date'],
        y=plot_df[plot_df['Type'] == 'Historical']['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='#1f77b4') # Blue
    ))

    # Add prediction trace
    fig.add_trace(go.Scatter(
        x=plot_df[plot_df['Type'] == 'Prediction']['Date'],
        y=plot_df[plot_df['Type'] == 'Prediction']['Close'],
        mode='markers',
        name='Predicted Close',
        marker=dict(color='#ff7f0e', size=10, symbol='star') # Orange star
    ))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Closing Price: Last 90 Days & Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark', # Use dark theme
        legend_title_text='Data Type',
        xaxis_rangeslider_visible=False
    )
    
    return fig

# --- Main App ---
st.title(f"📈 AI Stock Price Predictor for {TICKER}")
st.markdown(f"This app predicts the closing price for **{TICKER}** using an LSTM model, technical indicators, and news sentiment analysis.")

# Load models
model, scaler_x, scaler_y, features = load_assets()
sentiment_pipeline = load_sentiment_pipeline()

if model is None or scaler_x is None or scaler_y is None or features is None:
    st.error("Model assets could not be loaded. Cannot run the app.")
    st.info(f"Please ensure `{MODEL_PATH}`, `{SCALER_X_PATH}`, `{SCALER_Y_PATH}`, and `{SCALER_FEATURES_PATH}` exist in your project folder.")
else:
    st.success("Model, scalers, and feature list loaded successfully!")

    # --- Data Fetching ---
    st.subheader("Fetching Latest Data...")
    with st.spinner(f"Getting the latest stock and news data for {TICKER}..."):
        try:
            # Fetch data for the last TIME_STEP days + buffer
            end_date = date.today()
            start_date = end_date - timedelta(days=TIME_STEP + 100) # Get extra data for indicators
            
            # Get Stock Data
            stock_data = get_stock_data(TICKER, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Get Technical Indicators
            stock_data = add_technical_indicators(stock_data)
            
            # Get Sentiment
            daily_sentiment = get_daily_sentiment(TICKER, sentiment_pipeline)
            
            # --- Data Merging and Preprocessing ---
            if not isinstance(stock_data.index, pd.DatetimeIndex):
                stock_data.index = pd.to_datetime(stock_data.index)
            
            stock_data['Date'] = stock_data.index.date
            
            if isinstance(daily_sentiment.index, pd.DatetimeIndex):
                daily_sentiment['Date'] = daily_sentiment.index.date
            else:
                daily_sentiment['Date'] = pd.NaT 

            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

            stock_data = stock_data.reset_index(drop=True)
            daily_sentiment = daily_sentiment.reset_index(drop=True)

            data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')
            
            data = data.set_index('Date')
            
            data['Sentiment'] = data['Sentiment'].fillna(0)
            data = data.ffill().bfill() 
            
            last_60_days = data[features].tail(TIME_STEP)

            if last_60_days.shape[0] < TIME_STEP:
                st.error(f"Could not retrieve enough data ({last_60_days.shape[0]} days) to make a prediction. Need {TIME_STEP} days.")
            else:
                st.success("Data fetched successfully!")

                # --- Prediction ---
                st.subheader("Prediction")
                
                last_60_days_scaled = scaler_x.transform(last_60_days)
                
                X_test = []
                X_test.append(last_60_days_scaled)
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))
                
                predicted_price_scaled = model.predict(X_test)
                predicted_price = scaler_y.inverse_transform(predicted_price_scaled)
                
                last_actual_price = data['Close'].iloc[-1]
                prediction_val = predicted_price[0][0]
                
                # --- NEW: Column Layout ---
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display Prediction Metric
                    st.metric(
                        label=f"Predicted Closing Price for {TICKER} (Tomorrow)",
                        value=f"${prediction_val:.2f}",
                        delta=f"{prediction_val - last_actual_price:.2f} (vs last close)"
                    )
                
                with col2:
                    # Display Explanatory Text
                    st.markdown("#### How this prediction is made:")
                    st.markdown("""
                    This value is predicted by an AI model (LSTM) that analyzed:
                    1.  **Price History:** The last 60 days of stock prices.
                    2.  **Technical Indicators:** Trends like RSI and Moving Averages.
                    3.  **News Sentiment:** The emotional tone of recent news headlines.
                    
                    *Disclaimer: This is an AI-generated prediction, not financial advice.*
                    """)
                
                # --- NEW: Display Chart ---
                st.plotly_chart(
                    plot_prediction_chart(data, prediction_val, TICKER), 
                    use_container_width=True
                )
                
                # --- Display Data ---
                st.subheader("Recent Data Used for Prediction")
                st.dataframe(data.tail(10)) 

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e)

