"""
Stock Price Predictor - Streamlit Web Application.

Interactive web dashboard for predicting stock closing prices using
LSTM deep learning model with technical indicators and news sentiment.

Features:
    - Configurable ticker symbol and date range
    - Real-time prediction with progress indicators
    - Interactive Plotly charts
    - Downloadable prediction data and metrics
    - Error recovery with retry functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import timedelta, date
from typing import Optional, Any, List, Tuple
import plotly.graph_objects as go

# Import project-specific modules
from src.data_collection import get_stock_data, get_stock_news
from src.feature_engineering import add_technical_indicators
from src.sentiment_analysis import initialize_sentiment_model, get_daily_sentiment
from src.model_utils import (
    load_model,
    load_scaler,
    create_dataset
)
from src.data_preprocessing import merge_stock_sentiment, prepare_features
from src.evaluation import evaluate_predictions
from src.logger import get_logger

# Import configuration
from config import settings


logger = get_logger(__name__)


# --- Page Configuration ---
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon=settings.PAGE_ICON,
    layout=settings.LAYOUT
)

# --- Custom CSS for better styling ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
def init_session_state() -> None:
    """Initialize session state variables."""
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'prediction_date' not in st.session_state:
        st.session_state.prediction_date = None
    if 'last_updated' not in st.session_state:
        st.session_state.last_updated = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False


def clear_session_state() -> None:
    """Clear prediction-related session state."""
    st.session_state.prediction_result = None
    st.session_state.prediction_date = None
    st.session_state.error_message = None


# --- Caching Functions ---
@st.cache_resource
def load_assets() -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[List[str]]]:
    """
    Load all necessary model assets (model, scalers, feature list).

    Uses Streamlit's cache to load only once per session.

    Returns:
        Tuple of (model, scaler_x, scaler_y, features) where any value
        may be None if loading failed.
    """
    try:
        logger.info("Loading model assets...")
        model = load_model(settings.model_path)
        scaler_x, features = load_scaler(settings.scaler_x_path)
        scaler_y, _ = load_scaler(settings.scaler_y_path)

        # Load feature list from json (fallback)
        if features is None:
            if not os.path.exists(settings.scaler_features_path):
                logger.error(f"Features file not found: {settings.scaler_features_path}")
                return None, None, None, None
            with open(settings.scaler_features_path, 'r') as f:
                features = json.load(f)

        logger.info("Model assets loaded successfully")
        return model, scaler_x, scaler_y, features

    except Exception as e:
        logger.error(f"Error loading model assets: {e}")
        return None, None, None, None


@st.cache_resource
def load_sentiment_pipeline() -> Any:
    """
    Load the sentiment analysis pipeline.

    Uses Streamlit's cache to load only once per session.

    Returns:
        Initialized FinBERT sentiment analysis pipeline.
    """
    logger.info("Loading sentiment pipeline...")
    return initialize_sentiment_model()


# --- Prediction Function ---
def run_prediction(
    ticker: str,
    model: Any,
    scaler_x: Any,
    scaler_y: Any,
    features: List[str],
    time_step: int,
    buffer_days: int
) -> Tuple[Optional[float], Optional[pd.DataFrame], Optional[str]]:
    """
    Run the prediction pipeline for a given ticker.

    Args:
        ticker: Stock ticker symbol.
        model: Loaded Keras model.
        scaler_x: Fitted scaler for features.
        scaler_y: Fitted scaler for target.
        features: List of feature names.
        time_step: Number of days for LSTM input.
        buffer_days: Extra days for indicator calculation.

    Returns:
        Tuple of (predicted_price, data_df, error_message).
    """
    try:
        end_date = date.today()
        
        # FIX: Multiply by 1.5 to safely convert trading day requirements into calendar days, 
        # ensuring weekends and market holidays don't leave us short of the required time_step.
        calendar_days_needed = int((time_step + buffer_days) * 1.5)
        start_date = end_date - timedelta(days=calendar_days_needed)

        # Fetch stock data
        stock_data = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        # Add technical indicators
        stock_data = add_technical_indicators(stock_data)

        # Get sentiment (optional - don't fail if unavailable)
        try:
            sentiment_pipeline = load_sentiment_pipeline()
            daily_sentiment = get_daily_sentiment(ticker, sentiment_pipeline)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}, proceeding without sentiment")
            daily_sentiment = None

        # Merge data
        data = merge_stock_sentiment(stock_data, daily_sentiment)

        # Fill any remaining NaN values
        data = data.ffill().bfill()

        # Get last N days for prediction
        last_n_days = data[features].tail(time_step)

        if last_n_days.shape[0] < time_step:
            return None, data, f"Insufficient data: got {last_n_days.shape[0]} days, need {time_step}"

        # Scale and prepare input
        last_n_days_scaled = scaler_x.transform(last_n_days)

        X_test = np.array([last_n_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))

        # Make prediction
        predicted_price_scaled = model.predict(X_test, verbose=0)
        predicted_price = scaler_y.inverse_transform(predicted_price_scaled)

        return predicted_price[0][0], data, None

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None, None, str(e)


# --- Plotting Function ---
def plot_prediction_chart(
    history_df: pd.DataFrame,
    predicted_price: float,
    ticker: str,
    history_days: int
) -> go.Figure:
    """
    Create a Plotly chart for historical data and the prediction.

    Args:
        history_df: DataFrame with historical closing prices indexed by date.
        predicted_price: Predicted closing price for the next day.
        ticker: Stock ticker symbol for display.
        history_days: Number of days to display.

    Returns:
        Plotly Figure object with historical line chart and prediction marker.
    """
    logger.debug(f"Creating prediction chart for {ticker}")

    # Get the last N days of history
    history = history_df.tail(history_days).copy()

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
        line=dict(color='#1f77b4', width=2)
    ))

    # Add prediction trace
    fig.add_trace(go.Scatter(
        x=plot_df[plot_df['Type'] == 'Prediction']['Date'],
        y=plot_df[plot_df['Type'] == 'Prediction']['Close'],
        mode='markers+lines',
        name='Predicted Close',
        marker=dict(color='#ff7f0e', size=15, symbol='star'),
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Closing Price: Last {history_days} Days & Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        legend_title_text='Data Type',
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    return fig


def create_download_dataframe(
    data: pd.DataFrame,
    prediction: float,
    ticker: str
) -> pd.DataFrame:
    """
    Create a DataFrame for download with prediction info.

    Args:
        data: Historical data DataFrame.
        prediction: Predicted price.
        ticker: Stock ticker symbol.

    Returns:
        DataFrame ready for CSV download.
    """
    # Get last 60 days + prediction
    history = data.tail(60).copy()
    prediction_date = history.index.max() + pd.Timedelta(days=1)

    # Create prediction row
    pred_row = pd.DataFrame({
        'Date': [prediction_date],
        'Close': [prediction],
        'Type': ['Prediction']
    })

    history['Type'] = 'Historical'
    result = pd.concat([history.reset_index(), pred_row], ignore_index=True)
    result['Ticker'] = ticker
    result['GeneratedAt'] = date.today().isoformat()

    return result


# --- Sidebar Configuration ---
def render_sidebar() -> Tuple[str, int, int, int]:
    """
    Render the sidebar with configuration options.

    Returns:
        Tuple of (ticker, time_step, history_days, buffer_days).
    """
    st.sidebar.header("⚙️ Configuration")

    # Ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter the stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
    )

    st.sidebar.divider()

    # Model parameters
    st.sidebar.subheader("Model Parameters")

    time_step = st.sidebar.slider(
        "Time Steps (Days of History)",
        min_value=10,
        max_value=120,
        value=60,
        step=5,
        help="Number of days of historical data to use for each prediction"
    )

    buffer_days = st.sidebar.slider(
        "Buffer Days (for indicators)",
        min_value=30,
        max_value=200,
        value=100,
        step=10,
        help="Extra days to fetch for calculating technical indicators"
    )

    st.sidebar.divider()

    # Display parameters
    st.sidebar.subheader("Display Options")

    history_days = st.sidebar.slider(
        "History Display (Days)",
        min_value=30,
        max_value=365,
        value=90,
        step=15,
        help="Number of days to display in the chart"
    )

    st.sidebar.divider()

    # Info box
    st.sidebar.info(
        """
        **How it works:**
        1. Fetches historical stock data
        2. Calculates technical indicators (RSI, MA20, MA50)
        3. Analyzes news sentiment using FinBERT
        4. Uses LSTM model to predict next day's price

        *Note: This is not financial advice.*
        """
    )

    return ticker, time_step, history_days, buffer_days


# --- Main App ---
def main() -> None:
    """Main application entry point."""

    # Initialize session state
    init_session_state()

    # Render sidebar
    ticker, time_step, history_days, buffer_days = render_sidebar()

    # Main header
    st.markdown(f'<p class="main-header">📈 AI Stock Price Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        f"Predicting closing price for **{ticker.upper()}** using LSTM deep learning, "
        "technical indicators, and news sentiment analysis."
    )

    # Load models
    with st.spinner("Loading model assets..."):
        model, scaler_x, scaler_y, features = load_assets()

    # Check if assets loaded
    if model is None or scaler_x is None or scaler_y is None or features is None:
        st.error("❌ Model assets could not be loaded.")
        st.info(
            f"Please ensure the following files exist:\n\n"
            f"- `{settings.model_path}`\n"
            f"- `{settings.scaler_x_path}`\n"
            f"- `{settings.scaler_y_path}`\n"
            f"- `{settings.scaler_features_path}`\n\n"
            f"Run `python train_model.py` to train the model first."
        )
        return

    st.success("✅ Model loaded successfully!")

    # --- Prediction Section ---
    st.divider()
    st.subheader("🔮 Prediction")

    # Create columns for buttons
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        predict_button = st.button(
            "🚀 Run Prediction",
            type="primary",
            use_container_width=True,
            help="Fetch latest data and run prediction"
        )

    with col2:
        clear_button = st.button(
            "🗑️ Clear Results",
            type="secondary",
            use_container_width=True,
            help="Clear current prediction results"
        )

    # Handle clear action
    if clear_button:
        clear_session_state()
        st.rerun()

    # Handle prediction action
    if predict_button:
        st.session_state.is_loading = True

        # Use status container for progress
        with st.status(f"Running prediction for {ticker.upper()}...", expanded=True) as status:

            step1 = status.container()
            step1.write("📊 **Step 1/4:** Fetching historical data...")

            step2 = status.container()
            step2.write("📈 **Step 2/4:** Calculating technical indicators...")

            step3 = status.container()
            step3.write("📰 **Step 3/4:** Analyzing news sentiment...")

            step4 = status.container()
            step4.write("🤖 **Step 4/4:** Running LSTM prediction...")

            # Run prediction
            prediction, data, error = run_prediction(
                ticker.upper(),
                model,
                scaler_x,
                scaler_y,
                features,
                time_step,
                buffer_days
            )

            if error:
                status.update(label="❌ Prediction failed", state="error", expanded=True)
                st.session_state.error_message = error
                st.session_state.is_loading = False
                st.error(f"**Error:** {error}")
            else:
                status.update(label="✅ Prediction complete!", state="complete", expanded=True)
                st.session_state.prediction_result = {
                    'prediction': prediction,
                    'data': data,
                    'ticker': ticker.upper()
                }
                st.session_state.prediction_date = date.today()
                st.session_state.is_loading = False
                st.rerun()

    # --- Display Results ---
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        prediction_val = result['prediction']
        data = result['data']
        current_ticker = result['ticker']

        # Get last actual price
        last_actual_price = data['Close'].iloc[-1]
        price_change = prediction_val - last_actual_price
        price_change_pct = (price_change / last_actual_price) * 100

        # Display prediction metrics
        st.divider()
        st.subheader("📊 Prediction Results")

        # Metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=f"Predicted Price ({current_ticker})",
                value=f"${prediction_val:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )

        with col2:
            st.metric(
                label="Last Close Price",
                value=f"${last_actual_price:.2f}"
            )

        with col3:
            st.metric(
                label="Prediction Date",
                value=st.session_state.prediction_date.strftime("%Y-%m-%d") if st.session_state.prediction_date else "N/A"
            )

        # Explanation
        st.markdown("""
        #### 🧠 How This Prediction Works
        This prediction is generated by an **LSTM (Long Short-Term Memory)** neural network that analyzes:
        1. **Price History** - The last 60 days of stock prices and volume
        2. **Technical Indicators** - RSI, 20-day MA, and 50-day moving averages
        3. **News Sentiment** - Sentiment scores from recent financial news (FinBERT model)

        > ⚠️ **Disclaimer:** This is an AI-generated prediction for educational purposes only.
        > Do not use as the sole basis for investment decisions.
        """)

        # Chart
        st.divider()
        st.subheader("📈 Price Chart")
        fig = plot_prediction_chart(data, prediction_val, current_ticker, history_days)
        st.plotly_chart(fig, use_container_width=True)

        # Download section
        st.divider()
        st.subheader("📥 Download Results")

        download_df = create_download_dataframe(data, prediction_val, current_ticker)
        csv = download_df.to_csv(index=False)

        st.download_button(
            label="📊 Download Prediction Data (CSV)",
            data=csv,
            file_name=f"{current_ticker}_prediction_{date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Recent data table
        with st.expander("📋 View Recent Data"):
            st.dataframe(data.tail(10), use_container_width=True)

    elif st.session_state.error_message:
        st.error(f"❌ Previous error: {st.session_state.error_message}")
        st.info("Click 'Run Prediction' to try again.")

    else:
        # No prediction yet - show placeholder
        st.info("👆 Click 'Run Prediction' to get started!")

        # Show sample chart or info
        st.markdown("""
        ### What You'll See
        Once you run a prediction, you'll see:
        - **Prediction Metric** - Tomorrow's predicted closing price with change vs today
        - **Interactive Chart** - Last 90 days of history + prediction marker
        - **Download Button** - Export prediction data as CSV
        - **Data Table** - Recent data used for the prediction
        """)


if __name__ == "__main__":
    main()
