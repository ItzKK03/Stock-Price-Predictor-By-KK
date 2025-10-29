import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Import project-specific modules
from src.data_collection import get_stock_data, get_stock_news
from src.feature_engineering import add_technical_indicators
from src.sentiment_analysis import initialize_sentiment_model, get_daily_sentiment
from src.model_utils import build_model, create_dataset, save_model, save_scaler

# --- Constants ---
TICKER = 'AAPL'
START_DATE = '2015-01-01'
END_DATE = '2025-10-25' # Use a recent date
TIME_STEP = 60 # How many days of history to use for prediction

# --- File Paths ---
MODEL_PATH = 'stock_predictor.keras'
SCALER_X_PATH = 'scaler_x.joblib'
SCALER_Y_PATH = 'scaler_y.joblib'
SCALER_FEATURES_PATH = 'scaler_features.json'

def train():
    """
    Main training pipeline.
    """
    print("Starting training process for AAPL...")

    # --- 1. Initialize Models ---
    print("Initializing sentiment model...")
    sentiment_pipeline = initialize_sentiment_model()

    # --- 2. Data Collection ---
    print(f"Fetching data for {TICKER} from {START_DATE} to {END_DATE}...")
    stock_data = get_stock_data(TICKER, START_DATE, END_DATE)
    
    # --- 3. Feature Engineering ---
    print("Adding technical indicators...")
    stock_data = add_technical_indicators(stock_data)
    
    # --- 4. Sentiment Analysis ---
    print("Fetching and analyzing news sentiment...")
    daily_sentiment = get_daily_sentiment(TICKER, sentiment_pipeline)

    # --- 5. Data Merging and Preprocessing ---
    print("Combining stock data and sentiment...")
    
    # Ensure 'Date' is a column for merging
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)
    
    stock_data['Date'] = stock_data.index.date
    
    # --- FIX: Handle empty sentiment DataFrame ---
    if isinstance(daily_sentiment.index, pd.DatetimeIndex):
        daily_sentiment['Date'] = daily_sentiment.index.date
    else:
        daily_sentiment['Date'] = pd.NaT
    
    # --- FIX: Reset index to avoid merge ambiguity ---
    stock_data = stock_data.reset_index(drop=True)
    daily_sentiment = daily_sentiment.reset_index(drop=True)
    # --- End of Fix ---

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    
    # Merge data
    data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')
    
    # Set 'Date' as index for time-series processing
    data = data.set_index('Date')
    
    # Fill NaN sentiment with 0 (neutral)
    data['Sentiment'] = data['Sentiment'].fillna(0)
    
    # Drop any other NaNs (e.g., from indicators at the start)
    data = data.dropna()
    
    if data.empty:
        print("Error: No data available after merging and cleaning. Exiting.")
        return

    # Define features to be used for training
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'MA20', 'MA50', 'Sentiment']
    features_for_y = ['Close'] # What we want to predict
    
    # Ensure all features are present
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Error: Missing features in data: {missing_features}. Exiting.")
        return

    # Create the X (features) and y (target) datasets
    data_x = data[features]
    data_y = data[features_for_y]
    
    # --- 6. Data Scaling ---
    print("Scaling data...")
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaled_x = scaler_x.fit_transform(data_x)
    scaled_y = scaler_y.fit_transform(data_y)
    
    # Save scalers and feature list
    save_scaler(scaler_x, SCALER_X_PATH, features)
    save_scaler(scaler_y, SCALER_Y_PATH, features_for_y)
    with open(SCALER_FEATURES_PATH, 'w') as f:
        json.dump(features, f)
    print(f"Feature list saved to {SCALER_FEATURES_PATH}")
        
    # --- 7. Create Time-Series Dataset ---
    print(f"Creating time-series dataset with time_step={TIME_STEP}...")
    X_train, y_train = create_dataset(scaled_x, scaled_y, TIME_STEP)
    
    # Reshape X_train for LSTM [samples, time_steps, n_features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))
    
    if X_train.shape[0] == 0:
        print(f"Error: Not enough data to create training samples. Need at least {TIME_STEP+1} days of data. Exiting.")
        return

    # --- 8. Build and Train Model ---
    print("Building LSTM model...")
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Starting model training...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    print("Training complete. Model saved to stock_predictor.keras")

if __name__ == "__main__":
    # --- Clean up old files before training ---
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(SCALER_X_PATH):
        os.remove(SCALER_X_PATH)
    if os.path.exists(SCALER_Y_PATH):
        os.remove(SCALER_Y_PATH)
    if os.path.exists(SCALER_FEATURES_PATH):
        os.remove(SCALER_FEATURES_PATH)
        
    train()

