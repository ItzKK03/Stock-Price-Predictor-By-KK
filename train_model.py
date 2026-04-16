"""
Training Pipeline Module.

Main training script for the stock price prediction model.
Combines data collection, feature engineering, sentiment analysis,
and LSTM model training with comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from typing import List, Optional, Any, Dict, Tuple

# Import project-specific modules
from src.data_collection import get_stock_data, get_stock_news
from src.feature_engineering import add_technical_indicators
from src.sentiment_analysis import initialize_sentiment_model, get_daily_sentiment
from src.model_utils import build_model, create_dataset, save_model, save_scaler
from src.data_preprocessing import merge_stock_sentiment, prepare_features
from src.evaluation import (
    evaluate_predictions,
    generate_evaluation_report,
    create_evaluation_dataframe,
    calculate_improvement_vs_baseline
)
from src.logger import get_logger

# Import configuration
from config import settings

logger = get_logger(__name__)


def train() -> None:
    """
    Main training pipeline for the stock price prediction model.

    Orchestrates the following steps:
    1. Initialize sentiment model (FinBERT)
    2. Collect stock data from Yahoo Finance
    3. Add technical indicators (RSI, MA20, MA50)
    4. Fetch and analyze news sentiment
    5. Merge and preprocess data
    6. Scale features and target
    7. Create time-series dataset for LSTM
    8. Build and train the LSTM model
    9. Evaluate model on training data
    10. Save trained model, scalers, and evaluation report

    Returns:
        None. Saves model to settings.model_path and scalers to config paths.
    """
    logger.info(f"Starting training process for {settings.TICKER}...")

    # --- 1. Initialize Models ---
    logger.info("Initializing sentiment model...")
    sentiment_pipeline = initialize_sentiment_model()

    # --- 2. Data Collection ---
    logger.info(f"Fetching data for {settings.TICKER} from {settings.START_DATE} to {settings.END_DATE}...")
    stock_data = get_stock_data(settings.TICKER, settings.START_DATE, settings.END_DATE)

    # --- 3. Feature Engineering ---
    logger.info("Adding technical indicators...")
    stock_data = add_technical_indicators(stock_data)

    # --- 4. Sentiment Analysis ---
    logger.info("Fetching and analyzing news sentiment...")
    daily_sentiment = get_daily_sentiment(settings.TICKER, sentiment_pipeline)

    # --- 5. Data Merging and Preprocessing ---
    logger.info("Combining stock data and sentiment...")
    data = merge_stock_sentiment(stock_data, daily_sentiment)

    # Define features to be used for training
    features: List[str] = settings.FEATURES
    features_for_y: List[str] = [settings.TARGET_FEATURE]  # What we want to predict

    # --- 6. Prepare Features ---
    logger.info("Preparing features for training...")
    data_x, data_y, _ = prepare_features(data, features, settings.TARGET_FEATURE)

    # --- 7. Data Scaling ---
    logger.info("Scaling data...")
    # FIX: Safely fetch SCALER_FEATURE_RANGE from settings, defaulting to (0, 1) if missing
    scale_range = getattr(settings, 'SCALER_FEATURE_RANGE', (0, 1))
    
    scaler_x = MinMaxScaler(feature_range=scale_range)
    scaler_y = MinMaxScaler(feature_range=scale_range)

    scaled_x = scaler_x.fit_transform(data_x)
    scaled_y = scaler_y.fit_transform(data_y)

    # Save scalers and feature list
    save_scaler(scaler_x, settings.scaler_x_path, features)
    save_scaler(scaler_y, settings.scaler_y_path, features_for_y)
    with open(settings.scaler_features_path, 'w') as f:
        json.dump(features, f)
    logger.info(f"Feature list saved to {settings.scaler_features_path}")

    # --- 8. Create Time-Series Dataset ---
    logger.info(f"Creating time-series dataset with time_step={settings.TIME_STEP}...")
    X_train, y_train = create_dataset(scaled_x, scaled_y, settings.TIME_STEP)

    # Reshape X_train for LSTM [samples, time_steps, n_features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))

    if X_train.shape[0] == 0:
        logger.error(f"Not enough data to create training samples. Need at least {settings.TIME_STEP+1} days of data.")
        return

    logger.info(f"Training dataset shape: X={X_train.shape}, y={y_train.shape}")

    # Split into train and validation sets for evaluation
    val_split = settings.VALIDATION_SPLIT
    val_size = int(len(X_train) * val_split)
    train_size = len(X_train) - val_size

    X_train_final = X_train[:train_size]
    y_train_final = y_train[:train_size]
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]

    logger.info(f"Train size: {len(X_train_final)}, Validation size: {len(X_val)}")

    # --- 9. Build and Train Model ---
    logger.info("Building LSTM model...")
    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=settings.LSTM_UNITS,
        dropout_rate=settings.DROPOUT_RATE,
        dense_units=settings.DENSE_UNITS,
        learning_rate=settings.LEARNING_RATE
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        settings.model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=settings.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    logger.info("Starting model training...")
    logger.info(f"Training config: epochs={settings.EPOCHS}, batch_size={settings.BATCH_SIZE}, "
                f"validation_split={settings.VALIDATION_SPLIT}")

    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=settings.EPOCHS,
        batch_size=settings.BATCH_SIZE,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    logger.info("Training complete. Evaluating model...")

    # --- 10. Model Evaluation ---
    logger.info("Generating predictions for evaluation...")

    # Predict on validation set
    y_val_pred_scaled = model.predict(X_val)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
    y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1, 1))

    # Flatten for evaluation
    y_val_pred_flat = y_val_pred.flatten()
    y_val_actual_flat = y_val_actual.flatten()

    # Calculate metrics
    metrics = evaluate_predictions(y_val_actual_flat, y_val_pred_flat)

    # Calculate improvement over naive baseline
    improvement = calculate_improvement_vs_baseline(
        y_val_actual_flat,
        y_val_pred_flat,
        metric='mae'
    )

    # Generate report
    report = generate_evaluation_report(metrics, model_name=f"LSTM Stock Predictor ({settings.TICKER})")

    if improvement is not None:
        report += f"\nImprovement over Naive Baseline (MAE): {improvement:.2f}%\n"

    # Log and print report
    logger.info(report)
    print(report)  # Print to console so user sees it

    # Create evaluation DataFrame
    eval_df = create_evaluation_dataframe(
        y_val_actual_flat,
        y_val_pred_flat,
        dates=data.index[-len(y_val_actual_flat):] if len(data) >= len(y_val_actual_flat) else None
    )

    # Save evaluation results
    eval_csv_path = settings.model_path.replace('.keras', '_evaluation.csv')
    eval_df.to_csv(eval_csv_path)
    logger.info(f"Evaluation results saved to {eval_csv_path}")

    # Save metrics as JSON
    metrics_json_path = settings.model_path.replace('.keras', '_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_json_path}")

    # Save full history
    history_df = pd.DataFrame(history.history)
    history_csv_path = settings.model_path.replace('.keras', '_training_history.csv')
    history_df.to_csv(history_csv_path)
    logger.info(f"Training history saved to {history_csv_path}")

    logger.info(f"Model saved to {settings.model_path}")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    # --- Clean up old files before training ---
    for path in [
        settings.model_path,
        settings.scaler_x_path,
        settings.scaler_y_path,
        settings.scaler_features_path
    ]:
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Removed existing file: {path}")

    train()