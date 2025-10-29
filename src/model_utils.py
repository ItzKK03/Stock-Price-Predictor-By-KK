import pandas as pd
import numpy as np
import joblib
import json
import os
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Constants ---
MODEL_PATH = 'stock_predictor.keras'
SCALER_X_PATH = 'scaler_x.joblib'
SCALER_Y_PATH = 'scaler_y.joblib'
SCALER_FEATURES_PATH = 'scaler_features.json'

# --- Model Architecture ---
def build_model(input_shape):
    """Builds and compiles the LSTM model."""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Model/Scaler Saving & Loading ---

def save_model(model, path):
    """Saves the Keras model."""
    model.save(path)

def load_model(path):
    """Loads the Keras model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return keras_load_model(path)

def save_scaler(scaler, scaler_path, features):
    """Saves a scaler and its associated features."""
    joblib.dump(scaler, scaler_path)
    # Save features in a separate .json file
    features_path = scaler_path.replace('.joblib', '_features.json')
    with open(features_path, 'w') as f:
        json.dump(features, f)

def load_scaler(scaler_path):
    """Loads a scaler and its associated features."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
    scaler = joblib.load(scaler_path)
    
    # Load associated features
    features_path = scaler_path.replace('.joblib', '_features.json')
    features = None
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = json.load(f)
            
    return scaler, features # This correctly returns two values

# --- Data Preparation ---
def create_dataset(dataset_x, dataset_y, time_step=1):
    """Converts time-series data into samples for LSTM."""
    dataX, dataY = [], []
    for i in range(len(dataset_x) - time_step):
        a = dataset_x[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset_y[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

