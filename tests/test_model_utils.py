"""
Tests for Model Utilities Module.

Tests build_model, create_dataset, and scaler functions.
"""

import pytest
import numpy as np
import joblib
import json
import os
from src.model_utils import (
    build_model,
    create_dataset,
    save_scaler,
    load_scaler,
    validate_input_shape
)


class TestValidateInputShape:
    """Tests for validate_input_shape helper function."""

    def test_valid_shape(self):
        """Test with valid input shape."""
        assert validate_input_shape((60, 5)) is True
        assert validate_input_shape((10, 1)) is True

    def test_invalid_shape_none(self):
        """Test with None as input."""
        assert validate_input_shape(None) is False

    def test_invalid_shape_wrong_length(self):
        """Test with wrong tuple length."""
        assert validate_input_shape((60,)) is False
        assert validate_input_shape((60, 5, 3)) is False

    def test_invalid_shape_zero_values(self):
        """Test with zero values."""
        assert validate_input_shape((0, 5)) is False
        assert validate_input_shape((60, 0)) is False

    def test_invalid_shape_negative(self):
        """Test with negative values."""
        assert validate_input_shape((-1, 5)) is False


class TestBuildModel:
    """Tests for build_model function."""

    def test_invalid_input_shape(self):
        """Test with invalid input shape."""
        with pytest.raises(ValueError, match="input_shape must be a tuple"):
            build_model(None)  # type: ignore

    def test_invalid_lstm_units(self):
        """Test with invalid LSTM units."""
        with pytest.raises(ValueError, match="lstm_units must be positive"):
            build_model((10, 5), lstm_units=0)

    def test_invalid_dropout_rate(self):
        """Test with invalid dropout rate."""
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            build_model((10, 5), dropout_rate=1.5)

    def test_invalid_dense_units(self):
        """Test with invalid dense units."""
        with pytest.raises(ValueError, match="dense_units must be positive"):
            build_model((10, 5), dense_units=-1)

    def test_invalid_learning_rate(self):
        """Test with invalid learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            build_model((10, 5), learning_rate=0)

    def test_build_default_model(self):
        """Test building model with default parameters."""
        model = build_model((60, 5))

        assert model is not None
        assert len(model.layers) == 5  # LSTM, Dropout, LSTM, Dropout, Dense, Dense

    def test_build_custom_model(self):
        """Test building model with custom parameters."""
        model = build_model(
            input_shape=(30, 10),
            lstm_units=64,
            dropout_rate=0.3,
            dense_units=32,
            learning_rate=0.01
        )

        assert model is not None
        assert model.input_shape == (None, 30, 10)

    def test_model_compiles_successfully(self):
        """Test that model is properly compiled."""
        model = build_model((60, 5))

        assert model.optimizer is not None
        assert model.loss is not None


class TestCreateDataset:
    """Tests for create_dataset function."""

    def test_none_inputs_raises_error(self):
        """Test that None inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            create_dataset(None, np.array([1, 2, 3]))  # type: ignore

    def test_non_array_inputs_raises_error(self):
        """Test that non-array inputs raise ValueError."""
        with pytest.raises(ValueError, match="must be numpy arrays"):
            create_dataset([1, 2, 3], [1, 2, 3])  # type: ignore

    def test_invalid_time_step(self):
        """Test that invalid time step raises ValueError."""
        data = np.random.rand(100, 5)
        target = np.random.rand(100, 1)

        with pytest.raises(ValueError, match="time_step must be positive"):
            create_dataset(data, target, time_step=0)

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        data_x = np.random.rand(100, 5)
        data_y = np.random.rand(50, 1)

        with pytest.raises(ValueError, match="must have the same length"):
            create_dataset(data_x, data_y)

    def test_dataset_too_small_raises_error(self):
        """Test that dataset smaller than time_step raises ValueError."""
        data = np.random.rand(5, 5)
        target = np.random.rand(5, 1)

        with pytest.raises(ValueError, match="must have more than time_step"):
            create_dataset(data, target, time_step=10)

    def test_creates_correct_shape(self, sample_scaled_data: tuple):
        """Test that dataset has correct shape."""
        X, y = sample_scaled_data
        time_step = 5

        X_lstm, y_lstm = create_dataset(X, y, time_step)

        expected_samples = len(X) - time_step
        assert X_lstm.shape == (expected_samples, time_step, X.shape[1])
        assert y_lstm.shape == (expected_samples,)

    def test_creates_sequential_samples(self):
        """Test that sequential structure is preserved."""
        data = np.arange(20).reshape(10, 2).astype(float)
        target = np.arange(10, 20).reshape(10, 1).astype(float)

        X, y = create_dataset(data, target, time_step=3)

        # First sample should contain first 3 time steps
        assert X[0].shape == (3, 2)
        np.testing.assert_array_equal(X[0], data[:3])
        assert y[0] == target[3, 0]  # Target is next value


class TestSaveLoadScaler:
    """Tests for save_scaler and load_scaler functions."""

    def test_save_none_scaler_raises_error(self, temp_scaler_path: str):
        """Test that saving None scaler raises ValueError."""
        with pytest.raises(ValueError, match="Scaler cannot be None"):
            save_scaler(None, temp_scaler_path, ['feature1'])

    def test_save_empty_path_raises_error(self):
        """Test that saving to empty path raises ValueError."""
        scaler = joblib.__class__ if hasattr(joblib, '__class__') else None  # type: ignore
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        with pytest.raises(ValueError, match="Scaler path cannot be empty"):
            save_scaler(scaler, "", ['feature1'])

    def test_save_empty_features_raises_error(self, temp_scaler_path: str):
        """Test that saving with empty features raises ValueError."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        with pytest.raises(ValueError, match="Features must be a non-empty list"):
            save_scaler(scaler, temp_scaler_path, [])

    def test_save_and_load_scaler(self, temp_scaler_path: str):
        """Test saving and loading a scaler."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features = ['feature1', 'feature2', 'feature3']

        # Save
        save_scaler(scaler, temp_scaler_path, features)

        # Verify files exist
        assert os.path.exists(temp_scaler_path)
        features_path = temp_scaler_path.replace('.joblib', '_features.json')
        assert os.path.exists(features_path)

        # Load
        loaded_scaler, loaded_features = load_scaler(temp_scaler_path)

        assert loaded_scaler is not None
        assert loaded_features == features

    def test_load_nonexistent_scaler_raises_error(self, tmp_path):
        """Test that loading nonexistent scaler raises FileNotFoundError."""
        fake_path = str(tmp_path / "nonexistent.joblib")

        with pytest.raises(FileNotFoundError, match="Scaler file not found"):
            load_scaler(fake_path)

    def test_load_scaler_missing_features(self, temp_scaler_path: str):
        """Test loading scaler when features file is missing."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        # Save scaler manually without features
        joblib.dump(scaler, temp_scaler_path)

        # Load should still work, features should be None
        loaded_scaler, loaded_features = load_scaler(temp_scaler_path)

        assert loaded_scaler is not None
        assert loaded_features is None
