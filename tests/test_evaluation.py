"""
Tests for Evaluation Module.

Tests all evaluation metrics and helper functions.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_directional_accuracy,
    calculate_r2,
    evaluate_predictions,
    generate_evaluation_report,
    create_evaluation_dataframe,
    calculate_improvement_vs_baseline,
    _validate_inputs
)


class TestValidateInputs:
    """Tests for _validate_inputs helper function."""

    def test_none_inputs_raises_error(self):
        """Test that None inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            _validate_inputs(None, [1, 2, 3])

        with pytest.raises(ValueError, match="cannot be None"):
            _validate_inputs([1, 2, 3], None)

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="Input length mismatch"):
            _validate_inputs([1, 2, 3], [1, 2])

    def test_valid_inputs_returned_as_arrays(self):
        """Test that valid inputs are returned as numpy arrays."""
        y_true, y_pred = _validate_inputs([1, 2, 3], [1.1, 2.1, 3.1])

        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        np.testing.assert_array_equal(y_true, [1, 2, 3])


class TestCalculateMAE:
    """Tests for calculate_mae function."""

    def test_empty_inputs_raises_error(self):
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_mae([], [])

    def test_perfect_predictions(self):
        """Test MAE with perfect predictions."""
        mae = calculate_mae([1, 2, 3], [1, 2, 3])
        assert mae == 0.0

    def test_constant_error(self):
        """Test MAE with constant error."""
        mae = calculate_mae([1, 2, 3], [2, 3, 4])
        assert mae == 1.0

    def test_negative_values(self):
        """Test MAE with negative values."""
        mae = calculate_mae([-1, -2, -3], [-1, -2, -3])
        assert mae == 0.0


class TestCalculateRMSE:
    """Tests for calculate_rmse function."""

    def test_empty_inputs_raises_error(self):
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_rmse([], [])

    def test_perfect_predictions(self):
        """Test RMSE with perfect predictions."""
        rmse = calculate_rmse([1, 2, 3], [1, 2, 3])
        assert rmse == 0.0

    def test_constant_error(self):
        """Test RMSE with constant error."""
        rmse = calculate_rmse([1, 2, 3], [2, 3, 4])
        assert rmse == 1.0

    def test_penalizes_large_errors(self):
        """Test that RMSE penalizes large errors more than MAE."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 10]  # Large error on last value

        rmse = calculate_rmse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)

        assert rmse > mae  # RMSE should be larger due to squaring


class TestCalculateMAPE:
    """Tests for calculate_mape function."""

    def test_empty_inputs_raises_error(self):
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_mape([], [])

    def test_perfect_predictions(self):
        """Test MAPE with perfect predictions."""
        mape = calculate_mape([100, 200, 300], [100, 200, 300])
        assert mape == 0.0

    def test_ten_percent_error(self):
        """Test MAPE with 10% error."""
        mape = calculate_mape([100, 100], [110, 110])
        assert mape == pytest.approx(10.0, rel=0.01)

    def test_returns_percentage(self):
        """Test that MAPE returns percentage value."""
        mape = calculate_mape([100, 200], [105, 210])
        assert mape > 0
        assert mape < 100  # Should be percentage


class TestCalculateDirectionalAccuracy:
    """Tests for calculate_directional_accuracy function."""

    def test_insufficient_samples_raises_error(self):
        """Test that less than 2 samples raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            calculate_directional_accuracy([1], [1])

    def test_perfect_direction_prediction(self):
        """Test perfect directional accuracy."""
        y_true = [1, 2, 3, 4, 5]  # Always going up
        y_pred = [1, 2, 3, 4, 5]  # Always going up

        accuracy = calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 100.0

    def test_wrong_direction(self):
        """Test 0% directional accuracy."""
        y_true = [1, 2, 3, 4, 5]  # Always going up
        y_pred = [1, 0, -1, -2, -3]  # Always going down

        accuracy = calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 0.0

    def test_mixed_accuracy(self):
        """Test mixed directional accuracy."""
        y_true = [1, 2, 1, 2, 1]  # Up, Down, Up, Down
        y_pred = [1, 2, 1, 0, 1]  # Up, Down, Down, Up

        accuracy = calculate_directional_accuracy(y_true, y_pred)
        assert 0 <= accuracy <= 100


class TestCalculateR2:
    """Tests for calculate_r2 function."""

    def test_insufficient_samples_raises_error(self):
        """Test that less than 2 samples raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            calculate_r2([1], [1])

    def test_perfect_fit(self):
        """Test R² with perfect fit."""
        r2 = calculate_r2([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert r2 == pytest.approx(1.0, abs=0.001)

    def test_reasonable_fit(self):
        """Test R² with reasonable fit."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]

        r2 = calculate_r2(y_true, y_pred)
        assert r2 > 0.9  # Should be very high


class TestEvaluatePredictions:
    """Tests for evaluate_predictions function."""

    def test_returns_all_metrics(self, sample_predictions: tuple):
        """Test that all metrics are returned."""
        y_true, y_pred = sample_predictions

        metrics = evaluate_predictions(y_true, y_pred)

        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'directional_accuracy' in metrics
        assert 'r2' in metrics

    def test_metrics_are_numeric(self, sample_predictions: tuple):
        """Test that all metrics are numeric."""
        y_true, y_pred = sample_predictions

        metrics = evaluate_predictions(y_true, y_pred)

        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['rmse'], float)
        assert isinstance(metrics['mape'], float)
        assert isinstance(metrics['directional_accuracy'], float)
        assert isinstance(metrics['r2'], float)


class TestGenerateEvaluationReport:
    """Tests for generate_evaluation_report function."""

    def test_report_contains_model_name(self):
        """Test that report contains model name."""
        metrics = {'mae': 1.0, 'rmse': 2.0, 'mape': 5.0, 'directional_accuracy': 60.0, 'r2': 0.9}

        report = generate_evaluation_report(metrics, model_name="Test Model")

        assert "Test Model" in report

    def test_report_contains_all_metrics(self):
        """Test that report contains all metrics."""
        metrics = {'mae': 1.0, 'rmse': 2.0, 'mape': 5.0, 'directional_accuracy': 60.0, 'r2': 0.9}

        report = generate_evaluation_report(metrics, model_name="Test")

        assert "MAE" in report
        assert "RMSE" in report
        assert "MAPE" in report
        assert "Directional Accuracy" in report
        assert "R-squared" in report or "R²" in report


class TestCreateEvaluationDataframe:
    """Tests for create_evaluation_dataframe function."""

    def test_creates_correct_columns(self):
        """Test that DataFrame has correct columns."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]

        df = create_evaluation_dataframe(y_true, y_pred)

        assert 'Actual' in df.columns
        assert 'Predicted' in df.columns
        assert 'Error' in df.columns
        assert 'Error_%' in df.columns

    def test_error_calculation(self):
        """Test that error is calculated correctly."""
        y_true = [10, 20, 30]
        y_pred = [12, 22, 32]

        df = create_evaluation_dataframe(y_true, y_pred)

        assert df['Error'].iloc[0] == 2
        assert df['Error'].iloc[1] == 2
        assert df['Error'].iloc[2] == 2

    def test_with_dates_index(self):
        """Test DataFrame with date index."""
        y_true = [1, 2, 3]
        y_pred = [1.1, 2.1, 3.1]
        dates = pd.date_range('2024-01-01', periods=3)

        df = create_evaluation_dataframe(y_true, y_pred, dates)

        assert isinstance(df.index, pd.DatetimeIndex)


class TestCalculateImprovementVsBaseline:
    """Tests for calculate_improvement_vs_baseline function."""

    def test_model_better_than_baseline(self):
        """Test positive improvement."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]  # Good predictions
        baseline = [1, 1, 1, 1, 1]  # Bad baseline

        improvement = calculate_improvement_vs_baseline(y_true, y_pred, baseline, 'mae')

        assert improvement is not None
        assert improvement > 0  # Model should be better

    def test_model_worse_than_baseline(self):
        """Test negative improvement."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 1, 1, 1, 1]  # Bad predictions
        baseline = [1.1, 2.1, 3.1, 4.1, 5.1]  # Good baseline

        improvement = calculate_improvement_vs_baseline(y_true, y_pred, baseline, 'mae')

        assert improvement is not None
        assert improvement < 0  # Model should be worse

    def test_naive_baseline(self):
        """Test with naive baseline (no baseline provided)."""
        y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_pred = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Slightly shifted

        improvement = calculate_improvement_vs_baseline(y_true, y_pred, metric='mae')

        # Should calculate improvement over naive (previous value) baseline
        assert improvement is not None
