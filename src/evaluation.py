"""
Model Evaluation Module.

Comprehensive evaluation metrics and visualization for stock price prediction models.
Includes MAE, RMSE, MAPE, directional accuracy, and plotting utilities.

Usage:
    from src.evaluation import evaluate_predictions, plot_predictions

    metrics = evaluate_predictions(y_true, y_pred)
    fig = plot_predictions(y_true, y_pred)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

from src.logger import get_logger


logger = get_logger(__name__)


def calculate_mae(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE value.

    Raises:
        ValueError: If inputs have mismatched lengths or are empty.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    mae = mean_absolute_error(y_true, y_pred)
    logger.debug(f"MAE calculated: {mae:.6f}")
    return mae


def calculate_rmse(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE value.

    Raises:
        ValueError: If inputs have mismatched lengths or are empty.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    logger.debug(f"RMSE calculated: {rmse:.6f}")
    return rmse


def calculate_mape(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Small value to avoid division by zero (default: 1e-10).

    Returns:
        MAPE value as percentage (e.g., 5.2 means 5.2%).

    Raises:
        ValueError: If inputs have mismatched lengths or are empty.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Avoid division by zero
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        logger.warning("All true values are near zero, MAPE may be unreliable")
        return float('inf')

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    logger.debug(f"MAPE calculated: {mape:.4f}%")
    return mape


def calculate_directional_accuracy(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculate Directional Accuracy (percentage of correct direction predictions).

    Measures how often the model correctly predicts whether the price will
    go up or down compared to the previous day.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Directional accuracy as percentage (e.g., 65.5 means 65.5% accurate).

    Raises:
        ValueError: If inputs have mismatched lengths or have less than 2 samples.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) < 2:
        raise ValueError("Need at least 2 samples for directional accuracy")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate direction of change (1 = up, -1 = down, 0 = flat)
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    # Calculate accuracy
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    logger.debug(f"Directional accuracy calculated: {accuracy:.2f}%")
    return accuracy


def calculate_r2(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        R² value (higher is better, 1.0 = perfect fit).

    Raises:
        ValueError: If inputs have mismatched lengths or are empty.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) < 2:
        raise ValueError("Need at least 2 samples for R² calculation")

    r2 = r2_score(y_true, y_pred)
    logger.debug(f"R² calculated: {r2:.6f}")
    return r2


def _validate_inputs(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert inputs to numpy arrays.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Tuple of (y_true, y_pred) as numpy arrays.

    Raises:
        ValueError: If inputs are None or have mismatched lengths.
    """
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred cannot be None")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Input length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples"
        )

    return y_true, y_pred


def evaluate_predictions(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for predictions.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with all metrics:
            - 'mae': Mean Absolute Error
            - 'rmse': Root Mean Squared Error
            - 'mape': Mean Absolute Percentage Error (%)
            - 'directional_accuracy': Directional Accuracy (%)
            - 'r2': R-squared

    Raises:
        ValueError: If inputs are invalid.
    """
    logger.info("Calculating evaluation metrics...")

    y_true, y_pred = _validate_inputs(y_true, y_pred)

    metrics: Dict[str, float] = {
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'directional_accuracy': calculate_directional_accuracy(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred)
    }

    # Log summary
    logger.info(
        f"Evaluation complete: MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%, "
        f"Directional Accuracy={metrics['directional_accuracy']:.2f}%, "
        f"R²={metrics['r2']:.4f}"
    )

    return metrics


def generate_evaluation_report(
    metrics: Dict[str, float],
    model_name: str = "Model"
) -> str:
    """
    Generate a formatted evaluation report string.

    Args:
        metrics: Dictionary of metrics from evaluate_predictions().
        model_name: Name of the model for the report.

    Returns:
        Formatted report string.
    """
    report = f"""
{'='*60}
{model_name} - Evaluation Report
{'='*60}

Regression Metrics:
  - Mean Absolute Error (MAE):     {metrics.get('mae', 'N/A'):.4f}
  - Root Mean Squared Error (RMSE): {metrics.get('rmse', 'N/A'):.4f}
  - Mean Absolute Percentage Error: {metrics.get('mape', 'N/A'):.2f}%
  - R-squared (R²):                {metrics.get('r2', 'N/A'):.4f}

Classification Metric:
  - Directional Accuracy:          {metrics.get('directional_accuracy', 'N/A'):.2f}%

{'='*60}
"""
    return report


def create_evaluation_dataframe(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    dates: Optional[pd.DatetimeIndex] = None
) -> pd.DataFrame:
    """
    Create a DataFrame comparing predictions vs actual values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        dates: Optional DatetimeIndex for the rows.

    Returns:
        DataFrame with columns: 'Actual', 'Predicted', 'Error', 'Error_%'.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })

    df['Error'] = df['Predicted'] - df['Actual']
    df['Error_%'] = (df['Error'] / df['Actual'].replace(0, np.nan)) * 100

    if dates is not None:
        df.index = dates[:len(df)]

    return df


def calculate_improvement_vs_baseline(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    baseline_pred: Optional[Union[np.ndarray, List[float]]] = None,
    metric: str = 'mae'
) -> Optional[float]:
    """
    Calculate improvement over a naive baseline (last value prediction).

    Args:
        y_true: Ground truth values.
        y_pred: Model predictions.
        baseline_pred: Optional baseline predictions. If None, uses
                       naive forecast (shift y_true by 1).
        metric: Metric to compare ('mae', 'rmse', or 'mape').

    Returns:
        Improvement percentage (positive = model is better than baseline).
        Returns None if baseline cannot be calculated.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if baseline_pred is None:
        # Naive baseline: predict previous value
        baseline_pred = np.roll(y_true, 1)
        baseline_pred[0] = y_true[0]  # First value has no previous

    metric_func = {
        'mae': calculate_mae,
        'rmse': calculate_rmse,
        'mape': calculate_mape
    }.get(metric.lower())

    if metric_func is None:
        logger.error(f"Unknown metric: {metric}")
        return None

    model_score = metric_func(y_true, y_pred)
    baseline_score = metric_func(y_true, baseline_pred)

    if baseline_score == 0:
        return None

    improvement = ((baseline_score - model_score) / baseline_score) * 100
    logger.info(
        f"Model {metric.upper()}={model_score:.4f} vs Baseline={baseline_score:.4f}, "
        f"Improvement={improvement:.2f}%"
    )

    return improvement
