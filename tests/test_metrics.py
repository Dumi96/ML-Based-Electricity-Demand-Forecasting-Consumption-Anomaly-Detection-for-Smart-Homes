import numpy as np

from src.evaluation.metrics import calculate_metrics


def test_metrics_perfect_prediction():

    y_true = np.array([
        1.0,
        2.0,
        3.0,
        4.0
    ])

    y_pred = np.array([
        1.0,
        2.0,
        3.0,
        4.0
    ])

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["R2"] == 1


def test_metrics_returns_expected_keys():

    y_true = np.array([
        1.0,
        2.0,
        3.0
    ])

    y_pred = np.array([
        1.2,
        1.8,
        3.1
    ])

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    expected_keys = {
        "MAE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "R2"
    }

    assert set(metrics.keys()) == expected_keys