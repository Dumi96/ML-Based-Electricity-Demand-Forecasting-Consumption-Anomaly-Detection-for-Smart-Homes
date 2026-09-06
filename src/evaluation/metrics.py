import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate forecasting evaluation metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    # Avoid division by zero in MAPE
    non_zero = y_true != 0

    mape = (
        np.mean(
            np.abs(
                (
                    y_true[non_zero]
                    - y_pred[non_zero]
                )
                / y_true[non_zero]
            )
        )
        * 100
    )

    smape = (
        np.mean(
            2
            * np.abs(y_pred - y_true)
            / (
                np.abs(y_true)
                + np.abs(y_pred)
                + 1e-8
            )
        )
        * 100
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "R2": r2,
    }