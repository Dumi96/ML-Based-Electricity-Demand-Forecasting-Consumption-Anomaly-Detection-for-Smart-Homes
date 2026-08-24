import pandas as pd


DEFAULT_LAGS = [
    1,
    2,
    3,
    12,
    24,
    48,
    72,
    168,
]


def add_lag_features(
    df: pd.DataFrame,
    target_column: str = "demand_kw",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add historical demand lag features.

    Lags are measured in hours because the dataset
    has been aggregated to hourly frequency.
    """

    result = df.copy()

    if target_column not in result.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )

    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a pandas DatetimeIndex."
        )

    if lags is None:
        lags = DEFAULT_LAGS

    for lag in lags:
        result[f"lag_{lag}h"] = (
            result[target_column].shift(lag)
        )

    return result