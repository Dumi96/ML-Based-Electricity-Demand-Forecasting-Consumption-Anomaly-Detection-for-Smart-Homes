import pandas as pd


DEFAULT_WINDOWS = [
    3,
    6,
    12,
    24,
    48,
    168,
]


def add_rolling_features(
    df: pd.DataFrame,
    target_column: str = "demand_kw",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling statistics based on historical demand.

    Shift(1) is applied before rolling to prevent
    the current target from leaking into its own features.
    """

    result = df.copy()

    if target_column not in result.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )

    if windows is None:
        windows = DEFAULT_WINDOWS

    historical = result[target_column].shift(1)

    for window in windows:

        result[f"rolling_mean_{window}h"] = (
            historical
            .rolling(window)
            .mean()
        )

        result[f"rolling_std_{window}h"] = (
            historical
            .rolling(window)
            .std()
        )

    return result