import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features to an hourly electricity dataset.

    Expected:
        DataFrame indexed by a DatetimeIndex.
    """

    result = df.copy()

    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a pandas DatetimeIndex."
        )

    result["hour"] = result.index.hour
    result["day_of_week"] = result.index.dayofweek
    result["day_of_month"] = result.index.day
    result["month"] = result.index.month
    result["week_of_year"] = result.index.isocalendar().week.astype(int)

    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    result["is_night"] = (
        (result["hour"] < 6) |
        (result["hour"] >= 22)
    ).astype(int)

    result["is_evening_peak"] = (
        result["hour"].between(18, 22)
    ).astype(int)

    return result
import numpy as np


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode cyclical time features using sine/cosine transformations.
    """

    result = df.copy()

    if "hour" not in result.columns:
        result["hour"] = result.index.hour

    if "day_of_week" not in result.columns:
        result["day_of_week"] = result.index.dayofweek

    result["hour_sin"] = np.sin(
        2 * np.pi * result["hour"] / 24
    )

    result["hour_cos"] = np.cos(
        2 * np.pi * result["hour"] / 24
    )

    result["day_sin"] = np.sin(
        2 * np.pi * result["day_of_week"] / 7
    )

    result["day_cos"] = np.cos(
        2 * np.pi * result["day_of_week"] / 7
    )

    return result