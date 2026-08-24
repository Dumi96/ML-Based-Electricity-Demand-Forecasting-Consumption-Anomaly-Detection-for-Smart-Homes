import pandas as pd

from .calendar_features import (
    add_calendar_features,
    add_cyclical_features,
)

from .lag_features import (
    add_lag_features,
)

from .rolling_features import (
    add_rolling_features,
)


def build_features(
    df: pd.DataFrame,
    target_column: str = "demand_kw",
) -> pd.DataFrame:
    """
    Build the complete feature set for
    electricity demand forecasting.
    """

    result = df.copy()

    result = add_calendar_features(result)

    result = add_cyclical_features(result)

    result = add_lag_features(
        result,
        target_column=target_column,
    )

    result = add_rolling_features(
        result,
        target_column=target_column,
    )

    return result