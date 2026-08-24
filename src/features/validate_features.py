import pandas as pd


def validate_features(
    df: pd.DataFrame,
    target_column: str = "demand_kw",
) -> dict:
    """
    Validate engineered time-series features.

    Checks:
    - target exists
    - timestamp index
    - chronological ordering
    - duplicate timestamps
    - target availability
    - feature count
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    if not isinstance(
        df.index,
        pd.DatetimeIndex
    ):
        raise TypeError(
            "Feature dataframe must use a DatetimeIndex."
        )

    validation = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_timestamps": int(
            df.index.duplicated().sum()
        ),
        "timestamp_sorted": bool(
            df.index.is_monotonic_increasing
        ),
        "target_missing": int(
            df[target_column].isna().sum()
        ),
        "feature_columns": len(
            df.columns
        ),
    }

    return validation