import pandas as pd

from .preprocessing import NUMERIC_COLUMNS


def validate_timestamps(
    df: pd.DataFrame,
) -> dict:
    """
    Validate timestamp quality.
    """

    return {
        "invalid_timestamps": int(
            df["timestamp"].isna().sum()
        ),
        "duplicate_timestamps": int(
            df["timestamp"].duplicated().sum()
        ),
        "timestamp_sorted": bool(
            df["timestamp"].is_monotonic_increasing
        ),
    }


def validate_missing_measurements(
    df: pd.DataFrame,
) -> dict:
    """
    Validate missing electricity measurements.
    """

    measurement_missing = (
        df[NUMERIC_COLUMNS]
        .isna()
        .all(axis=1)
    )

    return {
        "measurement_outage_rows": int(
            measurement_missing.sum()
        ),
        "measurement_outage_percentage": round(
            measurement_missing.mean() * 100,
            2
        ),
        "target_missing": int(
            df["Global_active_power"].isna().sum()
        ),
    }
def validate_dataset(
    df: pd.DataFrame,
) -> dict:
    """
    Run the complete dataset validation.
    """

    timestamp_results = validate_timestamps(df)

    missing_results = validate_missing_measurements(df)

    results = {
        "rows": len(df),
        "columns": len(df.columns),
        **timestamp_results,
        **missing_results,
    }

    return results