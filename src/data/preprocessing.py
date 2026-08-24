import pandas as pd


NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def create_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Date and Time columns into a timestamp column.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
    df["Date"].astype(str)
    + " "
    + df["Time"].astype(str),
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

    return df


def convert_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert electricity measurement columns to numeric.
    Invalid values such as '?' become NaN.
    """

    df = df.copy()

    for column in NUMERIC_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def sort_by_timestamp(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort the dataset chronologically.
    """

    df = df.copy()

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df
def add_missing_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add flags identifying rows where all electricity
    measurements are unavailable.
    """

    df = df.copy()

    df["is_measurement_missing"] = (
        df[NUMERIC_COLUMNS]
        .isna()
        .all(axis=1)
    )

    df["target_available"] = (
        df["Global_active_power"].notna()
    )

    return df

def preprocess_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.
    """

    df = create_timestamp(df)

    df = convert_numeric_columns(df)

    df = sort_by_timestamp(df)

    df = add_missing_flags(df)

    return df