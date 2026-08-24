import pandas as pd

from src.data.preprocessing import (
    create_timestamp,
    convert_numeric_columns,
    sort_by_timestamp,
    add_missing_flags,
)


def test_create_timestamp():

    df = pd.DataFrame({
        "Date": ["16/12/2006"],
        "Time": ["17:24:00"],
    })

    result = create_timestamp(df)

    assert "timestamp" in result.columns
    assert result["timestamp"].notna().all()


def test_numeric_conversion():

    df = pd.DataFrame({
        "Global_active_power": ["2.5", "?"],
        "Global_reactive_power": ["0.1", "?"],
        "Voltage": ["230.5", "?"],
        "Global_intensity": ["5.2", "?"],
        "Sub_metering_1": ["1.0", "?"],
        "Sub_metering_2": ["2.0", "?"],
        "Sub_metering_3": ["3.0", "?"],
    })

    result = convert_numeric_columns(df)

    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for column in numeric_columns:

        assert pd.api.types.is_numeric_dtype(
            result[column]
        )

        assert pd.isna(
            result[column].iloc[1]
        )

        assert result[column].iloc[0] is not None


def test_timestamp_sorting():

    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2007-01-02",
            "2007-01-01",
        ])
    })

    result = sort_by_timestamp(df)

    assert result[
        "timestamp"
    ].is_monotonic_increasing


def test_missing_flags():

    df = pd.DataFrame({
        "Global_active_power": [1.0, None],
        "Global_reactive_power": [0.1, None],
        "Voltage": [230.0, None],
        "Global_intensity": [5.0, None],
        "Sub_metering_1": [1.0, None],
        "Sub_metering_2": [2.0, None],
        "Sub_metering_3": [3.0, None],
    })

    result = add_missing_flags(df)

    assert result[
        "is_measurement_missing"
    ].tolist() == [False, True]

    assert result[
        "target_available"
    ].tolist() == [True, False]