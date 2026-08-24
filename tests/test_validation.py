import pandas as pd

from src.data.validation import (
    validate_timestamps,
    validate_missing_measurements,
)


def test_timestamp_validation():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2007-01-01 00:00:00",
            "2007-01-01 00:01:00",
            "2007-01-01 00:02:00",
        ])
    })

    result = validate_timestamps(df)

    assert result["invalid_timestamps"] == 0
    assert result["duplicate_timestamps"] == 0
    assert result["timestamp_sorted"] is True


def test_missing_measurement_validation():
    df = pd.DataFrame({
        "Global_active_power": [1.0, None],
        "Global_reactive_power": [0.1, None],
        "Voltage": [230.0, None],
        "Global_intensity": [5.0, None],
        "Sub_metering_1": [1.0, None],
        "Sub_metering_2": [2.0, None],
        "Sub_metering_3": [3.0, None],
    })

    df["timestamp"] = pd.to_datetime([
        "2007-01-01 00:00:00",
        "2007-01-01 00:01:00",
    ])

    result = validate_missing_measurements(df)

    assert result["measurement_outage_rows"] == 1
    assert result["target_missing"] == 1