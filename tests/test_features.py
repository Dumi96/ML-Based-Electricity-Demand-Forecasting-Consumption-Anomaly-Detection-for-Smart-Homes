import pandas as pd

from src.features.build_features import build_features


def create_test_data():

    index = pd.date_range(
        "2024-01-01",
        periods=200,
        freq="h"
    )

    return pd.DataFrame(
        {
            "demand_kw": range(200)
        },
        index=index
    )


def test_feature_pipeline():

    df = create_test_data()

    result = build_features(
        df,
        target_column="demand_kw"
    )

    assert "hour" in result.columns
    assert "day_of_week" in result.columns

    assert "lag_1h" in result.columns
    assert "lag_24h" in result.columns
    assert "lag_168h" in result.columns

    assert "rolling_mean_24h" in result.columns
    assert "rolling_std_24h" in result.columns


def test_feature_pipeline_preserves_rows():

    df = create_test_data()

    result = build_features(
        df,
        target_column="demand_kw"
    )

    assert len(result) == len(df)


def test_lag_uses_previous_value():

    df = create_test_data()

    result = build_features(
        df,
        target_column="demand_kw"
    )

    assert result.iloc[1]["lag_1h"] == 0
    assert result.iloc[2]["lag_1h"] == 1