from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.build_features import build_features


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "tuned_xgboost_forecaster.joblib"
)

# Change this only if your processed dataset has a different path.
DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "hourly_energy.csv"
)

FEATURE_NAMES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "is_night",
    "is_evening_peak",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_12h",
    "lag_24h",
    "lag_48h",
    "lag_72h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_std_3h",
    "rolling_mean_6h",
    "rolling_std_6h",
    "rolling_mean_12h",
    "rolling_std_12h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_48h",
    "rolling_std_48h",
    "rolling_mean_168h",
    "rolling_std_168h",
]


def metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "MAE": mean_absolute_error(
            actual,
            predicted,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "R2": r2_score(
            actual,
            predicted,
        ),
    }


def recursive_forecast(model, history, horizon):

    history = list(history)

    predictions = []

    for _ in range(horizon):

        end_timestamp = pd.Timestamp(
            "2000-01-01"
        )

        index = pd.date_range(
            end=end_timestamp,
            periods=len(history),
            freq="1h",
        )

        temp = pd.DataFrame(
            {
                "demand_kw": history
            },
            index=index,
        )

        next_timestamp = (
            index[-1]
            + pd.Timedelta(hours=1)
        )

        temp.loc[
            next_timestamp,
            "demand_kw"
        ] = np.nan

        features = build_features(
            temp,
            target_column="demand_kw",
        )

        row = features.loc[
            [next_timestamp],
            FEATURE_NAMES,
        ]

        if row.isna().any().any():
            raise ValueError(
                "NaN feature detected during "
                "recursive forecasting."
            )

        prediction = float(
            model.predict(row)[0]
        )

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return predictions


def main():

    print("=" * 70)
    print("SMARTENERGY AI - RECURSIVE FORECAST VALIDATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print("\nLoading data...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        "Columns:",
        list(df.columns)
    )

    # Detect timestamp column
    timestamp_candidates = [
        "timestamp",
        "datetime",
        "date",
        "DateTime",
    ]

    timestamp_column = None

    for column in timestamp_candidates:

        if column in df.columns:
            timestamp_column = column
            break

    if timestamp_column is None:
        raise ValueError(
            "Could not find timestamp column."
        )

    if "demand_kw" not in df.columns:
        raise ValueError(
            "Dataset must contain demand_kw column."
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column]
    )

    df = df.sort_values(
        timestamp_column
    )

    df = df.set_index(
        timestamp_column
    )

    # --------------------------------------------------
    # Build features
    # --------------------------------------------------

    print("\nBuilding features...")

    feature_df = build_features(
        df,
        target_column="demand_kw",
    )

    feature_df = feature_df.dropna(
        subset=FEATURE_NAMES
        + ["demand_kw"]
    )

    print(
        f"Usable rows: {len(feature_df):,}"
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading XGBoost model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model:",
        type(model).__name__
    )

    # --------------------------------------------------
    # Use final portion as evaluation period
    # --------------------------------------------------

    test_size = 5000

    test_start = (
        len(feature_df)
        - test_size
    )

    test_df = feature_df.iloc[
        test_start:
    ]

    print(
        f"\nEvaluation rows: {len(test_df):,}"
    )

    print(
        "Evaluation period:",
        test_df.index.min(),
        "to",
        test_df.index.max(),
    )

    # --------------------------------------------------
    # Evaluate recursive horizons
    # --------------------------------------------------

    results = []

    horizons = [
        1,
        6,
        12,
        24,
    ]

    for horizon in horizons:

        print(
            f"\nEvaluating {horizon}-hour horizon..."
        )

        errors = []

        # Use windows separated by horizon.
        max_start = (
            len(test_df)
            - horizon
        )

        step = max(
            horizon,
            6,
        )

        for i in range(
            0,
            max_start,
            step,
        ):

            row = test_df.iloc[i]

            timestamp = (
                test_df.index[i]
            )

            history_end = i

            # We need the actual historical
            # demand before the forecast point.
            global_position = (
                test_start
                + history_end
            )

            history_start = max(
                0,
                global_position - 169,
            )

            history = (
                feature_df
                .iloc[
                    history_start:
                    global_position + 1
                ]["demand_kw"]
                .tolist()
            )

            if len(history) < 169:
                continue

            actual = (
                test_df
                .iloc[
                    i + 1:
                    i + horizon + 1
                ]["demand_kw"]
                .tolist()
            )

            if len(actual) != horizon:
                continue

            predictions = recursive_forecast(
                model,
                history,
                horizon,
            )

            errors.append(
                {
                    "actual": actual,
                    "predicted": predictions,
                }
            )

        actual_values = []

        predicted_values = []

        for item in errors:

            actual_values.extend(
                item["actual"]
            )

            predicted_values.extend(
                item["predicted"]
            )

        if not actual_values:
            print(
                "No valid forecast windows."
            )
            continue

        result = metrics(
            actual_values,
            predicted_values,
        )

        result["Horizon"] = (
            f"{horizon}h"
        )

        result["Samples"] = (
            len(actual_values)
        )

        results.append(
            result
        )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        print(
            "\nNo results generated."
        )
        return

    results_df = results_df[
        [
            "Horizon",
            "Samples",
            "MAE",
            "RMSE",
            "R2",
        ]
    ]

    print("\n")
    print("=" * 70)
    print("RECURSIVE FORECAST RESULTS")
    print("=" * 70)

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    report_dir = (
        BASE_DIR
        / "reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        report_dir
        / "recursive_forecast_validation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved results to:\n{output_path}"
    )


if __name__ == "__main__":
    main()