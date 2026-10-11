"""Shared local inference functions for the SmartEnergy AI project."""

from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import build_features


BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_MODEL_PATH = BASE_DIR / "models" / "tuned_xgboost_forecaster.joblib"
ANOMALY_MODEL_PATH = BASE_DIR / "models" / "isolation_forest_anomaly_detector.joblib"

FEATURE_NAMES = [
    "hour", "day_of_week", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_night", "is_evening_peak",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_12h", "lag_24h", "lag_48h",
    "lag_72h", "lag_168h",
    "rolling_mean_3h", "rolling_std_3h", "rolling_mean_6h",
    "rolling_std_6h", "rolling_mean_12h", "rolling_std_12h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_48h",
    "rolling_std_48h", "rolling_mean_168h", "rolling_std_168h",
]

ANOMALY_FEATURE_NAMES = [
    "demand_kw", "hour", "day_of_week", "is_weekend", "is_night",
    "is_evening_peak", "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_24h", "rolling_std_24h",
]

PUCSL_2026_BILL_REFERENCE = {
    30: 230, 60: 630, 90: 1840, 120: 3280, 150: 5100,
    180: 6420, 210: 9570, 240: 12120, 270: 14670, 300: 17220,
}


def _load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _history_frame(history_end_timestamp, hourly_consumption_kw):
    if len(hourly_consumption_kw) < 169:
        raise ValueError("At least 169 hourly readings are required.")

    values = [float(value) for value in hourly_consumption_kw]
    if not all(pd.notna(value) for value in values):
        raise ValueError("Historical readings must be valid numbers.")

    history_end = pd.Timestamp(history_end_timestamp)
    if pd.isna(history_end):
        raise ValueError("Invalid history-end timestamp.")

    index = pd.date_range(end=history_end, periods=len(values), freq="1h")
    return history_end, pd.DataFrame({"demand_kw": values}, index=index)


def forecast_from_history(history_end_timestamp, hourly_consumption_kw):
    history_end, history_df = _history_frame(
        history_end_timestamp, hourly_consumption_kw
    )
    next_timestamp = history_end + pd.Timedelta(hours=1)

    next_row = pd.DataFrame(
        {"demand_kw": [float("nan")]}, index=[next_timestamp]
    )
    combined = pd.concat([history_df, next_row])
    features = build_features(combined, target_column="demand_kw")
    row = features.loc[[next_timestamp], FEATURE_NAMES]

    if row.isna().any().any():
        missing = row.columns[row.isna().any()].tolist()
        raise ValueError(f"Unable to generate complete forecast features: {missing}")

    model = _load_model(FORECAST_MODEL_PATH)
    prediction = float(model.predict(row)[0])

    return {
        "forecast_timestamp": next_timestamp.isoformat(),
        "predicted_demand_kw": round(prediction, 4),
        "unit": "kW",
        "history_hours_used": len(hourly_consumption_kw),
        "model": "tuned_xgboost_forecaster",
    }


def detect_anomaly(history_end_timestamp, hourly_consumption_kw):
    history_end, history_df = _history_frame(
        history_end_timestamp, hourly_consumption_kw
    )
    features = build_features(history_df, target_column="demand_kw")
    row = features.loc[[history_end], ANOMALY_FEATURE_NAMES]

    if row.isna().any().any():
        missing = row.columns[row.isna().any()].tolist()
        raise ValueError(f"Unable to generate complete anomaly features: {missing}")

    model = _load_model(ANOMALY_MODEL_PATH)
    prediction = int(model.predict(row)[0])
    score = float(model.decision_function(row)[0])
    is_anomaly = prediction == -1

    current = float(hourly_consumption_kw[-1])
    previous = float(hourly_consumption_kw[-2])
    absolute_change = abs(current - previous)

    high_consumption = is_anomaly and current >= 2.855
    sudden_change = is_anomaly and absolute_change >= 1.493

    if high_consumption and sudden_change:
        category = "High Consumption + Sudden Change"
    elif high_consumption:
        category = "High Consumption"
    elif sudden_change:
        category = "Sudden Change"
    elif is_anomaly:
        category = "Other Detected Anomaly"
    else:
        category = "Normal"

    return {
        "timestamp": history_end.isoformat(),
        "demand_kw": round(current, 4),
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(score, 6),
        "absolute_change_kw": round(absolute_change, 4),
        "anomaly_category": category,
        "model": "isolation_forest_anomaly_detector",
    }


def estimate_cost(monthly_energy_kwh):
    energy = float(monthly_energy_kwh)
    if energy < 0:
        raise ValueError("Energy consumption cannot be negative.")

    points = sorted(PUCSL_2026_BILL_REFERENCE.items())

    if energy in PUCSL_2026_BILL_REFERENCE:
        bill = float(PUCSL_2026_BILL_REFERENCE[int(energy)])
    elif energy < points[0][0]:
        (x1, y1), (x2, y2) = points[:2]
        bill = y1 + ((energy - x1) / (x2 - x1)) * (y2 - y1)
    elif energy > points[-1][0]:
        (x1, y1), (x2, y2) = points[-2:]
        bill = y1 + ((energy - x1) / (x2 - x1)) * (y2 - y1)
    else:
        bill = 0.0
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            if x1 <= energy <= x2:
                bill = y1 + ((energy - x1) / (x2 - x1)) * (y2 - y1)
                break

    return {
        "monthly_energy_kwh": round(energy, 2),
        "estimated_bill_lkr": round(float(bill), 2),
        "currency": "LKR",
        "note": "Estimate based on the project's configured reference values; verify against the current official tariff.",
    }
