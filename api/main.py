from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_features import build_features


app = FastAPI(
    title="SmartEnergy AI API",
    description="Electricity demand forecasting API",
    version="1.1.0",
)


BASE_DIR = Path(__file__).resolve().parent.parent
FORECAST_MODEL_PATH = BASE_DIR / "models" / "tuned_xgboost_forecaster.joblib"
ANOMALY_MODEL_PATH = BASE_DIR / "models" / "isolation_forest_anomaly_detector.joblib"

model = joblib.load(FORECAST_MODEL_PATH)
anomaly_model = joblib.load(ANOMALY_MODEL_PATH)


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
ANOMALY_FEATURE_NAMES = [
    "demand_kw",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "is_evening_peak",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_mean_24h",
    "rolling_std_24h",
]
 
PUCSL_2026_BILL_REFERENCE = {
    30: 230,
    60: 630,
    90: 1840,
    120: 3280,
    150: 5100,
    180: 6420,
    210: 9570,
    240: 12120,
    270: 14670,
    300: 17220,
}

class ForecastRequest(BaseModel):
    features: dict[str, float]


class ForecastRequest(BaseModel):
    features: dict[str, float]


class HistoryForecastRequest(BaseModel):
    history_end_timestamp: str = Field(
        description="Timestamp of the most recent hourly consumption value"
    )
    hourly_consumption_kw: list[float] = Field(
        description="Hourly consumption values ordered from oldest to newest"
    )


class CostRequest(BaseModel):
    monthly_energy_kwh: float = Field(
        gt=0,
        description="Estimated monthly electricity consumption in kWh"
    )


@app.get("/")
def root():
    return {
        "service": "SmartEnergy AI API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SmartEnergy AI API",
        "model": "tuned_xgboost_forecaster",
        "features_required": len(FEATURE_NAMES),
    }


@app.post("/forecast")
def forecast(request: ForecastRequest):

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in request.features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required features",
                "missing_features": missing_features,
            },
        )

    input_data = pd.DataFrame(
        [[request.features[feature] for feature in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_demand_kw": round(float(prediction), 4),
        "unit": "kW",
    }


@app.post("/forecast-from-history")
def forecast_from_history(request: HistoryForecastRequest):

    # We need at least 169 historical hourly values
    # because the model uses a 168-hour lag and rolling window.
    if len(request.hourly_consumption_kw) < 169:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Insufficient historical data",
                "minimum_required_hours": 169,
                "received_hours": len(request.hourly_consumption_kw),
            },
        )

    try:
        history_end = pd.Timestamp(
            request.history_end_timestamp
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid timestamp",
                "message": str(exc),
            },
        )

    # Create timestamps for the historical observations.
    history_index = pd.date_range(
        end=history_end,
        periods=len(request.hourly_consumption_kw),
        freq="1h",
    )

    history_df = pd.DataFrame(
        {
            "demand_kw": request.hourly_consumption_kw
        },
        index=history_index,
    )

    # The model predicts the NEXT hour.
    next_timestamp = history_end + pd.Timedelta(hours=1)

    next_row = pd.DataFrame(
        {"demand_kw": [float("nan")]},
        index=[next_timestamp],
    )

    combined_df = pd.concat(
        [history_df, next_row]
    )

    # Use exactly the same feature-engineering pipeline
    # used during model training.
    feature_df = build_features(
        combined_df,
        target_column="demand_kw",
    )

    prediction_row = feature_df.loc[
        [next_timestamp],
        FEATURE_NAMES,
    ]

    # Check that all required features were generated.
    if prediction_row.isna().any().any():
        missing_features = prediction_row.columns[
            prediction_row.isna().any()
        ].tolist()

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unable to generate complete features",
                "missing_features": missing_features,
            },
        )

    prediction = model.predict(prediction_row)[0]

    return {
        "forecast_timestamp": next_timestamp.isoformat(),
        "predicted_demand_kw": round(float(prediction), 4),
        "unit": "kW",
        "history_hours_used": len(request.hourly_consumption_kw),
        "model": "tuned_xgboost_forecaster",
    }
@app.post("/anomaly")
def detect_anomaly(request: HistoryForecastRequest):
    if len(request.hourly_consumption_kw) < 169:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Insufficient historical data",
                "minimum_required_hours": 169,
                "received_hours": len(request.hourly_consumption_kw),
            },
        )

    try:
        history_end = pd.Timestamp(request.history_end_timestamp)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid timestamp",
                "message": str(exc),
            },
        )

    history_index = pd.date_range(
        end=history_end,
        periods=len(request.hourly_consumption_kw),
        freq="1h",
    )

    history_df = pd.DataFrame(
        {"demand_kw": request.hourly_consumption_kw},
        index=history_index,
    )

    feature_df = build_features(
        history_df,
        target_column="demand_kw",
    )

    current_row = feature_df.loc[
        [history_end],
        ANOMALY_FEATURE_NAMES,
    ]

    if current_row.isna().any().any():
        missing_features = current_row.columns[
            current_row.isna().any()
        ].tolist()

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unable to generate complete anomaly features",
                "missing_features": missing_features,
            },
        )

    prediction = anomaly_model.predict(current_row)[0]
    anomaly_score = anomaly_model.decision_function(current_row)[0]

    is_anomaly = prediction == -1

    absolute_change = abs(
        float(request.hourly_consumption_kw[-1])
        - float(request.hourly_consumption_kw[-2])
    )

    high_consumption = (
        is_anomaly
        and float(request.hourly_consumption_kw[-1]) >= 2.855
    )

    sudden_change = (
        is_anomaly
        and absolute_change >= 1.493
    )

    if high_consumption and sudden_change:
        anomaly_category = "High Consumption + Sudden Change"
    elif high_consumption:
        anomaly_category = "High Consumption"
    elif sudden_change:
        anomaly_category = "Sudden Change"
    elif is_anomaly:
        anomaly_category = "Other Detected Anomaly"
    else:
        anomaly_category = "Normal"

    return {
        "timestamp": history_end.isoformat(),
        "demand_kw": round(
            float(request.hourly_consumption_kw[-1]),
            4,
        ),
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(
            float(anomaly_score),
            6,
        ),
        "absolute_change_kw": round(
            absolute_change,
            4,
        ),
        "anomaly_category": anomaly_category,
        "model": "isolation_forest_anomaly_detector",
    }
@app.post("/cost-estimate")
def estimate_cost(request: CostRequest):
    energy_kwh = request.monthly_energy_kwh

    if energy_kwh < 0:
        raise HTTPException(
            status_code=400,
            detail="Energy consumption cannot be negative.",
        )

    points = sorted(PUCSL_2026_BILL_REFERENCE.items())

    # Exact official reference value
    if energy_kwh in PUCSL_2026_BILL_REFERENCE:
        estimated_bill = float(
            PUCSL_2026_BILL_REFERENCE[int(energy_kwh)]
        )

    # Below first reference point
    elif energy_kwh < points[0][0]:
        lower_kwh, lower_bill = points[0]
        upper_kwh, upper_bill = points[1]

        ratio = (
            (energy_kwh - lower_kwh)
            / (upper_kwh - lower_kwh)
        )

        estimated_bill = (
            lower_bill
            + ratio * (upper_bill - lower_bill)
        )

    # Above last reference point
    elif energy_kwh > points[-1][0]:
        lower_kwh, lower_bill = points[-2]
        upper_kwh, upper_bill = points[-1]

        ratio = (
            (energy_kwh - lower_kwh)
            / (upper_kwh - lower_kwh)
        )

        estimated_bill = (
            lower_bill
            + ratio * (upper_bill - lower_bill)
        )

    # Between reference points
    else:
        estimated_bill = None

        for i in range(len(points) - 1):
            lower_kwh, lower_bill = points[i]
            upper_kwh, upper_bill = points[i + 1]

            if lower_kwh <= energy_kwh <= upper_kwh:
                ratio = (
                    (energy_kwh - lower_kwh)
                    / (upper_kwh - lower_kwh)
                )

                estimated_bill = (
                    lower_bill
                    + ratio * (upper_bill - lower_bill)
                )
                break

    return {
        "monthly_energy_kwh": round(energy_kwh, 2),
        "estimated_bill_lkr": round(float(estimated_bill), 2),
        "currency": "LKR",
        "tariff_reference_year": 2026,
        "calculation_method": (
            "PUCSL published bill reference points "
            "with linear interpolation/extrapolation"
        ),
        "important_note": (
            "This is an estimated cost using the 2026 "
            "PUCSL reference tariff. It is not a "
            "historical electricity bill."
        ),
    }