from src.ai.recommendations import generate_energy_recommendations


def test_normal_consumption_returns_no_high_consumption():
    recommendations = generate_energy_recommendations(
        monthly_kwh=757.59,
        evening_percentage=20.0,
        night_percentage=10.0,
        anomaly_percentage=2.0,
        high_consumption_threshold=1136.92
    )

    categories = [r["category"] for r in recommendations]

    assert "High Consumption" not in categories


def test_high_consumption_is_detected():
    recommendations = generate_energy_recommendations(
        monthly_kwh=1500.0,
        evening_percentage=20.0,
        night_percentage=10.0,
        anomaly_percentage=2.0,
        high_consumption_threshold=1136.92
    )

    categories = [r["category"] for r in recommendations]

    assert "High Consumption" in categories


def test_evening_consumption_recommendation():
    recommendations = generate_energy_recommendations(
        monthly_kwh=757.59,
        evening_percentage=31.58,
        night_percentage=10.0,
        anomaly_percentage=2.0,
        high_consumption_threshold=1136.92
    )

    categories = [r["category"] for r in recommendations]

    assert "Evening Consumption" in categories


def test_night_consumption_recommendation():
    recommendations = generate_energy_recommendations(
        monthly_kwh=757.59,
        evening_percentage=20.0,
        night_percentage=25.0,
        anomaly_percentage=2.0,
        high_consumption_threshold=1136.92
    )

    categories = [r["category"] for r in recommendations]

    assert "Night Consumption" in categories


def test_anomaly_recommendation():
    recommendations = generate_energy_recommendations(
        monthly_kwh=757.59,
        evening_percentage=20.0,
        night_percentage=10.0,
        anomaly_percentage=5.0,
        high_consumption_threshold=1136.92
    )

    categories = [r["category"] for r in recommendations]

    assert "Potential Anomalies" in categories


def test_actual_project_result_returns_two_recommendations():
    recommendations = generate_energy_recommendations(
        monthly_kwh=757.59,
        evening_percentage=31.58,
        night_percentage=14.98,
        anomaly_percentage=5.00,
        high_consumption_threshold=1136.92
    )

    categories = [r["category"] for r in recommendations]

    assert len(recommendations) == 2
    assert "Evening Consumption" in categories
    assert "Potential Anomalies" in categories
    assert "High Consumption" not in categories
    assert "Night Consumption" not in categories
