def generate_energy_recommendations(
    monthly_kwh,
    evening_percentage,
    night_percentage,
    anomaly_percentage,
    high_consumption_threshold
):
    """Generate explainable electricity-consumption recommendations."""

    recommendations = []

    if monthly_kwh > high_consumption_threshold:
        recommendations.append({
            "category": "High Consumption",
            "severity": "High",
            "finding": (
                f"Monthly consumption of {monthly_kwh:.2f} kWh is above "
                f"the project reference threshold of "
                f"{high_consumption_threshold:.2f} kWh."
            ),
            "recommendation": (
                "Review major electricity-consuming appliances and "
                "investigate opportunities to reduce unnecessary usage."
            )
        })

    if evening_percentage >= 30:
        recommendations.append({
            "category": "Evening Consumption",
            "severity": "Medium",
            "finding": (
                f"{evening_percentage:.2f}% of consumption occurs "
                "during the evening period."
            ),
            "recommendation": (
                "Where practical, shift flexible loads such as washing, "
                "water heating, or other non-essential usage outside "
                "the evening period."
            )
        })

    if night_percentage >= 20:
        recommendations.append({
            "category": "Night Consumption",
            "severity": "Low",
            "finding": (
                f"{night_percentage:.2f}% of consumption occurs "
                "during the night/other period."
            ),
            "recommendation": (
                "Check appliances operating overnight and switch off "
                "unnecessary loads where practical."
            )
        })

    if anomaly_percentage >= 5:
        recommendations.append({
            "category": "Potential Anomalies",
            "severity": "Medium",
            "finding": (
                f"{anomaly_percentage:.2f}% of observations were detected "
                "as potential consumption anomalies."
            ),
            "recommendation": (
                "Review detected anomaly timestamps to investigate "
                "unusual appliance usage, sudden demand changes, "
                "or other abnormal consumption patterns."
            )
        })

    return recommendations
