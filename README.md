# SmartEnergy AI

## Electricity Demand Forecasting and Anomaly Detection for Smart Homes

SmartEnergy AI is an end-to-end machine learning application designed to analyze household electricity consumption, forecast future energy demand, detect unusual consumption patterns, estimate electricity costs, and provide AI-assisted energy insights.

The project combines **machine learning, time-series feature engineering, anomaly detection, FastAPI, Streamlit, and a voice-enabled AI assistant** into a single smart-energy analytics platform.

---

##  Project Objectives

The main objectives of SmartEnergy AI are to:

* Forecast future household electricity demand.
* Detect abnormal electricity consumption patterns.
* Analyze historical energy consumption.
* Identify high-demand periods.
* Estimate electricity costs.
* Provide interactive energy analytics.
* Compare consumption patterns across homes.
* Support what-if forecasting experiments.
* Provide AI-generated explanations and recommendations.
* Enable voice-based interaction with the energy assistant.

---

## System Architecture

```text
                    ┌─────────────────────────┐
                    │   Household Energy      │
                    │       Dataset           │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Data Cleaning & Quality │
                    │       Processing        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Feature Engineering     │
                    │                         │
                    │ • Calendar Features     │
                    │ • Cyclical Features     │
                    │ • Lag Features          │
                    │ • Rolling Features      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ Demand Forecast  │      │ Anomaly Detection│
          │     XGBoost      │      │ Isolation Forest │
          └────────┬─────────┘      └────────┬─────────┘
                   │                         │
                   └────────────┬────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │      FastAPI Backend    │
                    │                         │
                    │ Forecast / Anomaly /    │
                    │ Cost Estimation / Health│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Streamlit Dashboard   │
                    │                         │
                    │ Analytics & AI Assistant│
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
      ┌──────────────────┐                 ┌──────────────────┐
      │ AI Assistant     │                 │ Voice Assistant  │
      │ Text Interaction │                 │ Speech + TTS     │
      └──────────────────┘                 └──────────────────┘
```

---

##  Dataset

The project uses the **Individual Household Electric Power Consumption** dataset.

The original dataset contains more than **2 million electricity consumption records** collected at household level.

The raw data includes measurements such as:

* Global active power
* Global reactive power
* Voltage
* Global intensity
* Sub-metering measurements

The data is transformed into an hourly dataset before machine-learning modelling.

### Data processing pipeline

```text
Raw Data
   ↓
Missing Value Handling
   ↓
Datetime Construction
   ↓
Resampling
   ↓
Feature Engineering
   ↓
Lag Features
   ↓
Rolling Features
   ↓
Supervised Learning Dataset
   ↓
Train / Validation / Test
```

The raw dataset is intentionally excluded from the GitHub repository because of its large file size.

---

##  Feature Engineering

SmartEnergy AI uses time-series feature engineering to provide the forecasting model with historical and temporal context.

### Calendar features

Examples include:

* Hour
* Day
* Day of week
* Month
* Weekend indicator

### Cyclical features

Time variables are transformed using sine and cosine representations.

This allows the model to understand cyclical relationships such as:

```text
23:00 → 00:00 → 01:00
```

rather than treating them as completely different numerical values.

### Lag features

Historical demand is represented using multiple lag periods:

```text
1 hour
2 hours
3 hours
12 hours
24 hours
48 hours
72 hours
168 hours
```

The 168-hour lag represents the same hour from the previous week.

### Rolling features

Rolling statistics are calculated using historical observations to capture consumption trends and variability.

Rolling windows include:

```text
3 hours
6 hours
12 hours
24 hours
48 hours
168 hours
```

The feature engineering process is designed to avoid using future target information when constructing historical features.

---

#  Electricity Demand Forecasting

The forecasting component uses **XGBoost regression** to predict future household electricity demand.

The model uses historical consumption, temporal information, lag variables, and rolling statistics to generate future demand estimates.

### Forecasting workflow

```text
Historical Consumption
        ↓
Feature Generation
        ↓
XGBoost Forecasting Model
        ↓
Future Demand Prediction
        ↓
Forecast Visualization
```

The application also supports recursive multi-step forecasting using the latest available historical observations.

---

#  Anomaly Detection

SmartEnergy AI uses **Isolation Forest** to identify unusual electricity consumption behaviour.

The anomaly detection component evaluates historical consumption patterns and identifies observations that differ significantly from normal behaviour.

Possible examples include:

* Unexpected consumption spikes
* Unusually low consumption
* Abnormal demand patterns
* Potential appliance-related anomalies

The dashboard presents detected anomalies together with historical consumption context.

---

#  Electricity Cost Estimation

The application includes an electricity cost estimation component.

Users can explore estimated electricity costs based on consumption levels and tariff-related assumptions.

The dashboard provides:

* Estimated consumption
* Estimated monthly energy usage
* Estimated electricity cost
* Consumption-based comparisons

The cost calculation is intended for analytical and demonstration purposes and should not be considered an official electricity bill.

---

#  Interactive Streamlit Dashboard

SmartEnergy AI includes a multi-module Streamlit dashboard.

### 1. Command Center

Provides an overview of the current energy environment including:

* Current demand
* Consumption indicators
* Energy status
* Key insights

### 2. Live Energy

Provides a simulated live energy monitoring experience with:

* Live demand values
* Moving energy chart
* Current consumption status
* Real-time style visualization

### 3. Analytics

Provides historical consumption analysis across different time periods.

Users can explore:

* Hourly consumption
* Daily consumption
* Weekly consumption
* Monthly consumption
* Consumption trends
* Peak demand behaviour

### 4. Forecast

Provides future electricity demand predictions using the trained forecasting model.

### 5. AI Assistant

Users can ask natural-language questions about energy consumption.

Example questions:

```text
```
