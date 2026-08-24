# SmartEnergy AI — Data Dictionary

## Dataset

Household electricity consumption time-series dataset.

## Columns

| Column                 | Description                                        | Type     |
| ---------------------- | -------------------------------------------------- | -------- |
| Date                   | Measurement date                                   | Date     |
| Time                   | Measurement time                                   | Time     |
| Global_active_power    | Global active power consumption in kW              | Numeric  |
| Global_reactive_power  | Global reactive power                              | Numeric  |
| Voltage                | Average voltage                                    | Numeric  |
| Global_intensity       | Global current intensity                           | Numeric  |
| Sub_metering_1         | Energy sub-metering 1                              | Numeric  |
| Sub_metering_2         | Energy sub-metering 2                              | Numeric  |
| Sub_metering_3         | Energy sub-metering 3                              | Numeric  |
| timestamp              | Combined date and time                             | Datetime |
| is_measurement_missing | Indicates a complete measurement outage            | Boolean  |
| target_available       | Indicates whether active-power target is available | Boolean  |

## Target Variable

The primary forecasting target is:

`Global_active_power`

## Missing Data

The dataset contains 25,979 rows where all electricity measurements are unavailable.

These represent measurement outages rather than isolated missing values.

Therefore, these records are retained in the timeline but are not directly used as supervised target observations during model training.

## Temporal Resolution

The source data has a nominal one-minute sampling frequency.

## Forecasting Considerations

Future feature engineering will derive:

- Lag features
- Rolling statistics
- Calendar features
- Daily patterns
- Weekly patterns
- Seasonal features
- Peak-demand indicators
