from xgboost import XGBRegressor


def create_xgboost_model(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
):
    """
    Create an XGBoost regression model
    for electricity demand forecasting.
    """

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=n_jobs
    )

    return model