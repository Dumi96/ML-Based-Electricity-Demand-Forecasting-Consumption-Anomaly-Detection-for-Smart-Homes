from xgboost import XGBRegressor


def create_tuned_xgboost_model(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
):
    """
    Create an XGBoost regression model with
    configurable hyperparameters.
    """

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=n_jobs
    )

    return model