from lightgbm import LGBMRegressor


def create_tuned_lightgbm_model(
    n_estimators=500,
    max_depth=-1,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
):
    """
    Create a configurable LightGBM regression model
    for electricity demand forecasting.
    """

    model = LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="regression",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1
    )

    return model