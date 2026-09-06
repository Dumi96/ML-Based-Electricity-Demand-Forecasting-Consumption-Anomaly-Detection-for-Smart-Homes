from sklearn.ensemble import RandomForestRegressor


def create_random_forest_model(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
):
    """
    Create a Random Forest regression model
    for electricity demand forecasting.
    """

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )

    return model