from src.models.linear_regression import (
    create_linear_regression_model
)

from src.models.random_forest import (
    create_random_forest_model
)


def test_create_linear_regression_model():

    model = create_linear_regression_model()

    assert model is not None
    assert model.__class__.__name__ == "LinearRegression"


def test_create_random_forest_model():

    model = create_random_forest_model()

    assert model is not None
    assert model.__class__.__name__ == "RandomForestRegressor"