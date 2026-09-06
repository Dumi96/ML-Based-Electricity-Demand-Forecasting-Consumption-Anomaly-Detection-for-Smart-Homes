from src.models.linear_regression import (
    create_linear_regression_model
)


def test_create_linear_regression_model():

    model = create_linear_regression_model()

    assert model is not None
    assert model.__class__.__name__ == "LinearRegression"