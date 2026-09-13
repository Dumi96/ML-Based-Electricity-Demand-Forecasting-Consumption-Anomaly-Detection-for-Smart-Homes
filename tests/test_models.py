from src.models.lightgbm_tuning import create_tuned_lightgbm_model
from src.models.lightgbm_model import create_lightgbm_model
from src.models.xgboost_tuning import create_tuned_xgboost_model
from src.models.xgboost_model import create_xgboost_model
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

def test_create_xgboost_model():

    model = create_xgboost_model()

    assert model is not None
    assert model.__class__.__name__ == "XGBRegressor"

def test_create_tuned_xgboost_model():

    model = create_tuned_xgboost_model()

    assert model is not None
    assert model.__class__.__name__ == "XGBRegressor"

def test_create_lightgbm_model():
    model = create_lightgbm_model()

    assert model is not None
    assert model.__class__.__name__ == "LGBMRegressor"

def test_create_tuned_lightgbm_model():
    model = create_tuned_lightgbm_model()

    assert model is not None
    assert model.__class__.__name__ == "LGBMRegressor"

from src.models.peak_classifier import create_peak_classifier


def test_create_peak_classifier():
    model = create_peak_classifier()

    assert model is not None
    assert model.__class__.__name__ == "XGBClassifier"

from src.models.peak_regressor import create_peak_regressor


def test_create_peak_regressor():
    model = create_peak_regressor()

    assert model is not None
    assert model.__class__.__name__ == "XGBRegressor"

from src.models.peak_classifier_weighted import (
    create_weighted_peak_classifier
)


def test_create_weighted_peak_classifier():
    model = create_weighted_peak_classifier()

    assert model is not None
    assert model.__class__.__name__ == "XGBClassifier"