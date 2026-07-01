"""
Unit tests (UT): test only PricePredictor's pure logic, without the web layer.
"""
import pandas as pd

from app.config import settings
from app.model import FEATURE_NAMES, PricePredictor


def _load_df() -> pd.DataFrame:
    return pd.read_csv(settings.dataset_path)


def test_train_sets_state():
    p = PricePredictor().train(_load_df())
    assert p.is_ready
    assert p.n_training_samples == 50
    # all 7 features should have a coefficient
    assert set(p.coefficients.keys()) == set(FEATURE_NAMES)
    # metric fields present
    assert {"r2", "rmse", "mae"} <= p.metrics.keys()


def test_metrics_reasonable():
    p = PricePredictor().train(_load_df())
    # data is near-linear, so held-out R-squared should be quite high
    assert p.metrics["r2"] > 0.9
    assert p.metrics["rmse"] > 0


def test_predict_single_and_batch():
    p = PricePredictor().train(_load_df())
    one = {
        "square_footage": 1550, "bedrooms": 3, "bathrooms": 2,
        "year_built": 1997, "lot_size": 6800,
        "distance_to_city_center": 4.1, "school_rating": 7.6,
    }
    out = p.predict([one])
    assert len(out) == 1
    assert out[0] > 0  # price is positive

    batch = p.predict([one, one])
    assert len(batch) == 2
    assert batch[0] == batch[1]  # same input -> same output (deterministic)


def test_predict_before_ready_raises():
    p = PricePredictor()  # not trained
    try:
        p.predict([{}])
        assert False, "should have raised"
    except RuntimeError:
        pass


def test_save_and_load_roundtrip(tmp_path):
    # tmp_path is pytest's built-in temporary-directory fixture
    path = str(tmp_path / "model.joblib")
    p1 = PricePredictor().train(_load_df())
    p1.save(path)

    sample = {
        "square_footage": 2000, "bedrooms": 4, "bathrooms": 2.5,
        "year_built": 2005, "lot_size": 9000,
        "distance_to_city_center": 6.0, "school_rating": 8.5,
    }
    p2 = PricePredictor().load(path)
    # a reloaded model should predict identically to the original
    assert p2.predict([sample]) == p1.predict([sample])
    assert p2.coefficients == p1.coefficients
