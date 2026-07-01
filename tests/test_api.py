"""
Integration tests (IT): test the three endpoints over HTTP (status codes + response shape),
including input-validation failure cases (422).

TestClient runs the whole FastAPI app in-memory, no real server needed, so it is fast.
"""

VALID = {
    "square_footage": 1550, "bedrooms": 3, "bathrooms": 2,
    "year_built": 1997, "lot_size": 6800,
    "distance_to_city_center": 4.1, "school_rating": 7.6,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_single(client):
    r = client.post("/predict", json={"instances": [VALID]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["predictions"][0]["predicted_price"] > 0


def test_predict_batch(client):
    r = client.post("/predict", json={"instances": [VALID, VALID]})
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_predict_empty_instances_rejected(client):
    r = client.post("/predict", json={"instances": []})
    assert r.status_code == 422  # min_length=1 validation fails


def test_predict_missing_field_rejected(client):
    bad = {k: v for k, v in VALID.items() if k != "bedrooms"}
    r = client.post("/predict", json={"instances": [bad]})
    assert r.status_code == 422


def test_predict_negative_value_rejected(client):
    bad = {**VALID, "square_footage": -100}
    r = client.post("/predict", json={"instances": [bad]})
    assert r.status_code == 422


def test_predict_unknown_field_rejected(client):
    bad = {**VALID, "swimming_pool": True}  # extra="forbid"
    r = client.post("/predict", json={"instances": [bad]})
    assert r.status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"].startswith("Ridge")
    assert len(body["features"]) == 7
    assert len(body["coefficients"]) == 7
    assert {"r2", "rmse", "mae"} <= body["metrics"].keys()
