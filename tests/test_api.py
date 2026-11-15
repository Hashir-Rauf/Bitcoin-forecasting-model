import json

import pytest

from src.forecast_lib import synthetic_prices
from src.train_forecast import train


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    raw = d / "raw.csv"
    synthetic_prices(300).to_csv(raw, index_label="Date")
    m = train(str(raw), str(d), mape_threshold=1000)
    assert m["passed"]
    import app.app as a

    a.MODEL_DIR = str(d)
    assert a.load_model()
    a.app.testing = True
    return a.app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "ok" and j["model_loaded"] and "uptime_seconds" in j


def test_predict_default_7_days(client):
    r = client.post("/predict")
    assert r.status_code == 200
    p = r.get_json()["predictions"]
    assert len(p) == 7 and all(x["predicted_close"] > 0 for x in p)


def test_predict_custom_days(client):
    r = client.post("/predict", json={"forecast_days": 3})
    assert len(r.get_json()["predictions"]) == 3


def test_predict_custom_history(client):
    r = client.post("/predict", json={"history": [100 + i for i in range(60)]})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"forecast_days": 0},
        {"forecast_days": "x"},
        {"forecast_days": True},
        {"history": [1, 2]},
        {"history": ["a"] * 50},
        [1, 2],
    ],
)
def test_predict_bad_input(client, body):
    assert client.post("/predict", json=body).status_code == 400


def test_predict_invalid_json(client):
    r = client.post("/predict", data="{bad", content_type="application/json")
    assert r.status_code == 400


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert {"mae", "rmse", "mape"} <= set(json.loads(r.data))


def test_reload(client):
    assert client.post("/reload").status_code == 200
