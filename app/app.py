"""Flask REST API serving the 7-day BTC-USD close forecast."""

import json
import logging
import os
import time

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.forecast_lib import recursive_forecast

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "production")
API_KEY = os.environ.get("API_KEY", "")  # from K8s Secret; guards /reload when set

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("btc-api")

app = Flask(__name__)
START = time.time()
STATE = {"model": None, "version": None, "history": None, "metrics": {}}


def load_model():
    """(Re)load model, history and metrics from MODEL_DIR."""
    path = os.path.join(MODEL_DIR, "model.joblib")
    if not os.path.exists(path):
        log.warning("no model at %s", path)
        return False
    art = joblib.load(path)
    hist = pd.read_csv(os.path.join(MODEL_DIR, "history.csv"), index_col=0, parse_dates=True)
    mpath = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            STATE["metrics"] = json.load(f)
    STATE.update(model=art["model"], version=art["version"], history=hist["Close"])
    log.info("loaded model version %s", art["version"])
    return True


load_model()


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        uptime_seconds=round(time.time() - START, 1),
        model_version=STATE["version"],
        model_loaded=STATE["model"] is not None,
        stage=MODEL_STAGE,
    )


@app.post("/predict")
def predict():
    try:
        body = request.get_json(silent=True)
        if request.data and body is None:
            return jsonify(error="Body must be valid JSON"), 400
        body = body or {}
        if not isinstance(body, dict):
            return jsonify(error="Body must be a JSON object"), 400
        days = body.get("forecast_days", 7)
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 30:
            return jsonify(error="forecast_days must be an integer in [1, 30]"), 400
        if STATE["model"] is None:
            return jsonify(error="Model not loaded"), 503
        close = STATE["history"]
        if "history" in body:
            h = body["history"]
            if (
                not isinstance(h, list)
                or len(h) < 40
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0 for x in h)
            ):
                return jsonify(error="history must be a list of >=40 positive numbers"), 400
            end = pd.Timestamp.today().normalize()
            close = pd.Series(h, index=pd.date_range(end=end, periods=len(h), freq="D"))
        fc = recursive_forecast(STATE["model"], close, days)
        return jsonify(
            model_version=STATE["version"],
            forecast_days=days,
            predictions=fc.to_dict("records"),
        )
    except Exception:  # noqa: BLE001
        log.exception("prediction failed")
        return jsonify(error="Internal server error"), 500


@app.get("/metrics")
def metrics():
    if not STATE["metrics"]:
        return jsonify(error="No metrics available"), 404
    return jsonify(STATE["metrics"])


@app.post("/reload")
def reload_model():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="Unauthorized"), 401
    try:
        ok = load_model()
        return jsonify(reloaded=ok, model_version=STATE["version"]), (200 if ok else 404)
    except Exception:  # noqa: BLE001
        log.exception("reload failed")
        return jsonify(error="Internal server error"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
