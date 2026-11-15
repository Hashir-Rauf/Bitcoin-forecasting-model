"""Train the 7-day BTC close forecaster, evaluate on the latest 7-day window, save artifacts."""

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.forecast_lib import make_xy, recursive_forecast

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
HORIZON = 7


def load_raw(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mape": float(np.mean(np.abs((y - p) / y)) * 100),
    }


def fit(df):
    X, y = make_xy(df)
    m = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    m.fit(X.values, y.values)
    return m


def train(raw_path, model_dir=MODEL_DIR, mape_threshold=25.0):
    os.makedirs(model_dir, exist_ok=True)
    df = load_raw(raw_path)
    # Validation: hold out the latest 7 days, forecast them recursively.
    train_df, test_df = df.iloc[:-HORIZON], df.iloc[-HORIZON:]
    val_model = fit(train_df)
    pred = recursive_forecast(val_model, train_df["Close"], HORIZON)
    m = metrics(test_df["Close"].values, pred["predicted_close"].values)
    m["passed"] = bool(m["mape"] <= mape_threshold)
    final = fit(df)
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%d%H%M%S")
    m.update(
        version=version,
        trained_at=now.isoformat(),
        n_rows=int(len(df)),
        last_data_date=str(df.index[-1].date()),
    )
    if m["passed"]:  # "register" artifacts only if threshold met
        joblib.dump({"model": final, "version": version}, os.path.join(model_dir, "model.joblib"))
        df[["Close"]].tail(150).to_csv(os.path.join(model_dir, "history.csv"))
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        print(f"[train] registered model {version}: {m}")
    else:
        print(f"[train] threshold FAILED, model not registered: {m}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(DATA_DIR, "btc_raw.csv"))
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--mape-threshold", type=float, default=25.0)
    a = ap.parse_args()
    train(a.raw, a.model_dir, a.mape_threshold)
