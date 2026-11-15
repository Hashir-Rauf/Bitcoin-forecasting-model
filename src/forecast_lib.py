"""Shared feature engineering + recursive 7-day forecasting helpers."""

import numpy as np
import pandas as pd

LAGS = 10


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to a frame with a 'Close' column."""
    out = df.copy()
    out["ret"] = np.log(out["Close"]).diff()
    out["ma7"] = out["Close"].rolling(7).mean()
    out["ma30"] = out["Close"].rolling(30).mean()
    out["vol7"] = out["ret"].rolling(7).std()
    out["vol30"] = out["ret"].rolling(30).std()
    for i in range(1, LAGS + 1):
        out[f"lag{i}"] = out["ret"].shift(i - 1)
    out["ma7_ratio"] = out["ma7"] / out["Close"] - 1
    out["ma30_ratio"] = out["ma30"] / out["Close"] - 1
    return out


FEATURES = [f"lag{i}" for i in range(1, LAGS + 1)] + [
    "ma7_ratio",
    "ma30_ratio",
    "vol7",
    "vol30",
]


def make_xy(df: pd.DataFrame):
    f = add_features(df)
    f["target"] = f["ret"].shift(-1)
    f = f.dropna(subset=FEATURES + ["target"])
    return f[FEATURES], f["target"]


def recursive_forecast(model, close: pd.Series, days: int = 7) -> pd.DataFrame:
    """Predict next `days` closes by iterating one-step log-return predictions."""
    hist = pd.DataFrame({"Close": close.astype(float).values}, index=close.index)
    hist = hist.tail(120).copy()
    last_date = pd.Timestamp(close.index[-1])
    rows = []
    for k in range(1, days + 1):
        feats = add_features(hist).iloc[[-1]][FEATURES]
        r = float(model.predict(feats.values)[0])
        nxt = float(hist["Close"].iloc[-1] * np.exp(r))
        d = last_date + pd.Timedelta(days=k)
        hist.loc[d] = nxt
        rows.append({"date": d.strftime("%Y-%m-%d"), "predicted_close": round(nxt, 2)})
    return pd.DataFrame(rows)


def synthetic_prices(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Offline fallback so tests/CI never need the network."""
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0.0005, 0.025, n)))
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(10**9, 5 * 10**9, n).astype(float),
        },
        index=idx,
    )
