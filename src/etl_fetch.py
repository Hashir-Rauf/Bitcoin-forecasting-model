"""Fetch daily BTC-USD OHLCV data and store it as raw CSV."""

import argparse
import os

import pandas as pd

from src.forecast_lib import synthetic_prices

DATA_DIR = os.environ.get("DATA_DIR", "data")
RAW_PATH = os.path.join(DATA_DIR, "btc_raw.csv")


def fetch(period: str = "5y", allow_synthetic: bool = False) -> pd.DataFrame:
    try:
        import yfinance as yf

        df = yf.download("BTC-USD", period=period, interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.empty:
            raise ValueError("empty download")
        return df
    except Exception as exc:  # network/API failure
        if allow_synthetic:
            print(f"[etl_fetch] yfinance failed ({exc}); using synthetic data")
            return synthetic_prices()
        raise


def main(path: str = RAW_PATH, period: str = "5y", allow_synthetic: bool = False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = fetch(period, allow_synthetic)
    df.index.name = "Date"
    df.to_csv(path)
    print(f"[etl_fetch] saved {len(df)} rows to {path} (last {df.index[-1]})")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=RAW_PATH)
    ap.add_argument("--period", default="5y")
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args()
    main(a.path, a.period, a.allow_synthetic)
