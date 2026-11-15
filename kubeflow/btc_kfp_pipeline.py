"""Kubeflow pipeline: ingest -> features -> train (compile: python kubeflow/btc_kfp_pipeline.py)."""

from kfp import compiler, dsl

BASE = "python:3.10-slim"
PKGS = ["pandas", "numpy", "scikit-learn", "joblib", "yfinance"]


@dsl.component(base_image=BASE, packages_to_install=PKGS)
def ingest_data(raw_data: dsl.Output[dsl.Dataset], period: str = "5y"):
    import yfinance as yf

    df = yf.download("BTC-USD", period=period, interval="1d", auto_adjust=False, progress=False)
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    df[["Open", "High", "Low", "Close", "Volume"]].dropna().to_csv(raw_data.path)


@dsl.component(base_image=BASE, packages_to_install=PKGS)
def build_features(raw_data: dsl.Input[dsl.Dataset], features: dsl.Output[dsl.Dataset]):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(raw_data.path, index_col=0, parse_dates=True).sort_index()
    df["ret"] = np.log(df["Close"]).diff()
    df["ma7"] = df["Close"].rolling(7).mean()
    df["ma30"] = df["Close"].rolling(30).mean()
    df["vol7"] = df["ret"].rolling(7).std()
    df.dropna().to_csv(features.path)


@dsl.component(base_image=BASE, packages_to_install=PKGS)
def train_model(
    features: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor

    df = pd.read_csv(features.path, index_col=0, parse_dates=True)
    cols = ["ma7", "ma30", "vol7"]
    X = df[cols].shift(1).dropna()
    y = df["ret"].loc[X.index]
    X_tr, X_te, y_tr, y_te = X[:-7], X[-7:], y[:-7], y[-7:]
    m = GradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
    p = m.predict(X_te)
    metrics.log_metric("mae_return", float(np.mean(np.abs(p - y_te))))
    metrics.log_metric("rmse_return", float(np.sqrt(np.mean((p - y_te) ** 2))))
    joblib.dump(m, model.path)


@dsl.pipeline(name="btc-forecast-pipeline", description="BTC-USD ingest, features, training")
def btc_pipeline(period: str = "5y"):
    ingest = ingest_data(period=period)
    feats = build_features(raw_data=ingest.outputs["raw_data"])
    train_model(features=feats.outputs["features"])


if __name__ == "__main__":
    compiler.Compiler().compile(btc_pipeline, "kubeflow/btc_kfp_pipeline.yaml")
    print("compiled -> kubeflow/btc_kfp_pipeline.yaml")
