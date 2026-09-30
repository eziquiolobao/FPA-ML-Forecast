"""The model line-up for the forecasting tournament.

Contenders, simplest first (ties go to the simpler model):
- Naive          : next month = this month
- SeasonalNaive  : next month = same month last year
- ARRRunRate     : (subscription revenue only) the FP&A driver method - revenue = ARR / 12,
                   with ARR rolled forward at its trailing 6-month growth rate
- AutoETS        : exponential smoothing with trend/seasonality chosen automatically
- AutoARIMA      : ARIMA with seasonal terms chosen automatically
- LightGBM       : one gradient-boosted model across all P&L lines, using lags, calendar and
                   the headcount plan as a known future driver
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import Differences, LocalStandardScaler
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, SeasonalNaive

MODEL_ORDER = ["Naive", "SeasonalNaive", "ARRRunRate", "AutoETS", "AutoARIMA", "LightGBM"]
ARR_DRIVEN_SERIES = "Subscription Revenue | Company"


def forecast_stats(train: pd.DataFrame, h: int, n_jobs: int = 1) -> pd.DataFrame:
    sf = StatsForecast(
        models=[Naive(), SeasonalNaive(season_length=12), AutoETS(season_length=12), AutoARIMA(season_length=12)],
        freq="MS",
        n_jobs=n_jobs,
        fallback_model=SeasonalNaive(season_length=12),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = sf.forecast(df=train[["unique_id", "ds", "y"]], h=h)
    return fc.reset_index() if "unique_id" not in fc.columns else fc


def _lgb_model() -> MLForecast:
    reg = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=7, verbose=-1,
    )
    return MLForecast(
        models={"LightGBM": reg},
        freq="MS",
        lags=[1, 2, 3, 12],
        lag_transforms={1: [RollingMean(window_size=3), RollingMean(window_size=6)]},
        date_features=["month"],
        target_transforms=[Differences([12]), LocalStandardScaler()],
    )


def forecast_lgb(train: pd.DataFrame, future_x: pd.DataFrame, h: int) -> tuple[pd.DataFrame, pd.Series]:
    """Fit one global LightGBM model. Returns (forecast, feature importance by gain)."""
    fcst = _lgb_model()
    df = train.merge(future_x, on=["unique_id", "ds"], how="left").fillna({"hc_growth": 0.0})
    fcst.fit(df[["unique_id", "ds", "y", "hc_growth"]], static_features=[])
    last = train["ds"].max()
    x_df = future_x[(future_x["ds"] > last) & (future_x["ds"] <= last + pd.DateOffset(months=h))]
    pred = fcst.predict(h=h, X_df=x_df)
    booster = fcst.models_["LightGBM"]
    imp = pd.Series(booster.booster_.feature_importance("gain"), index=booster.booster_.feature_name())
    return pred, imp / imp.sum()


def forecast_arr_run_rate(arr: pd.DataFrame, origin: pd.Timestamp, h: int) -> pd.DataFrame:
    """Subscription revenue from ARR: month t earns roughly the average ARR during t, divided by 12."""
    hist = arr[arr["ds"] <= origin].set_index("ds")["ending_arr"]
    growth = (hist.iloc[-1] / hist.iloc[-7]) ** (1 / 6) - 1
    steps = np.arange(1, h + 1)
    return pd.DataFrame(
        {
            "unique_id": ARR_DRIVEN_SERIES,
            "ds": pd.date_range(origin + pd.DateOffset(months=1), periods=h, freq="MS"),
            "ARRRunRate": hist.iloc[-1] * (1 + growth) ** (steps - 0.5) / 12,
        }
    )


def forecast_all(
    train: pd.DataFrame, future_x: pd.DataFrame, arr: pd.DataFrame, h: int, n_jobs: int = 1
) -> tuple[pd.DataFrame, pd.Series]:
    """Point forecasts from every contender in long format: unique_id, ds, model, yhat."""
    stats = forecast_stats(train, h, n_jobs)
    lgbm, imp = forecast_lgb(train, future_x, h)
    driver = forecast_arr_run_rate(arr, train["ds"].max(), h)
    wide = stats.merge(lgbm, on=["unique_id", "ds"], how="left").merge(driver, on=["unique_id", "ds"], how="left")
    long = wide.melt(id_vars=["unique_id", "ds"], value_vars=MODEL_ORDER, var_name="model", value_name="yhat")
    return long.dropna(subset=["yhat"]), imp
