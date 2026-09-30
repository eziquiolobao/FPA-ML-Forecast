"""Rolling-origin backtest: replay past month-ends as if each were the latest close.

For every origin we train only on data available at that date, forecast 12 months ahead with
every contender, and later score those forecasts against what actually happened. The same runs
double as the history of forecast *vintages* shown in the app.

Champion selection and prediction intervals at origin `o` use only errors that were already
observable at `o`, so the evaluation stays honestly out-of-sample.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from fpa.forecast.clean import clean_outliers
from fpa.forecast.data import headcount_growth_feature
from fpa.forecast.models import MODEL_ORDER, forecast_all

TIE_TOLERANCE = 0.01  # a simpler model wins if within 1% (relative) of the best WAPE
MIN_INTERVAL_SAMPLES = 8


def wape(y: pd.Series, yhat: pd.Series) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom else np.nan


def bias(y: pd.Series, yhat: pd.Series) -> float:
    denom = np.abs(y).sum()
    return float((yhat - y).sum() / denom) if denom else np.nan


def run_origins(
    y: pd.DataFrame,
    meta: pd.DataFrame,
    marts_dir: Path,
    origins: list[pd.Period],
    horizon: int,
    arr: pd.DataFrame,
    n_jobs: int = 1,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Forecast from every origin with every model.

    Returns (predictions, live LightGBM importance, cleaned history as known at each origin).
    Models train on cleaned history; forecasts are scored against raw actuals.
    """
    out, cleaned, importance = [], [], None
    for o in origins:
        train = clean_outliers(y[y["ds"] <= o.to_timestamp()])
        future_x = headcount_growth_feature(marts_dir, meta, o, horizon)
        fit_df = train[["unique_id", "ds", "y_clean"]].rename(columns={"y_clean": "y"})
        fc, importance = forecast_all(fit_df, future_x, arr, horizon, n_jobs)
        fc["origin"] = o.to_timestamp()
        out.append(fc)
        cleaned.append(train[["unique_id", "ds", "y_clean"]].assign(origin=o.to_timestamp()))
        log(f"  origin {o}: {fc['unique_id'].nunique()} series x {fc['model'].nunique()} models")
    preds = pd.concat(out, ignore_index=True)
    preds["horizon"] = (
        (preds["ds"].dt.year - preds["origin"].dt.year) * 12 + (preds["ds"].dt.month - preds["origin"].dt.month)
    )
    preds = preds.merge(y, on=["unique_id", "ds"], how="left")  # y is NaN for months not yet closed
    return preds, importance, pd.concat(cleaned, ignore_index=True)


def select_champions(scored: pd.DataFrame) -> pd.DataFrame:
    """Lowest WAPE per series; near-ties go to the simpler model."""
    w = scored.groupby(["unique_id", "model"]).apply(lambda g: wape(g["y"], g["yhat"]), include_groups=False)
    w = w.rename("wape").reset_index()
    w["rank"] = w["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    rows = []
    for uid, g in w.groupby("unique_id"):
        best = g["wape"].min()
        pick = g[g["wape"] <= best * (1 + TIE_TOLERANCE)].sort_values("rank").iloc[0]
        rows.append((uid, pick["model"], pick["wape"]))
    return pd.DataFrame(rows, columns=["unique_id", "champion", "champion_wape"])


def _interval_quantiles(prior: pd.DataFrame, levels: list[int], horizon: int) -> pd.DataFrame:
    """Relative absolute error quantiles per (series, model, horizon), pooling adjacent horizons.

    Errors are measured against *cleaned* actuals, so the expected range describes normal
    business variation rather than being inflated by past one-offs.
    """
    prior = prior.assign(rel_err=(prior["y_clean"] - prior["yhat"]).abs() / prior["yhat"].abs().clip(lower=1.0))
    rows = []
    for (uid, model), g in prior.groupby(["unique_id", "model"]):
        for h in range(1, horizon + 1):
            sample = g.loc[g["horizon"].between(h - 1, h + 1), "rel_err"].to_numpy()
            if len(sample) < MIN_INTERVAL_SAMPLES:
                continue
            rows.append((uid, model, h, *[float(np.quantile(sample, lv / 100)) for lv in levels]))
    return pd.DataFrame(rows, columns=["unique_id", "model", "horizon", *[f"q{lv}" for lv in levels]])


def assemble_vintages(
    preds: pd.DataFrame, cleaned: pd.DataFrame, levels: list[int], horizon: int, min_prior_origins: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tag each origin's champion (chosen with information available then) and add intervals."""
    origins = sorted(preds["origin"].unique())
    tagged, history = [], []
    for o in origins:
        cur = preds[preds["origin"] == o].copy()
        prior = preds[(preds["origin"] < o) & (preds["ds"] <= o) & preds["y"].notna()]
        n_prior = prior["origin"].nunique()
        cur["is_champion"] = False
        for lv in levels:
            cur[f"lo{lv}"] = np.nan
            cur[f"hi{lv}"] = np.nan
        if n_prior >= min_prior_origins:
            champs = select_champions(prior)
            champs["origin"] = o
            history.append(champs)
            cur = cur.merge(champs[["unique_id", "champion"]], on="unique_id", how="left")
            cur["is_champion"] = cur["model"] == cur["champion"]
            cur = cur.drop(columns="champion")
            known = cleaned.loc[cleaned["origin"] == o, ["unique_id", "ds", "y_clean"]]
            q = _interval_quantiles(prior.merge(known, on=["unique_id", "ds"]), levels, horizon)
            cur = cur.merge(q, on=["unique_id", "model", "horizon"], how="left")
            for lv in levels:
                width = cur[f"q{lv}"] * cur["yhat"].abs()
                # P&L lines here don't go negative, so the range stops at zero
                cur[f"lo{lv}"] = (cur["yhat"] - width).where(cur["yhat"] < 0, (cur["yhat"] - width).clip(lower=0))
                cur[f"hi{lv}"] = cur["yhat"] + width
            cur = cur.drop(columns=[f"q{lv}" for lv in levels])
        tagged.append(cur)
    vintages = pd.concat(tagged, ignore_index=True)
    champion_history = pd.concat(history, ignore_index=True) if history else pd.DataFrame()
    return vintages, champion_history


def horizon_bucket(h: pd.Series) -> pd.Series:
    return pd.cut(h, bins=[0, 3, 6, 12], labels=["1-3 months", "4-6 months", "7-12 months"]).astype(str)


def leaderboard(preds: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """WAPE and bias for every series x model x horizon bucket across all scored origins."""
    s = preds[(preds["origin"] < as_of) & preds["y"].notna()].copy()
    s["bucket"] = horizon_bucket(s["horizon"])
    frames = [s, s.assign(bucket="All horizons")]
    s = pd.concat(frames, ignore_index=True)
    g = s.groupby(["unique_id", "model", "bucket"])
    out = g.apply(
        lambda x: pd.Series({"wape": wape(x["y"], x["yhat"]), "bias": bias(x["y"], x["yhat"]), "n": len(x)}),
        include_groups=False,
    )
    return out.reset_index()
