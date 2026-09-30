"""Dampen one-off spikes before training ("normalised actuals").

FP&A teams routinely strip one-offs (a legal settlement, a duplicate invoice) out of history
before projecting forward, because a one-off should not repeat in the forecast. I do the same
statistically: a robust STL decomposition splits each series into trend + seasonality + noise,
and any month whose noise is more than `z` robust standard deviations from normal is capped.
Recurring seasonal events (January renewals, the October conference) sit in the seasonal
component, so they are kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

MIN_MONTHS = 36


def clean_series(y: np.ndarray, z: float = 3.5) -> np.ndarray:
    if len(y) < MIN_MONTHS or np.allclose(y, y[0]):
        return y.copy()
    res = STL(y, period=12, seasonal=13, robust=True).fit()
    resid = res.resid
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    if mad == 0:
        return y.copy()
    capped = np.clip(resid, -z * mad, z * mad)
    return res.trend + res.seasonal + capped


def clean_outliers(df: pd.DataFrame, z: float = 3.5) -> pd.DataFrame:
    """df: unique_id, ds, y  ->  same rows plus y_clean."""
    parts = []
    for _, g in df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
        parts.append(g.assign(y_clean=clean_series(g["y"].to_numpy(dtype=float), z)))
    return pd.concat(parts, ignore_index=True)
