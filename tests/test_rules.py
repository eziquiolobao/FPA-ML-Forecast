"""Unit tests for the variance rules and helpers."""

import numpy as np
import pandas as pd

from fpa.forecast.backtest import bias, select_champions, wape
from fpa.forecast.clean import clean_series
from fpa.variance import rules as R
from fpa.variance.commentary import money
from fpa.variance.drilldown import headcount_rate_split


def test_materiality_needs_both_dollar_and_percent():
    var = np.array([60_000, 60_000, 10_000, np.nan])
    base = np.array([1_000_000, 300_000, 20_000, 100.0])
    got = R.is_material(var, base, np.full(4, 50_000.0), np.full(4, 0.10))
    assert got.tolist() == [False, True, False, False]


def test_favourability_flips_for_costs():
    assert R.favourable_sign(pd.Series(["Revenue", "Operating Expense"])).tolist() == [1.0, -1.0]


def test_severity_matrix():
    material = np.array([True, True, False, False, False])
    surprise = np.array([True, False, True, False, False])
    minor = np.array([False, False, True, False, False])
    drift = np.array([False, False, False, True, False])
    assert R.severity(material, surprise, minor, drift).tolist() == ["Red", "Amber", "Amber", "Amber", "Green"]


def test_drift_counts_same_direction_runs():
    s = pd.Series([1, 1, 1, -1, 0, -1, -1])
    assert R.drift_run_length(s).tolist() == [1, 2, 3, 1, 0, 1, 2]


def test_headcount_rate_split_explains_the_whole_variance():
    split = headcount_rate_split(actual=950_000, budget=1_000_000, hc_actual=95, hc_plan=100)
    assert round(split["volume_usd"] + split["rate_usd"], 2) == -50_000


def test_wape_and_bias():
    y, yhat = pd.Series([100.0, 200.0]), pd.Series([110.0, 180.0])
    assert wape(y, yhat) == 30 / 300
    assert bias(y, yhat) == -10 / 300


def test_champion_tie_goes_to_simpler_model():
    rows = []
    for model, err in (("Naive", 10.0), ("AutoARIMA", 9.95), ("SeasonalNaive", 30.0)):
        rows += [{"unique_id": "s", "model": model, "y": 100.0, "yhat": 100.0 + err}]
    assert select_champions(pd.DataFrame(rows)).loc[0, "champion"] == "Naive"


def test_cleaning_caps_one_off_spike_but_keeps_seasonality():
    t = np.arange(60)
    y = 100 + t + 30 * (t % 12 == 9)  # October conference every year
    y = y + np.random.default_rng(0).normal(0, 1, 60)
    y_spiked = y.copy()
    y_spiked[40] += 150  # one-off
    clean = clean_series(y_spiked)
    assert abs(clean[40] - y[40]) < 20
    assert clean[45] - y[45] < 5  # an October peak survives


def test_money_format():
    assert money(1_234_567) == "$1.23M"
    assert money(-45_300) == "-$45k"
