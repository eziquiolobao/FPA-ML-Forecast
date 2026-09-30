"""Guard the headline claims made in the README against the committed marts."""

import pandas as pd
import pytest

from fpa.variance.flagging import evaluate_against_answer_key
from tests.conftest import MARTS

pytestmark = pytest.mark.claims


def test_rolling_forecast_beats_budget():
    s = pd.read_parquet(MARTS / "accuracy_summary.parquet").iloc[0]
    assert s["rolling_forecast_wape"] < s["budget_wape"]


def test_flags_catch_most_injected_anomalies(raw_dir):
    flags = pd.read_parquet(MARTS / "variance_monthly.parquet")
    _, summary = evaluate_against_answer_key(flags, raw_dir / "_answer_key.json")
    assert summary["recall_red"] >= 0.8
    assert summary["precision_red"] >= 0.6


def test_variance_table_covers_every_line_for_twelve_months():
    flags = pd.read_parquet(MARTS / "variance_monthly.parquet")
    assert flags["period"].nunique() == 12
    assert flags.groupby("period")["unique_id"].nunique().eq(27).all()
    red_amber = flags[flags["severity"] != "Green"]
    assert red_amber["commentary"].str.len().gt(40).all()
