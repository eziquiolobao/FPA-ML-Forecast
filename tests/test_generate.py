"""The synthetic ERP extract must be deterministic, immutable over time and double-entry sound."""

import hashlib
import inspect
import json

import pandas as pd

import fpa.forecast.backtest
import fpa.forecast.rolling
import fpa.ingest
import fpa.transform
from fpa.generate import generate_raw


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_is_deterministic(tmp_path, settings, raw_dir, committed_as_of):
    again = tmp_path / "again"
    generate_raw(settings, committed_as_of, again)
    for name in ("gl_journal_lines.csv", "budget_fy2025.csv", "_answer_key.json"):
        assert _digest(raw_dir / name) == _digest(again / name), name


def test_history_does_not_change_when_as_of_moves_forward(tmp_path, settings, raw_dir, committed_as_of):
    earlier = committed_as_of - 3
    generate_raw(settings, earlier, tmp_path)
    old = pd.read_csv(tmp_path / "gl_journal_lines.csv", dtype=str)
    new = pd.read_csv(raw_dir / "gl_journal_lines.csv", dtype=str)
    cutoff = str(earlier)
    pd.testing.assert_frame_equal(
        old[old["period"] <= cutoff].reset_index(drop=True), new[new["period"] <= cutoff].reset_index(drop=True)
    )


def test_every_journal_entry_balances_after_extract_dedupe(raw):
    gl = raw.tables["gl"].drop_duplicates(["je_id", "line_no"])
    net = gl.groupby("je_id")[["debit", "credit"]].sum()
    assert ((net["debit"] - net["credit"]).abs() < 0.005).all()


def test_injected_data_quality_issues_are_detected(raw, raw_dir):
    injected = json.loads((raw_dir / "_answer_key.json").read_text())["data_quality_injections"]
    assert raw.dq["extract_duplicate_lines"] == injected["duplicated_extract_lines"]
    assert raw.dq["pnl_lines_blank_cost_center"] == injected["blank_cost_center"]
    assert raw.dq["pnl_lines_invalid_cost_center"] == injected["invalid_cost_center"]
    assert raw.dq["unbalanced_jes_before_dedupe"] > 0
    assert raw.dq["unbalanced_jes_after_dedupe"] == 0


def test_answer_key_is_never_read_by_the_pipeline():
    for module in (fpa.ingest, fpa.transform, fpa.forecast.backtest, fpa.forecast.rolling):
        assert "_answer_key" not in inspect.getsource(module), module.__name__


def test_budget_is_released_in_november(raw_dir, committed_as_of):
    budgets = sorted(int(p.stem[-4:]) for p in raw_dir.glob("budget_fy*.csv"))
    expected_last = committed_as_of.year + (1 if committed_as_of.month >= 11 else 0)
    assert budgets[-1] == expected_last
