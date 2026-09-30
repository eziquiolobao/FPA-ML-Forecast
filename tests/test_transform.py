"""Warehouse build: the P&L must tie to the ledger to the penny."""

import duckdb
import pandas as pd
import pytest

from fpa.transform import build_warehouse


@pytest.fixture(scope="module")
def warehouse(raw, tmp_path_factory, committed_as_of):
    tmp = tmp_path_factory.mktemp("wh")
    result = build_warehouse(raw, tmp / "fpa.duckdb", tmp / "marts", committed_as_of)
    return result, tmp


def test_reconciliation_ties_out(warehouse):
    checks = warehouse[0]["reconciliation"]
    assert checks["gl_to_pnl_mart_diff_usd"] == 0
    assert checks["unbalanced_jes_local"] == 0
    assert checks["fx_translation_rounding_max_usd"] <= 1.0
    assert checks["pnl_lines_without_department"] == 0


def test_missing_cost_centers_are_imputed_not_dropped(warehouse, raw):
    expected = raw.dq["pnl_lines_blank_cost_center"] + raw.dq["pnl_lines_invalid_cost_center"]
    assert warehouse[0]["reconciliation"]["cost_centers_imputed"] == expected


def test_monthly_pnl_grain(warehouse, committed_as_of):
    pnl = pd.read_parquet(warehouse[1] / "marts" / "fct_monthly_pnl.parquet")
    assert pnl["period"].max() == str(committed_as_of)
    assert pnl.groupby(["fs_line", "department"]).ngroups == 27
    assert not pnl.duplicated(["period", "fs_line", "department"]).any()


def test_extract_duplicates_are_removed(warehouse, raw):
    con = duckdb.connect(str(warehouse[1] / "fpa.duckdb"), read_only=True)
    n = con.execute("SELECT COUNT(*) FROM stg_gl_lines").fetchone()[0]
    assert n == raw.dq["rows_after_dedupe"]


def test_unbalanced_entry_blocks_the_close(raw):
    from fpa.ingest import RawData, validate

    broken = {k: v.copy() for k, v in raw.tables.items()}
    gl = broken["gl"]
    idx = gl.index[gl["debit"] > 0][0]
    gl.loc[idx, "debit"] += 100.0
    with pytest.raises(ValueError, match="do not balance"):
        validate(RawData(broken))
