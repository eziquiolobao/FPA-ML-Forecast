"""Load validated raw data into DuckDB, run the SQL models and export marts to parquet."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from fpa.ingest import RawData

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
MART_TABLES = ["fct_monthly_pnl", "fct_budget_monthly", "fct_headcount", "fct_arr"]
RAW_TABLE_NAMES = {
    "gl": "raw_gl", "chart_of_accounts": "raw_chart_of_accounts", "cost_centers": "raw_cost_centers",
    "vendors": "raw_vendors", "entities": "raw_entities", "fx": "raw_fx", "budget": "raw_budget",
    "hris": "raw_hris", "hc_plan": "raw_hc_plan", "arr": "raw_arr",
}


def run_sql_models(con: duckdb.DuckDBPyConnection) -> list[str]:
    ran = []
    for layer in ("staging", "marts"):
        for f in sorted((SQL_DIR / layer).glob("*.sql")):
            con.execute(f.read_text())
            ran.append(f"{layer}/{f.name}")
    return ran


def reconcile(con: duckdb.DuckDBPyConnection) -> dict:
    """Accounting checks that must hold before anything is forecast or reported."""
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    checks = {
        # every P&L line in the ledger lands in the monthly P&L mart, to the penny
        "gl_to_pnl_mart_diff_usd": q(
            """SELECT ROUND(ABS((SELECT SUM(amount_usd) FROM fct_gl_pnl_lines)
                               - (SELECT SUM(actual_usd) FROM fct_monthly_pnl)), 2)"""
        ),
        # double entry: all lines net to zero in local currency for every journal entry
        "unbalanced_jes_local": q(
            "SELECT COUNT(*) FROM (SELECT je_id FROM stg_gl_lines GROUP BY je_id HAVING ABS(SUM(amount_local)) > 0.005)"
        ),
        # after translating each line to USD cents, only immaterial rounding may remain
        "fx_translation_rounding_max_usd": q(
            "SELECT ROUND(MAX(ABS(s)), 2) FROM (SELECT SUM(amount_usd) s FROM stg_gl_lines GROUP BY period)"
        ),
        "pnl_lines_without_department": q("SELECT COUNT(*) FROM fct_gl_pnl_lines WHERE department IS NULL"),
        "pnl_lines_missing_fx": q("SELECT COUNT(*) FROM stg_gl_lines WHERE fx_rate IS NULL"),
        "cost_centers_imputed": q("SELECT COUNT(*) FROM fct_gl_pnl_lines WHERE cost_center_imputed"),
    }
    tolerance = {"gl_to_pnl_mart_diff_usd": 0.0, "fx_translation_rounding_max_usd": 1.0}
    failures = [k for k, v in checks.items() if k != "cost_centers_imputed" and (v or 0) > tolerance.get(k, 0)]
    if failures:
        raise ValueError(f"Reconciliation failed: { {k: checks[k] for k in failures} }")
    return checks


def build_warehouse(raw: RawData, db_path: Path, marts_dir: Path, as_of: pd.Period, detail_months: int = 13) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    marts_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for key, name in RAW_TABLE_NAMES.items():
        df = raw.tables[key]  # noqa: F841  (referenced by DuckDB's replacement scan)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
    ran = run_sql_models(con)
    checks = reconcile(con)

    for t in MART_TABLES:
        con.execute(f"COPY {t} TO '{marts_dir / (t + '.parquet')}' (FORMAT parquet)")
    first_detail = str(as_of - (detail_months - 1))
    con.execute(
        f"""COPY (SELECT period, je_id, posting_date, entity, account, account_name, fs_line, department,
                         cost_center, cost_center_name, vendor_id, vendor_name, customer_id, document_number,
                         description, source, is_reversal, ROUND(amount_usd, 2) AS amount_usd
                  FROM fct_gl_pnl_lines WHERE period >= '{first_detail}' ORDER BY period, je_id)
            TO '{marts_dir / 'gl_detail_recent.parquet'}' (FORMAT parquet)"""
    )
    con.close()
    return {"models": ran, "reconciliation": checks}
