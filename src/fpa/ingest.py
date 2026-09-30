"""Load the raw ERP extracts, enforce schemas and produce a data-quality report.

Hard failures (the close cannot proceed) raise. Soft issues are fixed downstream and reported,
so nothing is dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

SOURCES = ["AP", "AR", "REVREC", "PAYROLL", "ACCRUAL", "FA", "MANUAL"]
PERIOD_RE = r"^\d{4}-(0[1-9]|1[0-2])$"

GL_SCHEMA = pa.DataFrameSchema(
    {
        "je_id": pa.Column(str, pa.Check.str_matches(r"^JE\d{10,11}$")),
        "line_no": pa.Column(int, pa.Check.ge(1)),
        "period": pa.Column(str, pa.Check.str_matches(PERIOD_RE)),
        "posting_date": pa.Column("datetime64[ns]"),
        "entity": pa.Column(str),
        "account": pa.Column(str, pa.Check.str_matches(r"^\d{4}$")),
        "cost_center": pa.Column(str),
        "debit": pa.Column(float, pa.Check.ge(0)),
        "credit": pa.Column(float, pa.Check.ge(0)),
        "currency": pa.Column(str, pa.Check.isin(["USD", "GBP", "EUR"])),
        "source": pa.Column(str, pa.Check.isin(SOURCES)),
        "is_reversal": pa.Column(bool),
    },
    checks=[pa.Check(lambda df: (df["debit"] > 0) ^ (df["credit"] > 0), error="exactly one of debit/credit per line")],
    strict=False,
)
BUDGET_SCHEMA = pa.DataFrameSchema(
    {
        "fiscal_year": pa.Column(int),
        "period": pa.Column(str, pa.Check.str_matches(PERIOD_RE)),
        "account": pa.Column(str),
        "cost_center": pa.Column(str),
        "amount_usd": pa.Column(float),
    },
    strict=False,
)


@dataclass
class RawData:
    tables: dict[str, pd.DataFrame]
    dq: dict = field(default_factory=dict)


def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)


def load_raw(raw_dir: Path) -> RawData:
    t: dict[str, pd.DataFrame] = {}
    gl = _read(raw_dir / "gl_journal_lines.csv")
    gl["line_no"] = gl["line_no"].astype(int)
    gl["posting_date"] = pd.to_datetime(gl["posting_date"], format="%m/%d/%Y")
    for c in ("debit", "credit"):
        gl[c] = pd.to_numeric(gl[c].replace("", "0"))
    gl["is_reversal"] = gl["is_reversal"].eq("Y")
    t["gl"] = GL_SCHEMA.validate(gl, lazy=True)

    t["control_totals"] = _read(raw_dir / "gl_control_totals.csv").astype(
        {"line_count": int, "total_debit": float, "total_credit": float}
    )
    for name in ("chart_of_accounts", "cost_centers", "vendors", "entities"):
        t[name] = _read(raw_dir / f"{name}.csv")
    t["fx"] = _read(raw_dir / "fx_rates.csv").astype({"usd_per_unit": float})
    budgets = [_read(f) for f in sorted(raw_dir.glob("budget_fy*.csv"))]
    t["budget"] = BUDGET_SCHEMA.validate(
        pd.concat(budgets, ignore_index=True).astype({"fiscal_year": int, "amount_usd": float}), lazy=True
    )
    t["hris"] = _read(raw_dir / "hris_headcount.csv").astype({"headcount": int, "hires": int, "terminations": int})
    t["hc_plan"] = _read(raw_dir / "hris_headcount_plan.csv").astype({"planned_headcount": int})
    arr = _read(raw_dir / "crm_arr.csv")
    num = [c for c in arr.columns if c not in ("period", "entity")]
    t["arr"] = arr.astype(dict.fromkeys(num, float))
    return RawData(t)


def validate(raw: RawData) -> RawData:
    """Business-rule checks. Populates raw.dq; raises only on issues that block the close."""
    gl, coa, cc = raw.tables["gl"], raw.tables["chart_of_accounts"], raw.tables["cost_centers"]
    dq: dict = {}
    problems: list[str] = []

    key = ["je_id", "line_no"]
    dup_mask = gl.duplicated(key, keep="first")
    dq["extract_duplicate_lines"] = int(dup_mask.sum())
    deduped = gl[~dup_mask]

    def unbalanced(df):
        s = df.groupby("je_id")[["debit", "credit"]].sum()
        return s[(s["debit"] - s["credit"]).abs() > 0.005]

    dq["unbalanced_jes_before_dedupe"] = len(unbalanced(gl))
    dq["unbalanced_jes_after_dedupe"] = len(unbalanced(deduped))
    if dq["unbalanced_jes_after_dedupe"]:
        problems.append(f"{dq['unbalanced_jes_after_dedupe']} journal entries do not balance")

    ct = raw.tables["control_totals"].set_index(["period", "entity"])
    loaded = deduped.groupby(["period", "entity"]).agg(
        line_count=("je_id", "size"), total_debit=("debit", "sum"), total_credit=("credit", "sum")
    )
    diff = (loaded - ct).abs()
    breaks = diff[(diff["line_count"] > 0) | (diff["total_debit"] > 0.01) | (diff["total_credit"] > 0.01)]
    dq["control_total_breaks"] = len(breaks)
    if len(breaks):
        problems.append(f"control totals do not reconcile for {len(breaks)} period/entity pairs")

    unknown_accts = set(gl["account"]) - set(coa["account"])
    if unknown_accts:
        problems.append(f"accounts missing from chart of accounts: {sorted(unknown_accts)}")

    pnl_accounts = set(coa.loc[coa["fs_line"] != "", "account"])
    pnl = deduped[deduped["account"].isin(pnl_accounts)]
    dq["pnl_lines_blank_cost_center"] = int((pnl["cost_center"] == "").sum())
    dq["pnl_lines_invalid_cost_center"] = int((~pnl["cost_center"].isin(cc["cost_center"]) & (pnl["cost_center"] != "")).sum())

    ap = deduped[(deduped["source"] == "AP") & (deduped["debit"] > 0) & (deduped["document_number"] != "")]
    inv = ap.groupby(["vendor_id", "document_number"]).agg(jes=("je_id", "nunique"), amount=("debit", "sum"))
    suspects = inv[inv["jes"] > 1].reset_index()
    dq["possible_duplicate_invoices"] = suspects.assign(amount=suspects["amount"].round(2)).to_dict("records")

    period_start = pd.PeriodIndex(deduped["period"], freq="M").to_timestamp()
    lag_days = (deduped["posting_date"] - period_start).dt.days
    dq["lines_posted_outside_close_window"] = int(((lag_days < 0) | (lag_days > 65)).sum())

    dq["vendor_name_variants"] = int(
        deduped.loc[deduped["vendor_name"] != "", ["vendor_id", "vendor_name"]].drop_duplicates().shape[0]
        - deduped.loc[deduped["vendor_id"] != "", "vendor_id"].nunique()
    )
    dq["rows_loaded"] = len(gl)
    dq["rows_after_dedupe"] = len(deduped)
    dq["periods"] = f"{gl['period'].min()}..{gl['period'].max()}"

    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
    raw.dq = dq
    return raw
