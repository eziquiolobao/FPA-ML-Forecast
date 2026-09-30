"""Shape the P&L marts into model-ready time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SERIES_KEYS = ["fs_line", "department"]


def series_id(fs_line: pd.Series, department: pd.Series) -> pd.Series:
    return fs_line + " | " + department


def to_ds(period: pd.Series) -> pd.Series:
    return pd.PeriodIndex(period, freq="M").to_timestamp()


def load_actuals(marts_dir: Path, as_of: pd.Period) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (y: unique_id/ds/y on a complete monthly grid, meta: one row per series)."""
    m = pd.read_parquet(marts_dir / "fct_monthly_pnl.parquet")
    m = m[m["period"] <= str(as_of)]
    m["unique_id"] = series_id(m["fs_line"], m["department"])
    meta = m[["unique_id", "fs_line", "department", "function", "account_type"]].drop_duplicates("unique_id")
    m["ds"] = to_ds(m["period"])
    grid = pd.MultiIndex.from_product(
        [meta["unique_id"], pd.date_range(m["ds"].min(), as_of.to_timestamp(), freq="MS")], names=["unique_id", "ds"]
    )
    y = m.groupby(["unique_id", "ds"])["actual_usd"].sum().reindex(grid, fill_value=0.0).rename("y").reset_index()
    return y, meta.sort_values("unique_id").reset_index(drop=True)


def load_budget(marts_dir: Path, as_of: pd.Period | None = None) -> pd.DataFrame:
    b = pd.read_parquet(marts_dir / "fct_budget_monthly.parquet")
    b["unique_id"] = series_id(b["fs_line"], b["department"])
    b["ds"] = to_ds(b["period"])
    return b.groupby(["unique_id", "ds"], as_index=False)["budget_usd"].sum()


def headcount_growth_feature(marts_dir: Path, meta: pd.DataFrame, origin: pd.Period, horizon: int) -> pd.DataFrame:
    """Monthly headcount growth by series, as known at `origin`.

    History uses actual HRIS headcount; future months use the hiring plan from budgets already
    released by the origin date (flat beyond the last released plan). Revenue series use total
    company headcount.
    """
    hc = pd.read_parquet(marts_dir / "fct_headcount.parquet")
    hc["ds"] = to_ds(hc["period"])
    released_fys = _released_fiscal_years(marts_dir, origin)
    end = (origin + horizon).to_timestamp()
    months = pd.date_range(hc["ds"].min(), end, freq="MS")

    def path(df: pd.DataFrame) -> pd.Series:
        act = df.set_index("ds")["headcount"].where(df.set_index("ds").index <= origin.to_timestamp())
        plan = df.set_index("ds")["planned_headcount"]
        plan = plan[plan.index.year.isin(released_fys) & (plan.index > origin.to_timestamp())]
        s = act.dropna().combine_first(plan).reindex(months).ffill()
        return s.pct_change().fillna(0.0)

    dept = {d: path(g) for d, g in hc.groupby("department")}
    total = path(hc.groupby("ds", as_index=False)[["headcount", "planned_headcount"]].sum(min_count=1))
    rows = []
    for r in meta.itertuples():
        s = dept.get(r.department, total)
        rows.append(pd.DataFrame({"unique_id": r.unique_id, "ds": s.index, "hc_growth": s.to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out["hc_growth"] = out["hc_growth"].replace([np.inf, -np.inf], 0.0)
    return out


def _released_fiscal_years(marts_dir: Path, origin: pd.Period) -> list[int]:
    b = pd.read_parquet(marts_dir / "fct_budget_monthly.parquet", columns=["fiscal_year"])
    fys = sorted(b["fiscal_year"].unique())
    # the plan for FY N is released in November of N-1
    return [fy for fy in fys if pd.Period(f"{fy - 1}-11", "M") <= origin]
