"""Make the ERP extract look like a real one: inconsistent text, missing cost centers and
duplicated lines from an overlapping extract window. The ERP's own control totals are computed
before the extract duplicates are introduced, so the pipeline can reconcile against them."""

from __future__ import annotations

import numpy as np
import pandas as pd


def control_totals(gl: pd.DataFrame) -> pd.DataFrame:
    ct = gl.groupby(["period", "entity"], as_index=False).agg(
        line_count=("je_id", "size"), total_debit=("debit", "sum"), total_credit=("credit", "sum")
    )
    ct[["total_debit", "total_credit"]] = ct[["total_debit", "total_credit"]].round(2)
    return ct


def add_messiness(gl: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    parts, stats = [], {"duplicated_extract_lines": 0, "blank_cost_center": 0, "invalid_cost_center": 0}
    for p, g in gl.groupby("period", sort=True):
        rng = np.random.default_rng([seed, 5, p.year, p.month])
        g = g.copy()
        n = len(g)
        is_ap_expense = (g["source"] == "AP").to_numpy() & (g["cost_center"] != "").to_numpy()
        u = rng.random(n)
        blank = is_ap_expense & (u < 0.003)
        invalid = is_ap_expense & (u >= 0.003) & (u < 0.004)
        g.loc[blank, "cost_center"] = ""
        g.loc[invalid, "cost_center"] = "CC-999"
        stats["blank_cost_center"] += int(blank.sum())
        stats["invalid_cost_center"] += int(invalid.sum())

        u = rng.random(n)
        has_vendor = (g["vendor_name"] != "").to_numpy()
        g.loc[has_vendor & (u < 0.05), "vendor_name"] = g.loc[has_vendor & (u < 0.05), "vendor_name"].str.upper()
        sel = has_vendor & (u >= 0.05) & (u < 0.08)
        g.loc[sel, "vendor_name"] = g.loc[sel, "vendor_name"] + "  "
        u = rng.random(n)
        g.loc[u < 0.04, "description"] = g.loc[u < 0.04, "description"].str.upper()
        sel = (u >= 0.04) & (u < 0.07)
        g.loc[sel, "description"] = " " + g.loc[sel, "description"].str.replace(" - ", "  -  ", regex=False)

        dup = g[rng.random(n) < 0.0008]
        stats["duplicated_extract_lines"] += len(dup)
        parts.append(pd.concat([g, dup]).sort_values(["je_id", "line_no"], kind="stable"))
    return pd.concat(parts, ignore_index=True), stats


def to_erp_csv_format(gl: pd.DataFrame) -> pd.DataFrame:
    """Format like a typical ERP CSV export: US dates, blank zeros, Y/N flags."""
    out = gl.drop(columns=["_item"]).copy()
    out["period"] = out["period"].astype(str)
    out["posting_date"] = out["posting_date"].dt.strftime("%m/%d/%Y")
    for c in ("debit", "credit"):
        out[c] = out[c].map(lambda x: f"{x:.2f}" if x else "")
    out["is_reversal"] = out["is_reversal"].map({True: "Y", False: "N"})
    return out
