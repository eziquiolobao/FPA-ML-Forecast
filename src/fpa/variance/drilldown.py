"""Explain a variance: which vendors / customers / postings moved, and for payroll lines how
much came from headcount (volume) versus cost per head (rate)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_STRIP = [
    (re.compile(r"\b\d{2}/\d{2}/\d{4}\b"), ""),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b", re.I), ""),
    (re.compile(r"\bE\d{6}\b"), ""),
    (re.compile(r"\s+-\s*$"), ""),
    (re.compile(r"\s+"), " "),
    # accruals, their reversals and late invoices belong to the same spend item
    (re.compile(r"^\s*(reversal - accrual|accrual -|late invoice -)\s*", re.I), ""),
]


def driver_label(df: pd.DataFrame) -> pd.Series:
    """Business-friendly label for grouping postings: what was bought or billed (a normalised
    description), so an accrual, its reversal and the eventual invoice net into one driver."""
    desc = df["description"].fillna("").str.strip()
    for pat, rep in _STRIP:
        desc = desc.str.replace(pat, rep, regex=True)
    desc = desc.str.strip()
    label = desc.str[:1].str.upper() + desc.str[1:]
    label = label.where(df["source"] != "REVREC", "Recurring subscription billing")
    label = label.where(~df["description"].fillna("").str.strip().str.lower().str.startswith("expense report"), "Employee expense reports")
    return label


def top_drivers(detail: pd.DataFrame, fs_line: str, department: str, period: str, n: int = 3) -> pd.DataFrame:
    """Change by driver in `period` versus a typical recent month (median of the prior 3 months,
    so a one-off like January's annual renewals doesn't distort the comparison)."""
    d = detail[(detail["fs_line"] == fs_line) & (detail["department"] == department)]
    p = pd.Period(period, "M")
    prior = [str(p - k) for k in (1, 2, 3)]
    d = d[d["period"].isin([period, *prior])].copy()
    if d.empty:
        return pd.DataFrame(columns=["driver", "current_usd", "typical_usd", "change_usd", "documents"])
    label = driver_label(d)
    # group case-insensitively (the extract has inconsistent casing); display the best-cased variant
    d["driver"] = label.str.lower()
    display = label.groupby(d["driver"]).agg(lambda s: s[~s.str.isupper()].mode().iat[0] if (~s.str.isupper()).any() else s.iat[0].title())
    cur = d[d["period"] == period].groupby("driver").agg(current_usd=("amount_usd", "sum"), documents=("je_id", "nunique"))
    by_month = d[d["period"].isin(prior)].pivot_table(index="driver", columns="period", values="amount_usd", aggfunc="sum")
    typical = by_month.reindex(columns=prior).fillna(0.0).median(axis=1)
    out = cur.join(typical.rename("typical_usd"), how="outer").fillna(0.0)
    out.index = out.index.map(display)
    out["change_usd"] = out["current_usd"] - out["typical_usd"]
    out["documents"] = out["documents"].astype(int)
    out = out.loc[out["change_usd"].abs() >= 1, ["current_usd", "typical_usd", "change_usd", "documents"]]
    out = out.reindex(out["change_usd"].abs().sort_values(ascending=False).index).head(n)
    return out.rename_axis("driver").reset_index().round(2)


def headcount_rate_split(actual: float, budget: float, hc_actual: float, hc_plan: float) -> dict | None:
    """Personnel variance = volume (heads vs plan at planned cost) + rate (cost per head)."""
    if not all(np.isfinite([actual, budget, hc_actual, hc_plan])) or hc_plan <= 0 or hc_actual <= 0:
        return None
    plan_rate, act_rate = budget / hc_plan, actual / hc_actual
    return {
        "hc_actual": hc_actual,
        "hc_plan": hc_plan,
        "volume_usd": round((hc_actual - hc_plan) * plan_rate, 2),
        "rate_usd": round((act_rate - plan_rate) * hc_actual, 2),
    }
