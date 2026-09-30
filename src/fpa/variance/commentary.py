"""First-draft variance commentary from templates - no LLM, fully reproducible."""

from __future__ import annotations

import math

import pandas as pd


def money(x: float) -> str:
    a = abs(x)
    if a >= 1e6:
        s = f"${a / 1e6:.2f}M"
    elif a >= 1e3:
        s = f"${a / 1e3:.0f}k"
    else:
        s = f"${a:.0f}"
    return f"-{s}" if x < 0 else s


def pct(x: float) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{abs(x):.0%}"


def write_commentary(row: pd.Series, drivers: pd.DataFrame | None, split: dict | None) -> str:
    month = pd.Period(row["period"], "M").strftime("%b-%y")
    name = f"{row['fs_line']} ({row['department']})" if row["department"] != "Company" else row["fs_line"]
    if not math.isfinite(row["budget_usd"]):
        return f"{name}: no budget for {month}."
    direction = "over" if row["var_budget_usd"] > 0 else "under"
    tone = "favourable" if row["favourable"] else "unfavourable"
    parts = [
        f"{name} was {money(abs(row['var_budget_usd']))} ({pct(row['var_budget_pct'])}) {direction} budget in {month} ({tone})."
    ]
    if row["outside_range"]:
        parts.append(
            f"The actual of {money(row['actual_usd'])} fell outside the expected range of "
            f"{money(row['range_lo_usd'])} to {money(row['range_hi_usd'])} from last month's forecast"
            + (" - likely a one-off or error to investigate." if row["severity"] == "Red" else " - worth a quick check.")
        )
    elif math.isfinite(row.get("forecast_usd", float("nan"))):
        parts.append("It was in line with last month's forecast, so the gap reflects the original plan rather than a surprise.")
    if row["drift_months"] >= 3:
        parts.append(f"This is the {ordinal(int(row['drift_months']))} consecutive month {direction} budget.")
    if split:
        heads = split["hc_actual"] - split["hc_plan"]
        parts.append(
            f"Headcount was {int(split['hc_actual'])} vs a plan of {int(split['hc_plan'])} "
            f"({'+' if heads >= 0 else ''}{int(heads)} heads = {money(split['volume_usd'])}); "
            f"cost per head accounted for {money(split['rate_usd'])}."
        )
    if drivers is not None and not drivers.empty:
        d = drivers.iloc[0]
        if abs(d["change_usd"]) >= 1_000:
            docs = f" ({int(d['documents'])} document{'s' if d['documents'] != 1 else ''})" if d["documents"] else ""
            parts.append(f"Biggest change vs a typical recent month: {d['driver']} {'+' if d['change_usd'] >= 0 else ''}{money(d['change_usd'])}{docs}.")
    return " ".join(parts)


def ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}.get(n, f"{n}th")
