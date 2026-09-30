"""Annual operating plan (AOP / budget) built the way FP&A teams actually build one.

Each November the plan for the next fiscal year is locked. It starts from the latest actuals
(October), assumes trailing ARR growth continues with a little optimism, and plans headcount a
bit ahead of what recruiting will actually deliver. It knows nothing about future shocks
(customer churn, price increases, anomalies). Some lines are phased flat across the year, which
creates realistic timing variances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpa.generate.company import CHART_OF_ACCOUNTS, COST_CENTERS, ENTITY_REVENUE_SHARE
from fpa.generate.drivers import BOOKINGS_SEASONALITY, Drivers
from fpa.generate.lines import ENTITY_CCY, FX_CONST, HC_CCS, Context, expected_items

FLAT_PHASED_ITEMS = {
    "services", "outside_counsel", "finance_consultants", "field_events", "tax_advisory",
    "demand_gen", "eng_contractors", "data_contractors",
}
PLAN_BIAS = {  # systematic planning error by P&L line (log scale)
    "Services Revenue": 0.06, "Hosting": -0.04, "Marketing Programs": 0.03,
    "Travel & Entertainment": -0.03, "Contractors": -0.05,
}
GROWTH_OPTIMISM = 1.25  # board-approved stretch on top of the trailing growth rate
HIRING_AMBITION = 0.05
PLANNED_MERIT = 0.035


def _plan_context(fy: int, drivers: Drivers, actual_ctx: Context) -> tuple[Context, pd.Period]:
    first = drivers.periods[0]
    base = max(pd.Period(f"{fy - 1}-10", "M"), first)
    arr = drivers.arr_bridge.pivot(index="period", columns="entity", values="ending_arr")
    arr_total = arr.sum(axis=1)
    arr_base = float(arr_total[base])
    share = (arr.loc[base] / arr_base).to_dict() if base in arr.index else ENTITY_REVENUE_SHARE
    prior = base - 12
    growth = arr_base / float(arr_total[prior]) - 1 if prior in arr_total.index else 0.45
    r = (1 + growth * GROWTH_OPTIMISM) ** (1 / 12) - 1

    start = base if base == first else base + 1
    pers = pd.period_range(start, f"{fy}-12", freq="M")
    arr_plan, prev = [], arr_base
    for p in pers:
        cur = prev * (1 + r * BOOKINGS_SEASONALITY[p.month - 1]) if p != base else prev
        arr_plan.append((prev, cur))
        prev = cur
    arr_open = np.array([a for a, _ in arr_plan])
    arr_close = np.array([b for _, b in arr_plan])

    sub_rev = pd.DataFrame({e: (arr_open + arr_close) / 2 / 12 * share[e] for e in ENTITY_REVENUE_SHARE}, index=pers)
    seas = np.array([BOOKINGS_SEASONALITY[p.month - 1] for p in pers])
    bookings = pd.DataFrame({e: arr_open * (r * seas + 0.0058) * share[e] for e in ENTITY_REVENUE_SHARE}, index=pers)

    # the hiring plan assumes every approved req is filled on time (no recruiting lag), plus the
    # hiring team's ambition. Reality: hires land ~3 months later, so payroll usually under-runs.
    arr_path = np.maximum(arr_close, 1.0)
    ramp = np.minimum(1.0, (np.arange(len(pers)) + 1) / 6)
    hc_base = actual_ctx.hc.loc[base]
    el = COST_CENTERS.set_index("cost_center")["hc_elasticity"]
    hc = pd.DataFrame(
        {
            cc: np.round(hc_base[cc] * np.maximum(arr_path / arr_base, 1.0) ** el[cc] * (1 + HIRING_AMBITION * ramp))
            for cc in HC_CCS
        },
        index=pers,
    )
    hires = (hc.diff().clip(lower=0).fillna(0) + 0.012 * hc.shift(1).fillna(hc)).sum(axis=1).round()

    sal_base = actual_ctx.salary.loc[base]
    merit = np.array([1 + PLANNED_MERIT if (p.month >= 4 and p.year == fy and base < pd.Period(f"{fy}-04", "M")) else 1.0 for p in pers])
    salary = pd.DataFrame({cc: sal_base[cc] * merit for cc in HC_CCS}, index=pers)
    return Context(pers, sub_rev, bookings, hc, hires, salary), base


def build_budgets(drivers: Drivers, actual_ctx: Context, seed: int, last_fy: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (budget lines, headcount plan) for every fiscal year from the first to `last_fy`."""
    rng = np.random.default_rng([seed, 6])
    fs_line = CHART_OF_ACCOUNTS.set_index("account")["fs_line"]
    fx = drivers.fx.pivot(index="period", columns="currency", values="usd_per_unit")
    budgets, hc_plans = [], []
    for fy in range(drivers.periods[0].year, last_fy + 1):
        ctx, base = _plan_context(fy, drivers, actual_ctx)
        month_index = np.array([(p - pd.Period("2021-01", "M")).n for p in ctx.periods])
        items = expected_items(ctx, month_index)
        items = items[items["period"].dt.year == fy].copy()

        flat = items["item"].isin(FLAT_PHASED_ITEMS)
        keys = ["entity", "cost_center", "account", "item"]
        items.loc[flat, "amount"] = items[flat].groupby(keys)["amount"].transform("mean")

        items["fs_line"] = items["account"].map(fs_line)
        grp = items[["fs_line", "cost_center"]].drop_duplicates().sort_values(["fs_line", "cost_center"])
        grp["err"] = [rng.normal(PLAN_BIAS.get(f, 0.0), 0.06) for f in grp["fs_line"]]
        items = items.merge(grp, on=["fs_line", "cost_center"])
        items["amount"] *= np.exp(items["err"])

        rates = {e: fx.loc[base, c] / FX_CONST[c] for e, c in ENTITY_CCY.items()}  # plan FX = rate at plan date
        plan_rate = items["entity"].map(rates)
        items["amount_usd"] = items["amount"] * plan_rate
        released = pd.Timestamp(f"{fy - 1}-11-15")
        b = items.groupby(["period", "entity", "account", "cost_center"], as_index=False)["amount_usd"].sum()
        b.insert(0, "released_on", released.date())
        b.insert(0, "version", f"AOP FY{fy}")
        b.insert(0, "fiscal_year", fy)
        b["amount_usd"] = b["amount_usd"].round(2)
        budgets.append(b)

        h = ctx.hc.loc[ctx.hc.index.year == fy].stack().rename("planned_headcount").reset_index()
        h.columns = ["period", "cost_center", "planned_headcount"]
        h.insert(0, "released_on", released.date())
        h.insert(0, "plan_version", f"AOP FY{fy}")
        h["planned_headcount"] = h["planned_headcount"].astype(int)
        hc_plans.append(h)
    return pd.concat(budgets, ignore_index=True), pd.concat(hc_plans, ignore_index=True)
