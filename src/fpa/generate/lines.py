"""Monthly P&L amounts per (period, entity, cost center, account, item).

`expected_items` turns a driver context into noiseless amounts. The same formulas serve the
simulated actuals (actual drivers + noise + one-offs) and the budget (plan drivers, no noise), so
budget variances come from wrong assumptions, just as they do in real life.

All amounts are in "constant USD", i.e. local currency converted at FX_CONST. The GL writer
converts them back to local currency, and the pipeline re-translates at actual monthly rates, so
FX movements show up as genuine variances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpa.generate.company import COST_CENTERS, ENTITY_REVENUE_SHARE
from fpa.generate.drivers import Drivers

FX_CONST = {"USD": 1.0, "GBP": 1.36, "EUR": 1.21}
ENTITY_CCY = {"US01": "USD", "UK01": "GBP", "DE01": "EUR"}
CC_ENTITY = COST_CENTERS.set_index("cost_center")["entity"].to_dict()
CC_DEPT = COST_CENTERS.set_index("cost_center")["department"].to_dict()
HC_CCS = COST_CENTERS.loc[COST_CENTERS["hc_start"] > 0, "cost_center"].tolist()

BONUS_PCT = {
    "Sales": 0.02, "Marketing": 0.08, "Customer Success": 0.07, "Engineering": 0.10, "Product": 0.10,
    "G&A": 0.12, "Support": 0.05, "Professional Services": 0.07, "Cloud Operations": 0.10,
}
# US employer taxes are front-loaded because social-security wage caps are hit later in the year
US_TAX_RATE = [0.27, 0.26, 0.25, 0.24, 0.235, 0.23, 0.225, 0.22, 0.215, 0.21, 0.20, 0.20]
INTL_TAX_RATE = {"UK01": 0.24, "DE01": 0.21}
TE_RATE = {"CC-110": 1300, "CC-120": 1400, "CC-310": 600, "CC-210": 450, "CC-610": 250}
TE_SEASON = [1.0, 1.05, 1.1, 1.0, 1.05, 0.95, 0.8, 0.75, 1.1, 1.1, 1.05, 0.7]
MKT_SEASON = np.array([1.1, 1.05, 1.1, 1.0, 0.95, 0.9, 0.8, 0.85, 1.1, 1.15, 1.1, 0.9])
MKT_SEASON = MKT_SEASON / MKT_SEASON.mean()
NEW_HQ = pd.Period("2023-07", "M")

# Multiplicative noise (log-sd) applied to simulated actuals, by item
NOISE_SD = {
    "services": 0.18, "cloud": 0.035, "monitoring": 0.05, "salaries": 0.008, "bonus_accrual": 0.02,
    "payroll_taxes": 0.02, "commissions": 0.05, "eng_contractors": 0.15, "data_contractors": 0.20,
    "ps_subcontractors": 0.12, "finance_consultants": 0.30, "per_seat_saas": 0.04,
    "annual_renewals": 0.06, "dev_tools": 0.05, "demand_gen": 0.12, "trade_show": 0.10,
    "user_conference": 0.08, "field_events": 0.35, "travel": 0.10, "sales_kickoff": 0.10,
    "rent": 0.0, "facility_services": 0.12, "audit": 0.08, "tax_advisory": 0.20,
    "outside_counsel": 0.35, "depreciation": 0.005, "recruiting": 0.20,
}


@dataclass
class Context:
    """Per-period driver values (index = period)."""

    periods: pd.PeriodIndex
    sub_rev: pd.DataFrame  # columns = entity, monthly subscription revenue
    bookings: pd.DataFrame  # columns = entity, commissionable ARR booked in the month
    hc: pd.DataFrame  # columns = cost center, month-end headcount
    hires: pd.Series  # total hires in the month
    salary: pd.DataFrame  # columns = cost center, annual salary per head


def customer_revenue(drivers: Drivers) -> pd.DataFrame:
    """Monthly subscription revenue by customer; new customers get a half month."""
    cr = drivers.customer_arr.copy()
    first = cr.groupby("customer_id")["period"].transform("min")
    is_new = (cr["period"] == first) & (cr["period"] != drivers.periods[0])
    cr["revenue_usd"] = np.where(is_new, cr["arr_usd"] / 24, cr["arr_usd"] / 12).round(2)
    return cr


def context_from_actuals(drivers: Drivers, cust_rev: pd.DataFrame) -> Context:
    pers = drivers.periods
    sub_rev = cust_rev.pivot_table(index="period", columns="entity", values="revenue_usd", aggfunc="sum")
    # commissionable bookings: full credit on new logos, partial credit on expansion (no credit on churn)
    b = drivers.arr_bridge.assign(bookings=lambda x: x["new_arr"] + 0.35 * x["expansion_arr"])
    bookings = b.pivot_table(index="period", columns="entity", values="bookings", aggfunc="sum")
    hc = drivers.headcount.pivot(index="period", columns="cost_center", values="headcount")
    salary = drivers.headcount.pivot(index="period", columns="cost_center", values="salary_usd")
    hires = drivers.headcount.groupby("period")["hires"].sum()
    return Context(pers, sub_rev.reindex(pers), bookings.reindex(pers), hc.reindex(pers), hires.reindex(pers), salary.reindex(pers))


def _covid_travel(p: pd.Period) -> float:
    if p.year == 2021:
        return 0.25 if p.month <= 6 else 0.45
    if p.year == 2022 and p.month <= 6:
        return 0.75
    return 1.0


def _covid_events(p: pd.Period) -> float:
    if p.year == 2021:
        return 0.25
    if p.year == 2022 and p.month <= 6:
        return 0.6
    return 1.0


def _hosting_ratio(i: int) -> float:
    # unit-cost efficiency program: 15.0% of subscription revenue in 2021 -> 11.5% by end of 2026
    return max(0.113, 0.150 - (0.150 - 0.115) * i / 71)


def _rent(p: pd.Period) -> float:
    if p < NEW_HQ:
        base, start = 110_000, pd.Period("2021-01", "M")
    else:
        base, start = 165_000, NEW_HQ
    julys = sum(1 for y in range(start.year, p.year + 1) if pd.Period(f"{y}-07", "M") <= p and pd.Period(f"{y}-07", "M") > start)
    return base * 1.03**julys


def expected_items(ctx: Context, month_index: np.ndarray | None = None) -> pd.DataFrame:
    """Noiseless monthly amounts for every P&L item. `month_index` = months since Jan-2021."""
    pers = ctx.periods
    if month_index is None:
        month_index = np.array([(p - pd.Period("2021-01", "M")).n for p in pers])
    rows: list[tuple] = []

    def add(p, entity, cc, account, item, amount):
        if amount:
            rows.append((p, entity, cc, account, item, float(amount)))

    hc_prev = ctx.hc.shift(1).fillna(ctx.hc)
    hc_avg = (ctx.hc + hc_prev) / 2
    total_sub = ctx.sub_rev.sum(axis=1)

    for k, p in enumerate(pers):
        i, m, yrs = int(month_index[k]), p.month, p.year - 2021
        sub = float(total_sub.iloc[k])
        qe = 1.25 if m in (3, 6, 9, 12) else 0.875
        services = 0.085 * sub * qe

        # Revenue
        for ent in ENTITY_REVENUE_SHARE:
            add(p, ent, "CC-000", "4000", "subscription", ctx.sub_rev[ent].iloc[k])
        add(p, "US01", "CC-000", "4100", "services", 0.80 * services)
        add(p, "UK01", "CC-000", "4100", "services", 0.20 * services)

        # Hosting (cost of revenue)
        add(p, "US01", "CC-810", "5000", "cloud", sub * _hosting_ratio(i) * (1.03 if m in (11, 12) else 1.0))
        add(p, "US01", "CC-810", "5010", "monitoring", 30_000 * 1.018**i)

        # Personnel
        for cc in HC_CCS:
            ent, dept = CC_ENTITY[cc], CC_DEPT[cc]
            salaries = hc_avg[cc].iloc[k] * ctx.salary[cc].iloc[k] / 12
            tax = US_TAX_RATE[m - 1] if ent == "US01" else INTL_TAX_RATE[ent]
            add(p, ent, cc, "6000", "salaries", salaries)
            add(p, ent, cc, "6010", "bonus_accrual", salaries * BONUS_PCT[dept])
            add(p, ent, cc, "6020", "payroll_taxes", salaries * tax)

        # Commissions on bookings, with Q4 accelerators
        accel = 1.2 if m == 12 else 1.0
        add(p, "US01", "CC-110", "6100", "commissions", 0.14 * ctx.bookings["US01"].iloc[k] * accel)
        emea = ctx.bookings["UK01"].iloc[k] + ctx.bookings["DE01"].iloc[k]
        add(p, "UK01", "CC-120", "6100", "commissions", 0.14 * emea * accel)

        # Contractors
        add(p, "US01", "CC-410", "6200", "eng_contractors", 60_000 * 1.01**i)
        add(p, "DE01", "CC-420", "6200", "data_contractors", 30_000 * 1.008**i)
        add(p, "US01", "CC-720", "6200", "ps_subcontractors", 0.30 * services)
        add(p, "US01", "CC-610", "6200", "finance_consultants", 18_000 * 1.006**i * (1.6 if m in (1, 2, 11, 12) else 0.8))

        # Software & tools
        total_hc = float(ctx.hc.iloc[k].sum())
        add(p, "US01", "CC-640", "6300", "per_seat_saas", 150 * total_hc * 1.03**yrs)
        if m == 1:
            add(p, "US01", "CC-640", "6300", "annual_renewals", 380_000 * 1.15**yrs)
        if m == 7:
            add(p, "US01", "CC-640", "6300", "annual_renewals", 110_000 * 1.12**yrs)
        eng_hc = float(ctx.hc["CC-410"].iloc[k] + ctx.hc["CC-420"].iloc[k])
        add(p, "US01", "CC-410", "6300", "dev_tools", 280 * eng_hc)

        # Marketing programs & events
        add(p, "US01", "CC-210", "6400", "demand_gen", 0.042 * sub * MKT_SEASON[m - 1])
        ev = _covid_events(p) * 1.1**yrs
        if m == 5:
            add(p, "US01", "CC-210", "6410", "trade_show", 170_000 * ev)
        if m == 10:
            add(p, "US01", "CC-210", "6410", "user_conference", 300_000 * ev)
        add(p, "US01", "CC-210", "6410", "field_events", 12_000 * ev)

        # Travel & entertainment (with the January sales kickoff)
        for cc, rate in TE_RATE.items():
            travel = hc_avg[cc].iloc[k] * rate * 1.04**yrs * TE_SEASON[m - 1] * _covid_travel(p)
            add(p, CC_ENTITY[cc], cc, "6500", "travel", travel)
            if m == 1 and cc in ("CC-110", "CC-120"):
                add(p, CC_ENTITY[cc], cc, "6500", "sales_kickoff", ctx.hc[cc].iloc[k] * 2_200 * _covid_travel(p))

        # Facilities, professional fees, depreciation, recruiting
        add(p, "US01", "CC-610", "6600", "rent", _rent(p))
        add(p, "US01", "CC-610", "6600", "facility_services", 11_000 * 1.005**i)
        audit_phase = {2: 0.35, 3: 0.40, 4: 0.25}.get(m, 0.0)
        add(p, "US01", "CC-610", "6700", "audit", 240_000 * 1.08**yrs * audit_phase)
        add(p, "US01", "CC-610", "6700", "tax_advisory", 14_000 * 1.06**yrs + (45_000 if m == 10 else 0))
        add(p, "US01", "CC-630", "6700", "outside_counsel", 40_000 * 1.07**yrs)
        add(p, "US01", "CC-610", "6800", "depreciation", 52_000 * 1.012**i + (38_000 if p >= NEW_HQ else 0))
        add(p, "US01", "CC-620", "6900", "recruiting", 8_000 + float(ctx.hires.iloc[k]) * 4_800 * 1.03**yrs)

    return pd.DataFrame(rows, columns=["period", "entity", "cost_center", "account", "item", "amount"])


def actual_items(drivers: Drivers, cust_rev: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Simulated actuals = expected amounts x noise + bonus true-ups."""
    ctx = context_from_actuals(drivers, cust_rev)
    items = expected_items(ctx)
    rng = np.random.default_rng([seed, 2])
    sd = items["item"].map(NOISE_SD).fillna(0.0).to_numpy()
    items["amount"] = items["amount"] * np.exp(rng.normal(0, 1, len(items)) * sd - sd**2 / 2)

    # March bonus payout true-up vs. prior-year accrual (actual only - never in the plan)
    accr = items[items["item"] == "bonus_accrual"].assign(year=lambda x: x["period"].dt.year)
    prior = accr.groupby(["year", "entity", "cost_center"])["amount"].sum().reset_index()
    trueups = []
    for r in prior.itertuples():
        p = pd.Period(f"{r.year + 1}-03", "M")
        if p in ctx.periods:
            trueups.append((p, r.entity, r.cost_center, "6010", "bonus_trueup", r.amount * rng.normal(0.05, 0.10)))
    items = pd.concat([items, pd.DataFrame(trueups, columns=items.columns)], ignore_index=True)
    items["amount"] = items["amount"].round(2)
    return items
