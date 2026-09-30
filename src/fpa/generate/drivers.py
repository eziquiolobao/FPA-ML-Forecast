"""Simulate the business drivers behind the P&L: FX rates, the customer base (ARR) and headcount.

Everything is simulated for the full horizon with a single seeded RNG, so the history never
changes when the `as_of` date moves forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpa.generate.company import COST_CENTERS, ENTITY_REVENUE_SHARE

# Business storylines (real events, not anomalies)
WHALE_CHURN_PERIOD = pd.Period("2024-07", "M")  # largest customer leaves
PRICE_INCREASE_START = pd.Period("2025-07", "M")  # +8% uplift applied at each renewal from here
HIRING_FREEZE = (pd.Period("2022-11", "M"), pd.Period("2023-03", "M"))

# New-logo bookings as a share of opening ARR, by year (growth decelerates as the company scales)
NEW_LOGO_RATE = {2021: 0.030, 2022: 0.026, 2023: 0.021, 2024: 0.018, 2025: 0.017, 2026: 0.016}
BOOKINGS_SEASONALITY = np.array([0.70, 0.80, 1.15, 0.80, 0.90, 1.15, 0.75, 0.80, 1.15, 0.90, 1.05, 1.85])
BOOKINGS_SEASONALITY = BOOKINGS_SEASONALITY / BOOKINGS_SEASONALITY.mean()

# Annual merit increase applied every April
MERIT_INCREASE = {2021: 0.03, 2022: 0.05, 2023: 0.045}
DEFAULT_MERIT = 0.035

_NAME_A = [
    "Blue", "Pine", "Silver", "North", "Red", "Harbor", "Summit", "Clear", "Iron", "Maple", "Bright",
    "Stone", "River", "Cedar", "Atlas", "Nova", "Crest", "Golden", "Prairie", "Falcon", "Orchid", "Vector",
]
_NAME_B = [
    "field", "ridge", "wave", "point", "bridge", "gate", "stream", "path", "light", "forge", "view",
    "haven", "line", "peak", "works", "shore", "grove", "port", "mark", "stack",
]
_NAME_C = [
    "Logistics", "Health", "Retail", "Energy", "Capital", "Foods", "Robotics", "Media", "Insurance",
    "Labs", "Manufacturing", "Telecom", "Pharma", "Mobility", "Analytics", "Hospitality", "Realty",
]


def periods(start: str, end: str) -> pd.PeriodIndex:
    return pd.period_range(start, end, freq="M")


def new_logo_rate(year: int) -> float:
    return NEW_LOGO_RATE.get(year, max(0.011, 0.016 - 0.001 * (year - 2026)))


def merit_for(year: int) -> float:
    return MERIT_INCREASE.get(year, DEFAULT_MERIT)


@dataclass
class Drivers:
    periods: pd.PeriodIndex
    fx: pd.DataFrame  # period, currency, usd_per_unit
    customer_arr: pd.DataFrame  # period, customer_id, customer_name, entity, arr_usd
    arr_bridge: pd.DataFrame  # period, entity, beginning/new/expansion/contraction/churn/ending ARR
    headcount: pd.DataFrame  # period, cost_center, headcount, hires, terminations, salary_usd


def simulate_fx(pers: pd.PeriodIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    start = {"GBP": 1.36, "EUR": 1.21}
    anchor = {"GBP": 1.30, "EUR": 1.12}
    level = dict(start)
    for p in pers:
        for ccy in ("GBP", "EUR"):
            shock = rng.normal(0, 0.018)
            level[ccy] = level[ccy] * np.exp(shock) + 0.04 * (anchor[ccy] - level[ccy])
            rows.append((p, ccy, round(level[ccy], 6)))
        rows.append((p, "USD", 1.0))
    return pd.DataFrame(rows, columns=["period", "currency", "usd_per_unit"])


_SUFFIXES = ["Inc.", "LLC", "Group", "Co.", "Holdings", "Partners", "Ltd.", "Systems"]


def _customer_names(n: int, rng: np.random.Generator) -> list[str]:
    names, seen = [], set()
    while len(names) < n:
        base = f"{rng.choice(_NAME_A)}{rng.choice(_NAME_B)} {rng.choice(_NAME_C)}"
        name, k = base, 0
        while name in seen:  # disambiguate like real company names do
            name = f"{base} {_SUFFIXES[k % len(_SUFFIXES)]}" + (f" {k // len(_SUFFIXES) + 2}" if k >= len(_SUFFIXES) else "")
            k += 1
        seen.add(name)
        names.append(name)
    return names


def simulate_customers(pers: pd.PeriodIndex, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Customer-level ARR simulation -> (customer ARR by month, ARR bridge by entity)."""
    max_customers = 8000
    arr = np.zeros(max_customers)
    active = np.zeros(max_customers, dtype=bool)
    entity = rng.choice(list(ENTITY_REVENUE_SHARE), size=max_customers, p=list(ENTITY_REVENUE_SHARE.values()))
    renewal_month = rng.integers(1, 13, size=max_customers)
    names = _customer_names(max_customers, rng)

    # Opening book: ~420 customers, $18M ARR, plus one "whale" worth ~$1.8M
    n0 = 420
    sizes = rng.lognormal(mean=np.log(28_000), sigma=1.0, size=n0)
    sizes = sizes / sizes.sum() * 16_200_000
    arr[:n0] = sizes
    active[:n0] = True
    whale = n0
    arr[whale], active[whale], entity[whale] = 1_800_000, True, "US01"
    names[whale] = "Meridian Global Logistics"
    n_used = n0 + 1

    cust_rows, bridge_rows = [], []
    for p in pers:
        begin = arr.copy()
        idx = np.flatnonzero(active)

        # churn (the whale is protected until its storyline month)
        churn_mask = rng.random(idx.size) < 0.0075
        churn_mask[idx == whale] = p == WHALE_CHURN_PERIOD
        churned = idx[churn_mask]

        survivors = idx[~churn_mask]
        expand = survivors[rng.random(survivors.size) < 0.06]
        arr[expand] *= 1 + rng.lognormal(np.log(0.13), 0.5, size=expand.size)
        contract = survivors[rng.random(survivors.size) < 0.02]
        arr[contract] *= 1 - rng.uniform(0.10, 0.30, size=contract.size)
        if p >= PRICE_INCREASE_START:
            renewing = survivors[renewal_month[survivors] == p.month]
            if p < PRICE_INCREASE_START + 12:  # one full renewal cycle of uplifts
                arr[renewing] *= 1.08

        arr[churned] = 0.0
        active[churned] = False

        # new logos
        target = begin.sum() * new_logo_rate(p.year) * BOOKINGS_SEASONALITY[p.month - 1]
        target *= rng.lognormal(0, 0.15)
        median_deal = 30_000 * (1.06 ** (p.year - 2021))
        new_ids = []
        booked = 0.0
        while booked < target and n_used < max_customers:
            deal = float(np.clip(rng.lognormal(np.log(median_deal), 0.9), 6_000, 900_000))
            arr[n_used], active[n_used], renewal_month[n_used] = deal, True, p.month
            new_ids.append(n_used)
            booked += deal
            n_used += 1
        new_ids = np.array(new_ids, dtype=int)

        delta = arr - begin
        for ent in ENTITY_REVENUE_SHARE:
            ent_mask = entity == ent
            new_mask = np.zeros(max_customers, dtype=bool)
            new_mask[new_ids] = True
            churn_m = np.zeros(max_customers, dtype=bool)
            churn_m[churned] = True
            existing = ent_mask & ~new_mask & ~churn_m
            bridge_rows.append(
                (
                    p,
                    ent,
                    begin[ent_mask].sum(),
                    arr[ent_mask & new_mask].sum(),
                    delta[existing & (delta > 0)].sum(),
                    delta[existing & (delta < 0)].sum(),
                    -begin[ent_mask & churn_m].sum(),
                    arr[ent_mask].sum(),
                    int((active & ent_mask).sum()),
                )
            )

        act = np.flatnonzero(active)
        cust_rows.append(
            pd.DataFrame(
                {
                    "period": p,
                    "customer_id": [f"C{i + 1:05d}" for i in act],
                    "customer_name": [names[i] for i in act],
                    "entity": entity[act],
                    "arr_usd": arr[act].round(2),
                }
            )
        )

    bridge = pd.DataFrame(
        bridge_rows,
        columns=[
            "period",
            "entity",
            "beginning_arr",
            "new_arr",
            "expansion_arr",
            "contraction_arr",
            "churn_arr",
            "ending_arr",
            "customers",
        ],
    )
    num = bridge.columns.drop(["period", "entity", "customers"])
    bridge[num] = bridge[num].round(2)
    return pd.concat(cust_rows, ignore_index=True), bridge


def simulate_headcount(pers: pd.PeriodIndex, arr_total: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Headcount follows ARR growth with a lag, attrition, and a hiring freeze."""
    arr0 = arr_total.iloc[0]
    rows = []
    for cc in COST_CENTERS.itertuples():
        if cc.hc_start == 0:
            continue
        hc = cc.hc_start
        salary = float(cc.salary_start_usd)
        for i, p in enumerate(pers):
            if p.month == 4 and p.year > 2021:
                salary *= 1 + merit_for(p.year)
            lagged_arr = arr_total.iloc[max(0, i - 3)]
            target = cc.hc_start * (lagged_arr / arr0) ** cc.hc_elasticity
            terms = rng.binomial(hc, 0.012)
            gap = target - (hc - terms)
            hires = max(0, int(round(gap + rng.normal(0, 0.6))))
            hires = min(hires, max(1, int(0.12 * hc)))
            if HIRING_FREEZE[0] <= p <= HIRING_FREEZE[1]:
                hires = int(rng.binomial(hires, 0.25))
            hc = hc - terms + hires
            rows.append((p, cc.cost_center, hc, hires, terms, round(salary, 2)))
    return pd.DataFrame(rows, columns=["period", "cost_center", "headcount", "hires", "terminations", "salary_usd"])


def simulate_drivers(start: str, end: str, seed: int) -> Drivers:
    pers = periods(start, end)
    rng = np.random.default_rng([seed, 1])
    fx = simulate_fx(pers, rng)
    customer_arr, bridge = simulate_customers(pers, rng)
    arr_total = bridge.groupby("period")["ending_arr"].sum()
    headcount = simulate_headcount(pers, arr_total, rng)
    return Drivers(pers, fx, customer_arr, bridge, headcount)
