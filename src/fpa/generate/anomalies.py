"""Inject realistic accounting anomalies and record them in a hidden answer key.

The pipeline never reads the answer key. It exists only to measure how well the variance
flagging finds real problems (precision / recall).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpa.generate.company import CHART_OF_ACCOUNTS
from fpa.generate.lines import CC_DEPT, CC_ENTITY

ANOMALY_TYPES = {
    "duplicate_invoice": "Vendor invoice posted twice (same invoice number)",
    "missed_accrual": "Month-end accrual not booked; expense lands in the following month",
    "cloud_cost_spike": "Runaway cloud usage (untagged test environment left running)",
    "unbudgeted_contractors": "New contractor engagement started without budget approval",
    "miscoded_cost_center": "Professional-services subcontractor invoice coded to Engineering",
    "retention_bonus": "Unplanned off-cycle retention bonus",
    "revenue_credit_memo": "Large credit memo issued after a billing dispute",
    "legal_settlement": "One-off legal settlement",
    "software_auto_renewal": "Software contract auto-renewed at a higher tier",
    "unapproved_campaign": "Marketing campaign spend without an approved PO",
}
MISSED_ACCRUAL_ITEMS = ["eng_contractors", "ps_subcontractors", "outside_counsel", "finance_consultants"]
DUPLICATE_ITEMS = ["cloud", "eng_contractors", "demand_gen", "ps_subcontractors", "per_seat_saas"]
PERSONNEL_CCS = ["CC-110", "CC-210", "CC-310", "CC-410", "CC-510", "CC-710", "CC-720"]
FS_LINE = CHART_OF_ACCOUNTS.set_index("account")["fs_line"].to_dict()


def _impact(account: str, cc: str, period: pd.Period, direction: str) -> dict:
    return {"fs_line": FS_LINE[account], "department": CC_DEPT[cc], "period": str(period), "direction": direction}


def schedule(periods: pd.PeriodIndex, start: str, seed: int) -> list[dict]:
    """Draw the anomaly calendar for the full simulation horizon (independent of as_of)."""
    rng = np.random.default_rng([seed, 4])
    start_p = pd.Period(start, "M")
    types = list(ANOMALY_TYPES)
    out = []
    for p in periods:
        if p < start_p:
            continue
        for _ in range(min(2, rng.poisson(0.85))):
            t = str(rng.choice(types))
            if t == "missed_accrual" and p + 1 > periods[-1]:
                continue
            scale = 1.18 ** (p.year - 2024)
            a = {"type": t, "period": p, "scale": scale, "u": float(rng.uniform()), "k": int(rng.integers(0, 1000))}
            if t == "missed_accrual":
                a["item"] = MISSED_ACCRUAL_ITEMS[a["k"] % len(MISSED_ACCRUAL_ITEMS)]
            if t == "duplicate_invoice":
                a["item"] = DUPLICATE_ITEMS[a["k"] % len(DUPLICATE_ITEMS)]
            out.append(a)
    for i, a in enumerate(out, start=1):
        a["anomaly_id"] = f"A{i:03d}"
    return out


def apply_to_items(items: pd.DataFrame, anomalies: list[dict]) -> tuple[pd.DataFrame, set, list[dict], list[dict]]:
    """Apply item-level anomalies. Returns (items, skip_accruals, duplicate_requests, answer_key)."""
    items = items.copy()
    extra, skip, dups, key = [], set(), [], []

    def add(p, cc, account, item, amount):
        extra.append((p, CC_ENTITY[cc], cc, account, item, round(amount, 2)))

    for a in anomalies:
        t, p, s, u, k = a["type"], a["period"], a["scale"], a["u"], a["k"]
        rec = {"anomaly_id": a["anomaly_id"], "type": t, "period": str(p), "description": ANOMALY_TYPES[t]}
        if t == "duplicate_invoice":
            dups.append(a)
            continue  # impacts filled in once the GL exists
        if t == "missed_accrual":
            cc = items.loc[(items["period"] == p) & (items["item"] == a["item"]), "cost_center"].iloc[0]
            skip.add((p, cc, a["item"]))
            acct = "6700" if a["item"] == "outside_counsel" else "6200"
            rec["impacts"] = [_impact(acct, cc, p, "under"), _impact(acct, cc, p + 1, "over")]
            rec["detail"] = f"item={a['item']}"
        elif t == "cloud_cost_spike":
            m = (items["period"] == p) & (items["item"] == "cloud")
            base = float(items.loc[m, "amount"].sum())
            items.loc[m, "amount"] *= 1.25 + 0.2 * u
            rec["impacts"] = [_impact("5000", "CC-810", p, "over")]
            rec["approx_usd"] = round(base * (0.25 + 0.2 * u))
        elif t == "unbudgeted_contractors":
            cc = ["CC-410", "CC-610", "CC-720"][k % 3]
            amt = (60_000 + 50_000 * u) * s
            for j in range(3):
                add(p + j, cc, "6200", "unplanned_contractors", amt)
            rec["impacts"] = [_impact("6200", cc, p + j, "over") for j in range(3)]
            rec["approx_usd"] = round(amt)
        elif t == "miscoded_cost_center":
            m = (items["period"] == p) & (items["item"] == "ps_subcontractors")
            items.loc[m, "cost_center"] = "CC-410"
            rec["impacts"] = [_impact("6200", "CC-410", p, "over"), _impact("6200", "CC-720", p, "under")]
            rec["approx_usd"] = round(float(items.loc[m, "amount"].sum()))
        elif t == "retention_bonus":
            cc = PERSONNEL_CCS[k % len(PERSONNEL_CCS)]
            amt = (90_000 + 110_000 * u) * s
            add(p, cc, "6010", "retention_bonus", amt)
            rec["impacts"] = [_impact("6010", cc, p, "over")]
            rec["approx_usd"] = round(amt)
        elif t == "revenue_credit_memo":
            amt = (250_000 + 200_000 * u) * s
            add(p, "CC-000", "4000", "credit_memo", -amt)
            rec["impacts"] = [_impact("4000", "CC-000", p, "under")]
            rec["approx_usd"] = round(amt)
        elif t == "legal_settlement":
            amt = (200_000 + 200_000 * u) * s
            add(p, "CC-630", "6700", "legal_settlement", amt)
            rec["impacts"] = [_impact("6700", "CC-630", p, "over")]
            rec["approx_usd"] = round(amt)
        elif t == "software_auto_renewal":
            amt = (90_000 + 90_000 * u) * s
            add(p, "CC-640", "6300", "auto_renewal", amt)
            rec["impacts"] = [_impact("6300", "CC-640", p, "over")]
            rec["approx_usd"] = round(amt)
        elif t == "unapproved_campaign":
            amt = (120_000 + 130_000 * u) * s
            add(p, "CC-210", "6400", "unapproved_campaign", amt)
            rec["impacts"] = [_impact("6400", "CC-210", p, "over")]
            rec["approx_usd"] = round(amt)
        key.append(rec)

    if extra:
        items = pd.concat([items, pd.DataFrame(extra, columns=items.columns)], ignore_index=True)
    return items, skip, dups, key


def apply_duplicates(gl: pd.DataFrame, dups: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Re-post the largest matching AP invoice as a second JE with the same invoice number."""
    new_rows, key = [], []
    for n, a in enumerate(dups, start=1):
        p = a["period"]
        cand = gl[(gl["period"] == p) & (gl["source"] == "AP") & (gl["_item"] == a["item"]) & (gl["debit"] > 0)]
        if cand.empty:
            continue
        je = cand.loc[cand["debit"].idxmax(), "je_id"]
        rows = gl[gl["je_id"] == je].copy()
        rows["je_id"] = f"JE{p.year}{p.month:02d}9{n:04d}"
        last_day = p.to_timestamp(how="end").normalize()
        rows["posting_date"] = min(rows["posting_date"].iloc[0] + pd.Timedelta(days=4), last_day)
        rows["created_by"] = "ap.clerk2"
        new_rows.append(rows)
        exp = rows[rows["debit"] > 0].iloc[0]
        key.append(
            {
                "anomaly_id": a["anomaly_id"],
                "type": "duplicate_invoice",
                "period": str(p),
                "description": ANOMALY_TYPES["duplicate_invoice"],
                "detail": f"{exp['vendor_id']} invoice {exp['document_number']} re-posted as {rows['je_id'].iloc[0]}",
                "impacts": [_impact(exp["account"], exp["cost_center"], p, "over")],
                "approx_usd": round(float(rows["debit"].sum())),
            }
        )
    if new_rows:
        gl = pd.concat([gl, *new_rows], ignore_index=True)
    return gl, key
