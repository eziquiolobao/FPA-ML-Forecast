"""Explode monthly P&L amounts into balanced, ERP-style journal-entry lines.

Mechanics modelled after a typical NetSuite/SAP close:
- subscription revenue is recognised by customer from deferred revenue (REVREC)
- vendor spend arrives as AP invoices; services received but not yet billed are ACCRUED at
  month-end and reversed on day 1 of the next month when the invoice lands
- payroll posts twice a month, bonuses and commissions are accrued
- quarter-end MANUAL reclasses correct miscoded software licences
Every period uses its own seeded RNG, so extending `as_of` never changes history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpa.generate.company import VENDORS
from fpa.generate.lines import ENTITY_CCY, FX_CONST

ITEM_VENDORS = {
    "cloud": [("V1001", 0.8), ("V1002", 0.2)],
    "monitoring": [("V1003", 1.0)],
    "eng_contractors": [("V2001", 0.6), ("V2002", 0.4)],
    "data_contractors": [("V2002", 1.0)],
    "ps_subcontractors": [("V2003", 1.0)],
    "finance_consultants": [("V2004", 1.0)],
    "unplanned_contractors": [("V2005", 1.0)],
    "per_seat_saas": [("V3001", 0.3), ("V3002", 0.2), ("V3003", 0.15), ("V3004", 0.2), ("V3007", 0.15)],
    "annual_renewals": [("V3001", 0.45), ("V3004", 0.25), ("V3006", 0.15), ("V3007", 0.15)],
    "auto_renewal": [("V3001", 1.0)],
    "dev_tools": [("V3005", 0.7), ("V3006", 0.3)],
    "demand_gen": [("V4001", 0.4), ("V4002", 0.35), ("V4004", 0.25)],
    "unapproved_campaign": [("V4001", 1.0)],
    "trade_show": [("V4005", 1.0)],
    "user_conference": [("V4003", 1.0)],
    "field_events": [("V4003", 1.0)],
    "rent": [("V6001", 1.0)],
    "facility_services": [("V6002", 1.0)],
    "audit": [("V7001", 1.0)],
    "tax_advisory": [("V7003", 1.0)],
    "outside_counsel": [("V7002", 1.0)],
    "legal_settlement": [("V7002", 1.0)],
    "recruiting": [("V8001", 0.7), ("V8002", 0.3)],
}
ITEM_DESC = {
    "cloud": "Cloud compute & storage", "monitoring": "Monitoring & observability",
    "eng_contractors": "Engineering contractors", "data_contractors": "Data engineering contractors",
    "ps_subcontractors": "Implementation subcontractors", "finance_consultants": "Finance consulting",
    "unplanned_contractors": "Contract staff augmentation", "per_seat_saas": "SaaS seat licences",
    "annual_renewals": "Annual software renewal", "auto_renewal": "Enterprise tier renewal",
    "dev_tools": "Developer tooling", "demand_gen": "Paid media & demand gen",
    "unapproved_campaign": "Brand campaign", "trade_show": "SaaS Summit booth & sponsorship",
    "user_conference": "Altair Connect user conference", "field_events": "Field events & webinars",
    "rent": "Office rent", "facility_services": "Cleaning & facility services", "audit": "Annual financial audit",
    "tax_advisory": "Tax advisory services", "outside_counsel": "Outside legal counsel",
    "legal_settlement": "Settlement - contract dispute", "recruiting": "Recruiting fees & job boards",
}
ACCRUAL_SHARE = {
    "cloud": 1.0, "monitoring": 1.0, "eng_contractors": 0.6, "data_contractors": 0.6,
    "ps_subcontractors": 0.6, "finance_consultants": 0.5, "audit": 0.5, "outside_counsel": 0.5,
    "tax_advisory": 0.3,
}
LATE_PRONE = {"per_seat_saas", "demand_gen", "facility_services", "recruiting", "field_events"}
TE_CATEGORIES = ["Airfare", "Hotel", "Meals", "Ground transport", "Client entertainment", "Conference fees"]
PAYROLL_ITEMS = {"salaries", "payroll_taxes"}
VENDOR_NAME = VENDORS.set_index("vendor_id")["vendor_name"].to_dict()

GL_COLUMNS = [
    "je_id", "line_no", "period", "posting_date", "entity", "account", "cost_center", "vendor_id",
    "vendor_name", "customer_id", "document_number", "description", "debit", "credit", "currency",
    "source", "created_by", "is_reversal", "_item",
]


class _Writer:
    def __init__(self, p: pd.Period):
        self.p = p
        self.seq = 0
        self.rows: list[tuple] = []

    def je(self, entity, source, created_by, posting_date, lines, is_reversal=False, item=""):
        """lines = [(account, cost_center, signed_amount, vendor_id, customer_id, doc, desc)].
        The last line is the balancing line and is recomputed so the JE balances to the cent."""
        self.seq += 1
        je_id = f"JE{self.p.year}{self.p.month:02d}{self.seq:05d}"
        body = [(a, cc, round(amt, 2), v, c, d, ds) for a, cc, amt, v, c, d, ds in lines[:-1]]
        bal = lines[-1]
        body.append((bal[0], bal[1], -round(sum(x[2] for x in body), 2), *bal[3:]))
        ccy = ENTITY_CCY[entity]
        for n, (acct, cc, amt, vend, cust, doc, desc) in enumerate(body, start=1):
            if amt == 0:
                continue
            self.rows.append(
                (
                    je_id, n, self.p, posting_date, entity, acct, cc, vend, VENDOR_NAME.get(vend, ""), cust, doc, desc,
                    amt if amt > 0 else 0.0, -amt if amt < 0 else 0.0, ccy, source, created_by, is_reversal, item,
                )
            )


def _day(p: pd.Period, day: int) -> pd.Timestamp:
    return pd.Timestamp(year=p.year, month=p.month, day=min(day, p.days_in_month))


def _split(total: float, weights, rng) -> np.ndarray:
    w = rng.dirichlet(np.asarray(weights, dtype=float) * 20) if len(weights) > 1 else np.array([1.0])
    return total * w


def _ap_invoices(w: _Writer, rng, entity, account, cc, item, amount, post_period: pd.Period, service_period: pd.Period, prefix=""):
    for vendor, share in ITEM_VENDORS[item]:
        v_amt = amount * share
        n_inv = 1 if abs(v_amt) < 20_000 else int(rng.integers(1, 4))
        for k, part in enumerate(_split(v_amt, [1] * n_inv, rng), start=1):
            doc = f"INV-{vendor[1:]}-{service_period.year % 100:02d}{service_period.month:02d}-{k}{int(rng.integers(100, 999))}"
            desc = f"{prefix}{ITEM_DESC[item]} - {service_period.strftime('%b %Y')}"
            w.je(
                entity, "AP", f"ap.clerk{int(rng.integers(1, 3))}", _day(post_period, int(rng.integers(2, 28))),
                [(account, cc, part, vendor, "", doc, desc), ("2000", "", 0, vendor, "", doc, desc)],
                item=item,
            )


def build_gl(
    items: pd.DataFrame,
    cust_rev: pd.DataFrame,
    headcount: pd.DataFrame,
    last_period: pd.Period,
    seed: int,
    skip_accruals: set,
) -> pd.DataFrame:
    items = items[items["period"] <= last_period].copy()
    ccy = items["entity"].map(ENTITY_CCY)
    items["local"] = items["amount"] / ccy.map(FX_CONST)
    cust_rev = cust_rev[cust_rev["period"] <= last_period].copy()
    cust_rev["local"] = cust_rev["revenue_usd"] / cust_rev["entity"].map(ENTITY_CCY).map(FX_CONST)
    hc = headcount.set_index(["period", "cost_center"])["headcount"]
    by_period = dict(tuple(items.groupby("period")))
    rev_by_period = dict(tuple(cust_rev.groupby("period")))
    first = min(by_period)

    out, carry = [], []  # carry: work queued for the next period (reversals + late invoices)
    for p in sorted(by_period):
        rng = np.random.default_rng([seed, 3, p.year, p.month])
        w = _Writer(p)
        df = by_period[p]
        eom = _day(p, 31)

        if p == first:  # opening: reverse the (synthetic) prior-month accruals
            for r in df[df["item"].isin(ACCRUAL_SHARE)].itertuples():
                late = r.local * ACCRUAL_SHARE[r.item]
                carry.append(("reversal", r.entity, r.account, r.cost_center, r.item, late * 0.98, p - 1))
                carry.append(("late_invoice", r.entity, r.account, r.cost_center, r.item, late, p - 1))

        # 1) prior-period work: accrual reversals and invoices that arrived after month-end
        for kind, ent, acct, cc, item, amt, svc in carry:
            if kind == "reversal":
                desc = f"Reversal - accrual {ITEM_DESC[item]} - {svc.strftime('%b %Y')}"
                w.je(ent, "ACCRUAL", "gl.accountant", _day(p, 1), [(acct, cc, -amt, "", "", "", desc), ("2100", "", 0, "", "", "", desc)], is_reversal=True, item=item)
            else:
                prefix = "Late invoice - " if kind == "late_ap" else ""
                _ap_invoices(w, rng, ent, acct, cc, item, amt, p, svc, prefix=prefix)
        carry = []

        # 2) subscription revenue recognition, one JE per entity with a line per customer
        if p in rev_by_period:
            for ent, g in rev_by_period[p].groupby("entity"):
                lines = [
                    ("4000", "CC-000", -r.local, "", r.customer_id, f"RR-{p.year}{p.month:02d}", f"Subscription revenue - {r.customer_name}")
                    for r in g.itertuples()
                ]
                lines.append(("2400", "", 0, "", "", f"RR-{p.year}{p.month:02d}", "Deferred revenue release"))
                w.je(ent, "REVREC", "SYS_REVREC", eom, lines, item="subscription")

        customers = rev_by_period.get(p)
        for (ent, cc), g in df.groupby(["entity", "cost_center"], sort=True):
            amounts = dict(zip(g["item"], g["local"], strict=False))
            accounts = dict(zip(g["item"], g["account"], strict=False))

            if "salaries" in amounts:
                s, t = amounts["salaries"], amounts.get("payroll_taxes", 0.0)
                for day in (15, 31):
                    desc = f"Payroll run {_day(p, day):%m/%d/%Y}"
                    w.je(ent, "PAYROLL", "SYS_PAYROLL", _day(p, day), [("6000", cc, s / 2, "", "", "", desc), ("6020", cc, t / 2, "", "", "", desc), ("1000", "", 0, "", "", "", desc)], item="salaries")
            for item in ("bonus_accrual", "bonus_trueup"):
                if item in amounts:
                    desc = "Bonus accrual" if item == "bonus_accrual" else f"Bonus payout true-up FY{p.year - 1}"
                    w.je(ent, "ACCRUAL", "gl.accountant", eom, [("6010", cc, amounts[item], "", "", "", desc), ("2150", "", 0, "", "", "", desc)], item=item)
            if "retention_bonus" in amounts:
                desc = "Off-cycle payment - retention"
                w.je(ent, "PAYROLL", "SYS_PAYROLL", _day(p, 20), [("6010", cc, amounts["retention_bonus"], "", "", "", desc), ("1000", "", 0, "", "", "", desc)], item="retention_bonus")
            if "commissions" in amounts:
                desc = "Commission accrual - monthly bookings"
                w.je(ent, "PAYROLL", "SYS_PAYROLL", eom, [("6100", cc, amounts["commissions"], "", "", "", desc), ("2160", "", 0, "", "", "", desc)], item="commissions")
            if "depreciation" in amounts:
                desc = "Monthly depreciation run"
                w.je(ent, "FA", "SYS_FIXEDASSETS", eom, [("6800", cc, amounts["depreciation"], "", "", "", desc), ("1590", "", 0, "", "", "", desc)], item="depreciation")
            if "services" in amounts and customers is not None:
                pool = customers[customers["entity"] == ent]
                n = int(rng.integers(3, 9))
                picks = pool.sample(n=min(n, len(pool)), random_state=int(rng.integers(0, 2**31)))
                for part, cust in zip(_split(amounts["services"], [1] * len(picks), rng), picks.itertuples(), strict=False):
                    doc = f"SI-{p.year % 100}{p.month:02d}-{int(rng.integers(1000, 9999))}"
                    desc = f"Implementation milestone - {cust.customer_name}"
                    w.je(ent, "AR", "ar.specialist", _day(p, int(rng.integers(5, 28))), [("4100", "CC-000", -part, "", cust.customer_id, doc, desc), ("1200", "", 0, "", cust.customer_id, doc, desc)], item="services")
            if "credit_memo" in amounts and customers is not None:
                cust = customers[customers["entity"] == ent].nlargest(15, "arr_usd").sample(1, random_state=int(rng.integers(0, 2**31))).iloc[0]
                doc, desc = f"CM-{p.year % 100}{p.month:02d}-{int(rng.integers(100, 999))}", f"Credit memo - billing dispute - {cust.customer_name}"
                w.je(ent, "AR", "ar.specialist", _day(p, 22), [("4000", "CC-000", -amounts["credit_memo"], "", cust.customer_id, doc, desc), ("1200", "", 0, "", cust.customer_id, doc, desc)], item="credit_memo")
            for item in ("travel", "sales_kickoff"):
                if item not in amounts:
                    continue
                if item == "sales_kickoff":
                    desc = f"Sales Kickoff {p.year} - venue, travel & lodging"
                    doc = f"SKO-{p.year}-{cc[-3:]}"
                    w.je(ent, "AP", "SYS_EXPENSE", _day(p, 25), [("6500", cc, amounts[item], "V5001", "", doc, desc), ("2000", "", 0, "V5001", "", doc, desc)], item=item)
                    continue
                heads = int(hc.get((p, cc), 5))
                n_rep = max(1, int(round(heads * 0.8)))
                for part in _split(amounts[item], [1] * n_rep, rng):
                    emp = f"E{cc[-3:]}{int(rng.integers(1, heads + 1)):03d}"
                    cats = rng.choice(TE_CATEGORIES, size=int(rng.integers(1, 5)), replace=False)
                    doc = f"ER-{p.year % 100}{p.month:02d}-{w.seq + 1:05d}"
                    lines = [("6500", cc, x, "V5001", "", doc, f"Expense report {emp} - {c}") for x, c in zip(_split(part, [1] * len(cats), rng), cats, strict=False)]
                    lines.append(("2000", "", 0, "V5001", "", doc, f"Expense report {emp}"))
                    w.je(ent, "AP", "SYS_EXPENSE", _day(p, int(rng.integers(3, 28))), lines, item=item)

            # vendor-invoiced spend with accrual mechanics
            for item, amt in amounts.items():
                if item not in ITEM_VENDORS:
                    continue
                acct = accounts[item]
                share = ACCRUAL_SHARE.get(item, 0.0)
                now = amt * (1 - share)
                if item in LATE_PRONE and rng.random() < 0.03:
                    carry.append(("late_ap", ent, acct, cc, item, now, p))
                elif now:
                    _ap_invoices(w, rng, ent, acct, cc, item, now, p, p)
                if share:
                    late = amt * share
                    carry.append(("late_invoice", ent, acct, cc, item, late, p))
                    if (p, cc, item) not in skip_accruals:
                        est = late * (1 + rng.normal(0, 0.04))
                        desc = f"Accrual - {ITEM_DESC[item]} - {p.strftime('%b %Y')}"
                        w.je(ent, "ACCRUAL", "gl.accountant", _day(p + 1, 2), [(acct, cc, est, "", "", "", desc), ("2100", "", 0, "", "", "", desc)], item=item)
                        carry.append(("reversal", ent, acct, cc, item, est, p))

        # 3) quarter-end reclass of engineering licences miscoded to IT
        if p.month in (3, 6, 9, 12):
            q = df[(df["item"] == "per_seat_saas")]["local"].sum()
            amt = round(3 * 0.08 * q, 2)
            desc = "Reclass: engineering tool licences coded to IT (Q/E true-up)"
            w.je("US01", "MANUAL", "fpa.controller", _day(p + 1, 3), [("6300", "CC-410", amt, "", "", "", desc), ("6300", "CC-640", 0, "", "", "", desc)], item="reclass")

        out.append(pd.DataFrame(w.rows, columns=GL_COLUMNS))
    return pd.concat(out, ignore_index=True)
