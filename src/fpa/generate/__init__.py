"""Synthetic ERP data generator for a fictional B2B SaaS company."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fpa.generate import anomalies as anom
from fpa.generate.budget import build_budgets
from fpa.generate.company import COST_CENTERS, master_tables
from fpa.generate.drivers import simulate_drivers
from fpa.generate.gl import build_gl
from fpa.generate.lines import actual_items, context_from_actuals, customer_revenue
from fpa.generate.messiness import add_messiness, control_totals, to_erp_csv_format


def generate_raw(settings: dict, as_of: pd.Period, out_dir: Path) -> dict:
    """Simulate the business and write the ERP extracts as they would look at `as_of` close."""
    sim = settings["simulation"]
    seed = int(sim["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    drivers = simulate_drivers(sim["history_start"], sim["simulation_end"], seed)
    cust_rev = customer_revenue(drivers)
    items = actual_items(drivers, cust_rev, seed)
    schedule = anom.schedule(drivers.periods, sim["anomalies_start"], seed)
    items, skip, dup_requests, key = anom.apply_to_items(items, schedule)

    gl = build_gl(items, cust_rev, drivers.headcount, as_of, seed, skip)
    gl, dup_key = anom.apply_duplicates(gl, [a for a in dup_requests if a["period"] <= as_of])
    gl = gl.sort_values(["period", "je_id", "line_no"], kind="stable").reset_index(drop=True)
    controls = control_totals(gl)
    gl, dq_stats = add_messiness(gl, seed)

    to_erp_csv_format(gl).to_csv(out_dir / "gl_journal_lines.csv", index=False)
    controls.assign(period=controls["period"].astype(str)).to_csv(out_dir / "gl_control_totals.csv", index=False)
    for name, df in master_tables().items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    upto = lambda df: df[df["period"] <= as_of].assign(period=lambda x: x["period"].astype(str))  # noqa: E731
    upto(drivers.fx).to_csv(out_dir / "fx_rates.csv", index=False)
    upto(drivers.arr_bridge).to_csv(out_dir / "crm_arr.csv", index=False)
    hris = drivers.headcount.merge(COST_CENTERS[["cost_center", "department"]], on="cost_center")
    upto(hris[["period", "cost_center", "department", "headcount", "hires", "terminations"]]).to_csv(
        out_dir / "hris_headcount.csv", index=False
    )

    # Budgets are released each November for the following fiscal year
    ctx = context_from_actuals(drivers, cust_rev)
    budgets, hc_plan = build_budgets(drivers, ctx, seed, as_of.year + 1)
    cutoff = as_of.to_timestamp(how="end").date()
    for old in out_dir.glob("budget_fy*.csv"):
        old.unlink()
    released = budgets[budgets["released_on"] <= cutoff]
    for fy, b in released.groupby("fiscal_year"):
        b.assign(period=b["period"].astype(str)).to_csv(out_dir / f"budget_fy{fy}.csv", index=False)
    hp = hc_plan[hc_plan["released_on"] <= cutoff]
    hp.assign(period=hp["period"].astype(str)).to_csv(out_dir / "hris_headcount_plan.csv", index=False)

    answer = sorted(
        [k for k in key + dup_key if pd.Period(k["period"], "M") <= as_of],
        key=lambda k: (k["period"], k["anomaly_id"]),
    )
    for k in answer:  # an impact beyond as_of has not happened yet
        k["impacts"] = [i for i in k["impacts"] if pd.Period(i["period"], "M") <= as_of]
    (out_dir / "_answer_key.json").write_text(
        json.dumps({"as_of": str(as_of), "anomalies": answer, "data_quality_injections": dq_stats}, indent=2)
    )
    return {
        "gl_lines": len(gl),
        "journal_entries": gl["je_id"].nunique(),
        "periods": f"{gl['period'].min()}..{gl['period'].max()}",
        "budgets": sorted(released["fiscal_year"].unique().tolist()),
        "anomalies": len(answer),
        **dq_stats,
    }
