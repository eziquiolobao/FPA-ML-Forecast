"""Variance stage of the monthly close: compare actuals with budget, last forecast and prior
year, flag what needs attention, explain it, and score the flags against the answer key."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from fpa.variance import rules as R
from fpa.variance.commentary import write_commentary
from fpa.variance.drilldown import headcount_rate_split, top_drivers

WINDOW_MONTHS = 12


def build_monthly_variances(marts_dir: Path, as_of: pd.Period, rules: R.Rules) -> pd.DataFrame:
    meta = pd.read_parquet(marts_dir / "series_meta.parquet")
    act = pd.read_parquet(marts_dir / "fct_monthly_pnl.parquet")
    act["unique_id"] = act["fs_line"] + " | " + act["department"]
    act = act.groupby(["unique_id", "period"], as_index=False)["actual_usd"].sum()
    bud = pd.read_parquet(marts_dir / "fct_budget_monthly.parquet")
    bud["unique_id"] = bud["fs_line"] + " | " + bud["department"]
    bud = bud.groupby(["unique_id", "period"], as_index=False)["budget_usd"].sum()

    lv = rules.surprise_level
    vint = pd.read_parquet(marts_dir / "forecast_vintages.parquet")
    last_fc = vint[vint["is_champion"] & (vint["horizon"] == 1)].copy()
    last_fc["period"] = last_fc["ds"].dt.strftime("%Y-%m")
    last_fc = last_fc[["unique_id", "period", "model", "yhat", f"lo{lv}", f"hi{lv}"]].rename(
        columns={"model": "forecast_model", "yhat": "forecast_usd", f"lo{lv}": "range_lo_usd", f"hi{lv}": "range_hi_usd"}
    )

    # history needed for drift detection, reporting window = last 12 closed months
    periods = [str(as_of - k) for k in range(WINDOW_MONTHS + rules.drift_months, -1, -1)]
    grid = pd.MultiIndex.from_product([meta["unique_id"], periods], names=["unique_id", "period"]).to_frame(index=False)
    df = grid.merge(meta, on="unique_id").merge(act, on=["unique_id", "period"], how="left")
    df["actual_usd"] = df["actual_usd"].fillna(0.0)
    df = df.merge(bud, on=["unique_id", "period"], how="left").merge(last_fc, on=["unique_id", "period"], how="left")
    py = act.assign(period=(pd.PeriodIndex(act["period"], freq="M") + 12).astype(str)).rename(columns={"actual_usd": "prior_year_usd"})
    df = df.merge(py, on=["unique_id", "period"], how="left")

    sign = R.favourable_sign(df["account_type"])
    abs_usd, pct = rules.thresholds(df["fs_line"])
    df["var_budget_usd"] = df["actual_usd"] - df["budget_usd"]
    df["var_budget_pct"] = df["var_budget_usd"] / df["budget_usd"].abs()
    df["var_forecast_usd"] = df["actual_usd"] - df["forecast_usd"]
    df["var_forecast_pct"] = df["var_forecast_usd"] / df["forecast_usd"].abs()
    df["var_yoy_usd"] = df["actual_usd"] - df["prior_year_usd"]
    df["var_yoy_pct"] = df["var_yoy_usd"] / df["prior_year_usd"].abs()
    df["favourable"] = (df["var_budget_usd"] * sign) >= 0
    df["impact_usd"] = df["var_budget_usd"] * sign  # + good for the P&L, - bad

    a, b = df["actual_usd"].to_numpy(), df["budget_usd"].to_numpy()
    df["material"] = R.is_material(a - b, b, abs_usd, pct)
    df["outside_range"] = R.outside_range(a, df["range_lo_usd"].to_numpy(), df["range_hi_usd"].to_numpy())
    df["unexpected_minor"] = df["outside_range"] & ~df["material"] & (np.abs(df["var_forecast_usd"].fillna(0)) >= abs_usd / 2)

    # drift catches creeping variances below monthly materiality: half the $ floor, full % test
    creeping = (np.abs(df["var_budget_pct"].fillna(0)) >= pct) & (np.abs(df["var_budget_usd"].fillna(0)) >= abs_usd / 2)
    breach = np.sign(df["var_budget_usd"].fillna(0)) * creeping
    df["drift_months"] = breach.groupby(df["unique_id"]).transform(R.drift_run_length).astype(int)
    df["drift"] = df["drift_months"] >= rules.drift_months
    df["severity"] = R.severity(df["material"].to_numpy(), df["outside_range"].to_numpy(), df["unexpected_minor"].to_numpy(), df["drift"].to_numpy())

    window = {str(as_of - k) for k in range(WINDOW_MONTHS)}
    return df[df["period"].isin(window)].reset_index(drop=True)


def explain(df: pd.DataFrame, marts_dir: Path) -> pd.DataFrame:
    detail = pd.read_parquet(marts_dir / "gl_detail_recent.parquet")
    hc = pd.read_parquet(marts_dir / "fct_headcount.parquet").set_index(["period", "department"])
    comments, drivers_json, splits = [], [], []
    for _, row in df.iterrows():
        drv, split = None, None
        if row["severity"] != R.GREEN:
            drv = top_drivers(detail, row["fs_line"], row["department"], row["period"])
            if row["fs_line"] == "Personnel" and (row["period"], row["department"]) in hc.index:
                h = hc.loc[(row["period"], row["department"])]
                split = headcount_rate_split(row["actual_usd"], row["budget_usd"], float(h["headcount"]), float(h["planned_headcount"]))
        comments.append(write_commentary(row, drv, split) if row["severity"] != R.GREEN else "")
        drivers_json.append(drv.to_json(orient="records") if drv is not None else "[]")
        splits.append(json.dumps(split) if split else "")
    return df.assign(commentary=comments, drivers=drivers_json, personnel_split=splits)


def build_ytd(marts_dir: Path, as_of: pd.Period, rules: R.Rules) -> pd.DataFrame:
    meta = pd.read_parquet(marts_dir / "series_meta.parquet")
    fy = str(as_of.year)
    act = pd.read_parquet(marts_dir / "fct_monthly_pnl.parquet")
    bud = pd.read_parquet(marts_dir / "fct_budget_monthly.parquet")
    keep = lambda d: d[(d["period"].str[:4] == fy) & (d["period"] <= str(as_of))]  # noqa: E731
    a = keep(act).groupby(["fs_line", "department"])["actual_usd"].sum()
    b = keep(bud).groupby(["fs_line", "department"])["budget_usd"].sum()
    df = pd.concat([a.rename("ytd_actual_usd"), b.rename("ytd_budget_usd")], axis=1).reset_index()
    df["unique_id"] = df["fs_line"] + " | " + df["department"]
    df = df.merge(meta[["unique_id", "function", "account_type"]], on="unique_id")
    abs_usd, pct = rules.thresholds(df["fs_line"])
    df["var_usd"] = df["ytd_actual_usd"] - df["ytd_budget_usd"]
    df["var_pct"] = df["var_usd"] / df["ytd_budget_usd"].abs()
    df["favourable"] = (df["var_usd"] * R.favourable_sign(df["account_type"])) >= 0
    df["material"] = R.is_material(df["var_usd"].to_numpy(), df["ytd_budget_usd"].to_numpy(), abs_usd * rules.ytd_multiplier, pct)
    df["fiscal_year"] = int(fy)
    df["through_period"] = str(as_of)
    return df


def evaluate_against_answer_key(flags: pd.DataFrame, answer_key_path: Path) -> tuple[pd.DataFrame, dict]:
    """Precision/recall of the flags versus the hidden list of injected anomalies."""
    if not answer_key_path.exists():
        return pd.DataFrame(), {}
    key = json.loads(answer_key_path.read_text())
    window = set(flags["period"])
    status = flags.set_index(["unique_id", "period"])["severity"].to_dict()
    impacted: set = set()
    rows = []
    for a in key["anomalies"]:
        imp = [(f"{i['fs_line']} | {i['department']}", i["period"]) for i in a["impacts"] if i["period"] in window]
        if not imp:
            continue
        impacted.update(imp)
        sev = [status.get(k, R.GREEN) for k in imp]
        best = R.RED if R.RED in sev else (R.AMBER if R.AMBER in sev else R.GREEN)
        rows.append(
            {
                "anomaly_id": a["anomaly_id"], "type": a["type"], "period": a["period"], "description": a["description"],
                "series": ", ".join(sorted({k[0] for k in imp})), "approx_usd": a.get("approx_usd"),
                "best_flag": best, "caught_red": best == R.RED, "caught_red_or_amber": best != R.GREEN,
            }
        )
    ev = pd.DataFrame(rows)
    red = flags[flags["severity"] == R.RED]
    red_or_amber = flags[flags["severity"] != R.GREEN]
    hit = lambda d: sum((u, p) in impacted for u, p in zip(d["unique_id"], d["period"], strict=False))  # noqa: E731
    summary = {
        "window": f"{min(window)}..{max(window)}",
        "anomalies_in_window": len(ev),
        "red_flags": len(red),
        "recall_red": round(float(ev["caught_red"].mean()), 3) if len(ev) else None,
        "precision_red": round(hit(red) / len(red), 3) if len(red) else None,
        "recall_red_or_amber": round(float(ev["caught_red_or_amber"].mean()), 3) if len(ev) else None,
        "amber_flags": int((flags["severity"] == R.AMBER).sum()),
        "precision_red_or_amber": round(hit(red_or_amber) / len(red_or_amber), 3) if len(red_or_amber) else None,
    }
    return ev, summary


def run_variance_stage(marts_dir: Path, as_of: pd.Period, rules_path: Path, raw_dir: Path, log: Callable[[str], None] = print) -> dict:
    rules = R.Rules.load(rules_path)
    log("Flagging variances ...")
    flags = explain(build_monthly_variances(marts_dir, as_of, rules), marts_dir)
    ytd = build_ytd(marts_dir, as_of, rules)
    ev, ev_summary = evaluate_against_answer_key(flags, raw_dir / "_answer_key.json")
    flags.to_parquet(marts_dir / "variance_monthly.parquet", index=False)
    ytd.to_parquet(marts_dir / "variance_ytd.parquet", index=False)
    if not ev.empty:
        ev.to_parquet(marts_dir / "flag_evaluation.parquet", index=False)
    cur = flags[flags["period"] == str(as_of)]
    summary = {
        "current_month": {s: int((cur["severity"] == s).sum()) for s in (R.RED, R.AMBER, R.GREEN)},
        "evaluation": ev_summary,
    }
    log(f"  {as_of}: {summary['current_month']}")
    if ev_summary:
        log(
            f"  last 12 months vs answer key: recall {ev_summary['recall_red']:.0%} (Red), "
            f"precision {ev_summary['precision_red']:.0%} - {ev_summary['red_flags']} Red flags, "
            f"{ev_summary['anomalies_in_window']} injected anomalies"
        )
    return summary
