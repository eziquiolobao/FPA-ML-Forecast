"""Forecast stage of the monthly close: run the tournament, publish the rolling forecast,
score it against the budget and build the full-year landing estimate."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fpa.forecast.backtest import (
    assemble_vintages,
    horizon_bucket,
    leaderboard,
    run_origins,
    select_champions,
    wape,
)
from fpa.forecast.data import load_actuals, load_budget, to_ds


def accuracy_vs_budget(vintages: pd.DataFrame, budget: pd.DataFrame, meta: pd.DataFrame, as_of: pd.Period) -> dict[str, pd.DataFrame]:
    """Out-of-sample accuracy over the last 12 closed months: rolling forecast vs static budget."""
    start, end = (as_of - 12).to_timestamp(), as_of.to_timestamp()
    v = vintages[(vintages["origin"] >= start) & (vintages["origin"] < end) & vintages["y"].notna()]
    champ = v[v["is_champion"]].merge(budget, on=["unique_id", "ds"], how="inner")
    snaive = v[v["model"] == "SeasonalNaive"].merge(budget[["unique_id", "ds"]], on=["unique_id", "ds"])
    targets = champ.drop_duplicates(["unique_id", "ds"])

    by_series = []
    for uid, g in champ.groupby("unique_id"):
        t = targets[targets["unique_id"] == uid]
        sn = snaive[snaive["unique_id"] == uid]
        by_series.append(
            {
                "unique_id": uid,
                "rolling_forecast_wape": wape(g["y"], g["yhat"]),
                "budget_wape": wape(t["y"], t["budget_usd"]),
                "seasonal_naive_wape": wape(sn["y"], sn["yhat"]),
                "actual_12m_usd": float(t["y"].sum()),
            }
        )
    by_series = pd.DataFrame(by_series).merge(meta, on="unique_id")

    champ["bucket"] = horizon_bucket(champ["horizon"])
    by_h = champ.groupby("bucket").apply(lambda g: wape(g["y"], g["yhat"]), include_groups=False).rename("rolling_forecast_wape").reset_index()
    by_h["budget_wape"] = wape(targets["y"], targets["budget_usd"])

    # company-level view: sum lines up to total revenue and total costs before scoring
    champ = champ.merge(meta[["unique_id", "account_type"]], on="unique_id")
    champ["group"] = champ["account_type"].where(champ["account_type"] == "Revenue", "Total costs")
    agg = champ.groupby(["group", "origin", "ds"], as_index=False)[["y", "yhat"]].sum()
    tgt = champ.drop_duplicates(["unique_id", "ds"]).groupby(["group", "ds"], as_index=False)[["y", "budget_usd"]].sum()
    by_group = pd.DataFrame(
        [
            {
                "group": grp,
                "rolling_forecast_wape": wape(agg.loc[agg["group"] == grp, "y"], agg.loc[agg["group"] == grp, "yhat"]),
                "budget_wape": wape(tgt.loc[tgt["group"] == grp, "y"], tgt.loc[tgt["group"] == grp, "budget_usd"]),
            }
            for grp in ("Revenue", "Total costs")
        ]
    )
    overall = pd.DataFrame(
        [
            {
                "window_start": str(as_of - 11),
                "window_end": str(as_of),
                "rolling_forecast_wape": wape(champ["y"], champ["yhat"]),
                "budget_wape": wape(targets["y"], targets["budget_usd"]),
                "seasonal_naive_wape": wape(snaive["y"], snaive["yhat"]),
                "rolling_forecast_1m_wape": wape(champ.loc[champ["horizon"] == 1, "y"], champ.loc[champ["horizon"] == 1, "yhat"]),
            }
        ]
    )
    overall["improvement_vs_budget"] = 1 - overall["rolling_forecast_wape"] / overall["budget_wape"]
    return {"accuracy_by_series": by_series, "accuracy_by_horizon": by_h, "accuracy_by_group": by_group, "accuracy_summary": overall}


def fy_outlook(y: pd.DataFrame, live: pd.DataFrame, budget: pd.DataFrame, meta: pd.DataFrame, as_of: pd.Period) -> pd.DataFrame:
    """Landing estimate = YTD actuals + forecast for the remaining months. At the December close
    the current year is fully actual, so the outlook rolls to next year (12 months of forecast)."""
    fy = as_of.year if as_of.month < 12 else as_of.year + 1
    in_fy = lambda df: df[df["ds"].dt.year == fy]  # noqa: E731
    ytd = in_fy(y).groupby("unique_id")["y"].sum().rename("ytd_actual_usd")
    rest = in_fy(live).groupby("unique_id")["yhat"].sum().rename("remaining_forecast_usd")
    bud = in_fy(budget).groupby("unique_id")["budget_usd"].sum().rename("fy_budget_usd")
    ytd_bud = in_fy(budget[budget["ds"] <= as_of.to_timestamp()]).groupby("unique_id")["budget_usd"].sum().rename("ytd_budget_usd")
    out = pd.concat([ytd, ytd_bud, rest, bud], axis=1).fillna(0.0).reset_index().rename(columns={"index": "unique_id"})
    out["fy_outlook_usd"] = out["ytd_actual_usd"] + out["remaining_forecast_usd"]
    out["outlook_vs_budget_usd"] = out["fy_outlook_usd"] - out["fy_budget_usd"]
    out["fiscal_year"] = fy
    out["months_actual"] = as_of.month if fy == as_of.year else 0
    return out.merge(meta, on="unique_id")


def run_forecast_stage(marts_dir: Path, as_of: pd.Period, cfg: dict, n_jobs: int = 1, log: Callable[[str], None] = print) -> dict:
    horizon, levels = int(cfg["horizon"]), list(cfg["levels"])
    y, meta = load_actuals(marts_dir, as_of)
    budget = load_budget(marts_dir)
    origins = [as_of - k for k in range(int(cfg["backtest_origins"]), -1, -1)]
    log(f"Backtesting {len(origins) - 1} origins + live forecast at {as_of} ...")
    arr = pd.read_parquet(marts_dir / "fct_arr.parquet").assign(ds=lambda d: to_ds(d["period"]))
    preds, importance, cleaned = run_origins(y, meta, marts_dir, origins, horizon, arr, n_jobs, log)
    vintages, champ_hist = assemble_vintages(preds, cleaned, levels, horizon, int(cfg["min_prior_origins"]))

    live_ts = as_of.to_timestamp()
    live = vintages[(vintages["origin"] == live_ts) & vintages["is_champion"]].merge(budget, on=["unique_id", "ds"], how="left")
    champions = select_champions(preds[(preds["origin"] < live_ts) & (preds["ds"] <= live_ts) & preds["y"].notna()])
    lb = leaderboard(preds, live_ts)
    acc = accuracy_vs_budget(vintages, budget, meta, as_of)
    outlook = fy_outlook(y, live, budget, meta, as_of)

    tables = {
        "series_meta": meta,
        "forecast_vintages": vintages,
        "cleaned_history": cleaned[cleaned["origin"] == live_ts].drop(columns="origin"),
        "champion_history": champ_hist,
        "champions": champions.merge(meta, on="unique_id"),
        "leaderboard": lb,
        "forecast_live": live.merge(meta, on="unique_id"),
        "fy_outlook": outlook,
        "lgb_importance": importance.rename("gain_share").rename_axis("feature").reset_index(),
        **acc,
    }
    for name, df in tables.items():
        df.to_parquet(marts_dir / f"{name}.parquet", index=False)
    s = acc["accuracy_summary"].iloc[0]
    return {
        "champion_mix": champions["champion"].value_counts().to_dict(),
        "rolling_forecast_wape": round(float(s["rolling_forecast_wape"]), 4),
        "budget_wape": round(float(s["budget_wape"]), 4),
        "improvement_vs_budget": round(float(s["improvement_vs_budget"]), 4),
    }
