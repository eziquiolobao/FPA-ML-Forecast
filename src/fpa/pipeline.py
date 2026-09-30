"""Monthly close pipeline: generate -> validate -> transform -> forecast -> flag -> report.

Usage:
    python -m fpa.pipeline run --as-of 2026-08
    python -m fpa.pipeline run --as-of auto          # last completed calendar month
    python -m fpa.pipeline run --stages forecast,variance
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
STAGES = ["generate", "transform", "forecast", "variance", "report"]


def load_settings(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "config" / "settings.yaml").read_text())


def resolve_as_of(value: str, today: date | None = None) -> pd.Period:
    if value == "auto":
        return pd.Period(today or date.today(), "M") - 1
    return pd.Period(value, "M")


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(as_of: pd.Period, stages: list[str], settings: dict, n_jobs: int = 1) -> dict:
    paths = {k: ROOT / v for k, v in settings["paths"].items()}
    marts = paths["marts"]
    marts.mkdir(parents=True, exist_ok=True)
    summary: dict = {"as_of": str(as_of), "company": settings["company"]["name"]}
    meta_path = marts / "run_metadata.json"
    if meta_path.exists():
        summary = {**json.loads(meta_path.read_text()), **summary}
    t0 = time.time()

    if "generate" in stages:
        from fpa.generate import generate_raw

        _log(f"Generating ERP extracts as of {as_of} close ...")
        summary["generate"] = generate_raw(settings, as_of, paths["raw"])
        _log(f"  {summary['generate']['gl_lines']:,} GL lines, {summary['generate']['anomalies']} anomalies in answer key")

    if "transform" in stages:
        from fpa.ingest import load_raw, validate
        from fpa.transform import build_warehouse

        _log("Validating raw data ...")
        raw = validate(load_raw(paths["raw"]))
        _log(f"  data-quality report: {raw.dq['extract_duplicate_lines']} duplicate extract lines, "
             f"{len(raw.dq['possible_duplicate_invoices'])} possible duplicate invoices")
        _log("Building DuckDB warehouse and marts ...")
        wh = build_warehouse(raw, paths["warehouse"], marts, as_of)
        summary["data_quality"] = raw.dq
        summary["reconciliation"] = wh["reconciliation"]
        _log(f"  reconciliation: {wh['reconciliation']}")

    if "forecast" in stages:
        from fpa.forecast.rolling import run_forecast_stage

        summary["forecast"] = run_forecast_stage(marts, as_of, settings["forecast"], n_jobs=n_jobs, log=_log)
        f = summary["forecast"]
        _log(f"  rolling-forecast WAPE {f['rolling_forecast_wape']:.1%} vs budget {f['budget_wape']:.1%} "
             f"({f['improvement_vs_budget']:+.0%} more accurate)")

    if "variance" in stages:
        from fpa.variance.flagging import run_variance_stage

        summary["variance"] = run_variance_stage(marts, as_of, ROOT / "config" / "variance_rules.yaml", paths["raw"], log=_log)

    if "report" in stages:
        from fpa.reporting.excel_pack import write_variance_pack

        out = write_variance_pack(marts, as_of, paths["reports"], settings["company"]["name"])
        summary["report"] = str(out.relative_to(ROOT))
        _log(f"  variance pack written to {summary['report']}")

    summary["last_run_utc"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    summary["runtime_seconds"] = round(time.time() - t0)
    meta_path.write_text(json.dumps(summary, indent=2, default=str))
    _log(f"Done in {summary['runtime_seconds']}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the monthly close pipeline")
    r.add_argument("--as-of", default=None, help="YYYY-MM or 'auto' (default: settings.yaml)")
    r.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {STAGES}")
    r.add_argument("--n-jobs", type=int, default=1, help="parallel workers for statistical models")
    args = ap.parse_args()

    settings = load_settings()
    as_of = resolve_as_of(args.as_of or str(settings["as_of"]))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")
    run(as_of, stages, settings, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
