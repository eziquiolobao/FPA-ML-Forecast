# Interview talk track

## The 2-minute version (for any audience)

> "Every month, finance teams close the books and then spend days answering two questions: *where will we land this year?* and *which numbers are off, and why?*
>
> I built a system that does the first pass automatically for a SaaS company. After each month-end close it:
> 1. **pulls the raw general ledger**, cleans it, and rebuilds the P&L. It ties back to the ledger to the penny.
> 2. **re-forecasts the next 12 months** for all 27 P&L lines. Instead of betting on one method, it runs a small tournament for each line and keeps whichever method has actually been most accurate for that line.
> 3. **flags the variances that matter**: lines that are off budget by a material amount *and* weren't predictable from recent trends. Each flag comes with a first-draft explanation and a drill-down to the actual invoices.
>
> The result: the rolling forecast is about **20% more accurate than the annual budget** overall and about **30% more accurate for the next quarter**. To prove the flagging works, I secretly planted accounting problems in the data, like duplicate invoices, missed accruals and runaway cloud spend. The red flags caught **12 of 13**, with about **4 in 5 red flags being real problems**. The analyst ends up reviewing one or two lines a month instead of 27.
>
> It's fully automated and free: a scheduled job re-runs everything each month and the dashboard updates itself."

## The 5-minute technical deep-dive

1. **Data.** A synthetic ERP extract shaped like NetSuite/SAP: about 89k journal lines, chart of accounts, cost centers, FX, budgets, HRIS and CRM. It is messy on purpose: extract duplicates, missing cost centers and inconsistent text. Every journal entry balances, and accruals reverse the following month.
2. **Validation and warehouse.** pandera schemas. Business rules: journal entries balance, and the load reconciles to the ERP's control totals. Plain SQL models in DuckDB (a dbt-style staging → marts layer), with FX translation at monthly average rates. Reconciliation checks run inside the pipeline and fail the close if they break.
3. **Forecasting.** A rolling-origin backtest over 24 month-ends × 12-month horizon × 6 contenders:
   - naive, seasonal naive
   - an ARR run-rate driver model for subscription revenue
   - AutoETS and AutoARIMA (statsforecast)
   - a global LightGBM (mlforecast) with lags, calendar and the headcount plan as a *known future* covariate

   Champion per line by WAPE, with a simplicity tie-break. The champion at each origin is chosen **only with errors observable at that time**, so the accuracy numbers are honestly out of sample.
4. **Normalised actuals.** A robust STL decomposition caps one-off spikes before training (the statistical version of stripping one-offs). Prediction intervals come from each champion's own empirical error distribution per horizon, measured against *cleaned* actuals, so they describe normal variation.
5. **Flagging.** Materiality (dollar AND percentage, configurable per line in YAML) combined with statistical surprise (outside last month's 95% range), plus a drift rule. Drill-down ranks spend items against a typical recent month (median, robust to seasonal spikes). Payroll gets a headcount-vs-rate split. Commentary comes from templates, so it is reproducible and needs no LLM cost.
6. **Evaluation.** Precision and recall against a hidden answer key of injected anomalies.
7. **Engineering.** uv, ruff and pytest (determinism, double entry, penny tie-out, rules and headline claims). GitHub Actions for CI and a monthly cron that commits refreshed marts. The Streamlit Community Cloud app reads only small parquet files.

## Likely questions

**Why not just use one model, like Prophet?**
Different lines behave differently: rent is flat, marketing is seasonal, revenue follows ARR. The tournament showed that. Exponential smoothing wins 13 lines, naive wins rent-like lines, LightGBM wins some payroll lines, and the ARR driver wins revenue. One model everywhere would be worse on most lines.

**How do you know you didn't overfit?**
Every number is from a rolling-origin backtest: the model only sees data up to each month-end, and even the *choice* of champion uses only past errors. The budget comparison uses the same months.

**The data is synthetic. Isn't that cheating?**
The generator is built from FP&A mechanics (ARR bridge, headcount × salary, accrual/reversal cycles, budget built each November with stretch targets). The pipeline never reads the answer key. On real data I would swap the generator for the ERP connector. Everything downstream stays the same.

**What are the weaknesses?**
Costs are under-forecast by a few percent. About 40% of that is one-off spikes, which is what the flags are for. The rest is a hiring acceleration that history-only models lag. Revenue runs low at 9-12 months because a price increase kept lifting ARR as contracts renewed. Fixes: a driver-based personnel model (headcount plan × cost per head), overlaying known pricing and hiring decisions, and bias correction from the tracking signal.

**Why are some flags false positives?**
A false positive is a genuine but unusual swing on a volatile line. It is still worth a 2-minute look. The thresholds are a precision/recall dial that FP&A owns in a YAML file.

**How would this work in production?**
Replace the generator with scheduled extracts (NetSuite SuiteQL / SAP OData), keep DuckDB or move the SQL to Snowflake/BigQuery with dbt, run the pipeline in an orchestrator after close, and publish the variance pack to budget owners. Commentary could be enriched by an LLM once the numbers are computed deterministically.

**Why DuckDB?**
It is free, embedded and fast for analytics, and it's plain SQL, so the same models port to a cloud warehouse.

**What would you add next?**
Hierarchical reconciliation (so department forecasts sum to the company view), scenario planning (pricing and hiring what-ifs), and a feedback loop where analysts mark flags as "explained" to tune the thresholds.
