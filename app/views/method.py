import pandas as pd
import streamlit as st
from common import REPO_URL, run_metadata

meta = run_metadata()

st.title("How it works")
st.markdown(
    "A plain-English tour of the pipeline that runs after every month-end close. It is fully automated: a scheduled GitHub "
    "Actions job re-runs everything and this dashboard refreshes itself."
)

steps = [
    ("1 · Extract", "The ERP exports journal-entry lines, the chart of accounts, cost centers, FX rates, the annual budget, HR headcount and CRM ARR. The data is synthetic but shaped like a real NetSuite/SAP extract, messiness included."),
    ("2 · Validate", "Schemas are enforced, every journal entry must balance, and the load is reconciled to the ERP's control totals. Duplicate extract lines are removed, missing cost centers are imputed from each vendor's usual coding, and possible duplicate vendor invoices are reported to AP. Nothing is dropped silently."),
    ("3 · Transform", "Plain SQL models in DuckDB clean the lines, translate foreign-currency entries at monthly average rates, and build a monthly P&L by line and department. The P&L ties back to the ledger to the penny."),
    ("4 · Forecast", "For each of 27 P&L lines, six methods compete: simple baselines, an ARR-driven revenue method, exponential smoothing, ARIMA and a LightGBM model that uses the hiring plan. One-off spikes are dampened before training. Each line's champion is picked on its out-of-sample track record."),
    ("5 · Flag", "Actuals are compared with the budget, last month's forecast and the prior year. Lines are Red when the variance is material and outside the forecast's expected range, and Amber when it's material but expected, a small surprise, or a creeping drift."),
    ("6 · Explain & share", "Each flag gets its top drivers (spend items, customers, postings), a headcount-vs-rate split for payroll, and a first-draft commentary. The Excel variance pack is produced for budget owners."),
]
for row in (steps[:3], steps[3:]):
    cols = st.columns(3)
    for col, (title, text) in zip(cols, row, strict=True):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.caption(text)

st.subheader("Glossary")
glossary = pd.DataFrame(
    [
        ("Rolling forecast", "A forecast that is re-done every month and always looks 12 months ahead, unlike the annual budget, which is set once a year."),
        ("Budget (AOP)", "The annual operating plan approved each November. It is a target, not a forecast."),
        ("WAPE", "Weighted absolute percentage error = total absolute error ÷ total actuals. 8% means forecasts were off by 8 cents per dollar on average."),
        ("Expected range", "The band the forecast expects the actual to land in 95% of the time, built from the model's own past errors."),
        ("Material", "Big enough to matter: over both a dollar threshold (e.g. $50k) and a percentage threshold (e.g. 10%)."),
        ("Favourable / unfavourable", "Good or bad for profit: revenue above budget is favourable, costs above budget are unfavourable."),
        ("Backtest", "Replaying past month-ends to see how a method would have performed, so it's judged on data it never saw."),
        ("ARR", "Annual recurring revenue: the yearly value of all active subscriptions. Monthly subscription revenue ≈ ARR ÷ 12."),
    ],
    columns=["Term", "Meaning"],
)
st.dataframe(glossary, hide_index=True, use_container_width=True)

st.subheader("This run")
dq, rec = meta.get("data_quality", {}), meta.get("reconciliation", {})
c = st.columns(4)
c[0].metric("GL lines loaded", f"{dq.get('rows_loaded', 0):,}", border=True)
c[1].metric("Duplicate extract lines removed", dq.get("extract_duplicate_lines", 0), border=True)
c[2].metric("Cost centers imputed", rec.get("cost_centers_imputed", 0), border=True)
c[3].metric("GL → P&L difference", f"${rec.get('gl_to_pnl_mart_diff_usd', 0):,.2f}", border=True)
with st.expander("Raw files (data dictionary)"):
    files = pd.DataFrame(
        [
            ("gl_journal_lines.csv", "General ledger", "One row per journal-entry line: account, cost center, vendor/customer, debit/credit, currency, source"),
            ("gl_control_totals.csv", "General ledger", "The ERP's line counts and debit/credit totals per period and entity, used for reconciliation"),
            ("chart_of_accounts.csv", "Master data", "Account → P&L line mapping"),
            ("cost_centers.csv", "Master data", "Cost center → department, function, budget owner"),
            ("vendors.csv / entities.csv", "Master data", "Vendor master and legal entities (US, UK, Germany)"),
            ("fx_rates.csv", "Treasury", "Monthly average GBP and EUR to USD rates"),
            ("budget_fyYYYY.csv", "FP&A", "Annual operating plan by month, account and cost center"),
            ("hris_headcount*.csv", "HR system", "Actual headcount, hires and exits; approved hiring plan"),
            ("crm_arr.csv", "CRM", "Monthly ARR bridge: new, expansion, contraction, churn"),
        ],
        columns=["File", "Source", "Contents"],
    )
    st.dataframe(files, hide_index=True, use_container_width=True)

st.subheader("Built with (all free)")
st.markdown(
    "Python · pandas · DuckDB (SQL) · pandera (data validation) · statsforecast & mlforecast (Nixtla) · LightGBM · "
    "statsmodels · Plotly · Streamlit Community Cloud · GitHub Actions (CI + monthly schedule)"
)
st.markdown(f"Source code, notebooks and documentation: [GitHub repository]({REPO_URL})")
