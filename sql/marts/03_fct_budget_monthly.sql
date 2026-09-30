-- Budget at the same grain as actuals.
CREATE OR REPLACE TABLE fct_budget_monthly AS
SELECT
    b.fiscal_year,
    b.version,
    b.period,
    a.fs_line,
    c.department,
    c.function,
    a.account_type,
    ROUND(SUM(b.amount_usd), 2) AS budget_usd
FROM stg_budget b
JOIN raw_chart_of_accounts a ON a.account = b.account
JOIN raw_cost_centers c ON c.cost_center = b.cost_center
GROUP BY ALL
ORDER BY b.period, a.account_type DESC, a.fs_line, c.department;
