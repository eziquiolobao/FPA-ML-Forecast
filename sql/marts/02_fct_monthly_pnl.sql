-- Monthly actuals at the forecast grain: P&L line x department, consolidated in USD.
CREATE OR REPLACE TABLE fct_monthly_pnl AS
SELECT
    period,
    fs_line,
    department,
    function,
    account_type,
    ROUND(SUM(amount_usd), 2) AS actual_usd
FROM fct_gl_pnl_lines
GROUP BY ALL
ORDER BY period, account_type DESC, fs_line, department;
