-- P&L lines with reporting attributes. Sign convention: revenue and costs both positive.
CREATE OR REPLACE TABLE fct_gl_pnl_lines AS
SELECT
    g.period,
    g.je_id,
    g.line_no,
    g.posting_date,
    g.entity,
    g.account,
    a.account_name,
    a.account_type,
    a.fs_line,
    g.cost_center,
    c.cost_center_name,
    c.department,
    c.function,
    c.budget_owner,
    g.vendor_id,
    g.vendor_name,
    g.customer_id,
    g.document_number,
    g.description,
    g.source,
    g.is_reversal,
    g.cost_center_imputed,
    g.currency,
    g.amount_local,
    CASE WHEN a.account_type = 'Revenue' THEN -g.amount_usd ELSE g.amount_usd END AS amount_usd
FROM stg_gl_lines g
JOIN raw_chart_of_accounts a ON a.account = g.account
LEFT JOIN raw_cost_centers c ON c.cost_center = g.cost_center
WHERE a.fs_line <> '';
