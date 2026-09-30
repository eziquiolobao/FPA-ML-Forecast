-- Clean GL lines: remove extract duplicates, standardise text, impute missing cost centers,
-- and translate local-currency amounts to USD at the monthly average rate.
CREATE OR REPLACE TABLE stg_gl_lines AS
WITH deduped AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY je_id, line_no ORDER BY je_id) AS rn
    FROM raw_gl
),
lines AS (SELECT * FROM deduped WHERE rn = 1),
valid_cc AS (SELECT cost_center FROM raw_cost_centers),
-- each vendor/account's usual cost center, used to impute missing or invalid coding
coding_history AS (
    SELECT vendor_id, account, mode(cost_center) AS usual_cost_center
    FROM lines
    WHERE cost_center IN (SELECT cost_center FROM valid_cc) AND vendor_id <> ''
    GROUP BY vendor_id, account
)
SELECT
    l.je_id,
    l.line_no,
    l.period,
    l.posting_date,
    l.entity,
    l.account,
    CASE
        WHEN l.cost_center IN (SELECT cost_center FROM valid_cc) THEN l.cost_center
        WHEN a.fs_line <> '' THEN h.usual_cost_center
    END AS cost_center,
    (a.fs_line <> '' AND l.cost_center NOT IN (SELECT cost_center FROM valid_cc)) AS cost_center_imputed,
    NULLIF(l.vendor_id, '') AS vendor_id,
    COALESCE(v.vendor_name, NULLIF(TRIM(l.vendor_name), '')) AS vendor_name,
    NULLIF(l.customer_id, '') AS customer_id,
    NULLIF(l.document_number, '') AS document_number,
    regexp_replace(TRIM(l.description), '\s+', ' ', 'g') AS description,
    l.debit,
    l.credit,
    l.debit - l.credit AS amount_local,
    l.currency,
    fx.usd_per_unit AS fx_rate,
    ROUND((l.debit - l.credit) * fx.usd_per_unit, 2) AS amount_usd,  -- translated to cents, like the ERP
    l.source,
    l.created_by,
    l.is_reversal
FROM lines l
LEFT JOIN raw_chart_of_accounts a ON a.account = l.account
LEFT JOIN raw_vendors v ON v.vendor_id = l.vendor_id
LEFT JOIN coding_history h ON h.vendor_id = l.vendor_id AND h.account = l.account
LEFT JOIN raw_fx fx ON fx.period = l.period AND fx.currency = l.currency;
