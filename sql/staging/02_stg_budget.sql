-- Budget (annual operating plan) lines, one row per period x entity x account x cost center.
CREATE OR REPLACE TABLE stg_budget AS
SELECT fiscal_year, version, CAST(released_on AS DATE) AS released_on, period, entity, account,
       cost_center, amount_usd
FROM raw_budget;
